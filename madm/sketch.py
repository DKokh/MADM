"""MinHash sketches: "how close is this antibody to what the model was trained on?"

A score is not a property of a model. It is a property of the RELATIONSHIP
between the evaluation set and the training set -- which is why cluster CV, a
random split and a hold-out disagree, and why none of them is the true number.
They sample different parts of one curve: performance against distance from
training. To report that curve, or to let a user place their own molecules on it,
you need the distance. To let them do it WITHOUT the training sequences, you need
a sketch.

WHAT IT IS. Every sequence becomes its set of k-mers. Each k-mer is hashed; only
the MINIMUM hash is kept. Repeat with `n_hashes` independent hash functions and
the result is a short vector of integers. For two sets the probability that their
minima agree is exactly their Jaccard similarity -- the minimum is equally likely
to be any element of the union, and it matches only when that element lies in the
intersection. So

    Jaccard(A, B)  ~=  mean(sketch_A == sketch_B)

with a standard error of sqrt(J(1-J)/n_hashes): +-0.04 at n_hashes=128, which is
ample for binning into 0.7 / 0.8 / 0.9 bands. At 512 bytes per sequence, a
thousand molecules is half a megabyte.

SAME SIMILARITY AS THE SPLITS. k-mer Jaccard with k=4 is what `data.cluster_graph`
uses to decide what counts as leakage, so a sketch distance and a cluster
boundary mean the same thing. Sketching with a different k would quietly answer a
different question.

WHAT IT DOES NOT DO. With k=4 there are only 160k possible k-mers, so anyone can
hash all of them and invert a minimum by lookup. That reveals up to `n_hashes`
k-mers per training sequence, unordered, out of roughly 230 -- not a sequence,
not reconstructible into one, but not nothing. Salting does not help, since users
need the same hash to sketch their own molecules and so the salt ships too. This
is obfuscation, not encryption. Sketching only cluster REPRESENTATIVES exposes
far less and answers the same question, since the bins are about regions of
sequence space rather than individual molecules.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .table import clean_seq

N_HASHES = 128
K = 4                      # matches data.cluster_graph; see the note above
FORMAT = "madm-sketch-1"


def kmers(vh: str, vl: str, k: int = K) -> set:
    """k-mers of both chains, pooled.

    Pooled rather than per chain because the cluster rule is "VH >= t OR VL >= t"
    and a single pooled set cannot express OR. The consequence is that a sketch
    distance is slightly stricter than the cluster rule: an antibody sharing only
    its heavy chain scores lower here than the clustering would give it. Stated
    rather than hidden -- it makes the bins conservative, which is the safe
    direction for "is this molecule in domain?".
    """
    out = set()
    for s in (clean_seq(vh), clean_seq(vl)):
        out.update(s[i:i + k] for i in range(len(s) - k + 1))
    return out


def _base_hash(km: str) -> int:
    return int.from_bytes(hashlib.blake2b(km.encode(), digest_size=8).digest(),
                          "little")


def _coeffs(n_hashes: int, seed: int = 0):
    """`n_hashes` universal hash functions h_i(x) = a_i*x + b_i over uint64.

    Derived from a FIXED seed and stored in the file, so a sketch built today
    compares correctly against one built next year. A sketch made with different
    coefficients is not wrong, it is incomparable -- which is why `compare`
    refuses rather than producing plausible nonsense.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2 ** 63, size=n_hashes, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2 ** 63, size=n_hashes, dtype=np.uint64)
    return a, b


def sketch_many(vh: Sequence[str], vl: Sequence[str], n_hashes: int = N_HASHES,
                k: int = K, seed: int = 0) -> np.ndarray:
    """[n_molecules, n_hashes] of uint64 minima."""
    a, b = _coeffs(n_hashes, seed)
    out = np.zeros((len(vh), n_hashes), dtype=np.uint64)
    with np.errstate(over="ignore"):          # uint64 wraparound IS the hash
        for i, (h, l) in enumerate(zip(vh, vl)):
            ks = kmers(h, l, k)
            if not ks:
                out[i] = np.iinfo(np.uint64).max      # empty -> similarity 0
                continue
            base = np.fromiter((_base_hash(x) for x in ks),
                               dtype=np.uint64, count=len(ks))
            out[i] = (base[None, :] * a[:, None] + b[:, None]).min(axis=1)
    return out


def similarity(A: np.ndarray, B: np.ndarray, chunk: int = 256) -> np.ndarray:
    """[len(A), len(B)] estimated Jaccard. Chunked: the full comparison is
    len(A) x len(B) x n_hashes booleans, which is gigabytes on a real panel."""
    A = np.asarray(A); B = np.asarray(B)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"sketches have different widths ({A.shape[1]} vs "
                         f"{B.shape[1]}) -- they were built with different "
                         "n_hashes and cannot be compared")
    out = np.empty((len(A), len(B)), dtype=np.float32)
    for s in range(0, len(A), chunk):
        blk = A[s:s + chunk]
        out[s:s + chunk] = (blk[:, None, :] == B[None, :, :]).mean(axis=2)
    return out


def max_similarity(A: np.ndarray, B: np.ndarray, exclude_self: bool = False,
                   chunk: int = 256) -> np.ndarray:
    """Nearest-neighbour similarity of each row of A to the set B.

    exclude_self: for A inside B, ignore the perfect self-match. Without it every
    training molecule reports 1.0 and the "how dense is training space" question
    cannot be asked at all.
    """
    A = np.asarray(A); B = np.asarray(B)
    if not len(B):
        return np.zeros(len(A), dtype=np.float32)
    out = np.empty(len(A), dtype=np.float32)
    for s in range(0, len(A), chunk):
        sim = similarity(A[s:s + chunk], B)
        if exclude_self:
            for r in range(len(sim)):
                j = s + r
                if j < len(B):
                    sim[r, j] = -1.0
        out[s:s + chunk] = sim.max(axis=1)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
def save(path, groups: Dict[str, np.ndarray], n_hashes: int = N_HASHES,
         k: int = K, seed: int = 0, note: str = "",
         index: Optional[Dict[str, np.ndarray]] = None) -> Path:
    """One compressed .npz: the sketches, and who belongs to which group.

    `index` maps group -> row indices into the "ALL" matrix. Storing membership
    rather than only the subsets is what makes SET questions answerable -- "BVP
    molecules that do NOT have AC-SINS" is the analogue question, and without the
    indices an auxiliary curve is dominated by co-measured molecules sitting at
    similarity 1.0 for the trivial reason that they are the same molecules.

    The build parameters travel with the data. A sketch compared against one
    built with a different k or seed is not approximately right, it is
    meaningless, and the only defence is to record what made it.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    extra = {f"i_{g}": np.asarray(v, dtype=np.int64)
             for g, v in (index or {}).items()}
    np.savez_compressed(p, format=np.array(FORMAT), n_hashes=np.array(n_hashes),
                        k=np.array(k), seed=np.array(seed), note=np.array(note),
                        groups=np.array(list(groups)),
                        **{f"g_{g}": v for g, v in groups.items()}, **extra)
    return p


def load(path):
    """(groups, meta, index). `index` is empty for files built before it existed,
    so the caller must treat set operations as unavailable rather than assume
    them."""
    d = np.load(Path(path), allow_pickle=True)
    if str(d["format"]) != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    meta = {"n_hashes": int(d["n_hashes"]), "k": int(d["k"]),
            "seed": int(d["seed"]), "note": str(d["note"])}
    groups = {str(g): d[f"g_{g}"] for g in d["groups"]}
    index = {str(g)[2:]: d[g] for g in d.files if str(g).startswith("i_")}
    return groups, meta, index


def sketch_like(vh, vl, meta) -> np.ndarray:
    """Sketch new sequences with the parameters a stored file was built with."""
    return sketch_many(vh, vl, n_hashes=meta["n_hashes"], k=meta["k"],
                       seed=meta["seed"])


BINS = [0.0, 0.5, 0.7, 0.8, 0.9, 1.0001]
BIN_LABELS = ["< 0.5  novel", "0.5-0.7  distant", "0.7-0.8  related family",
              "0.8-0.9  same lineage", ">= 0.9  near-duplicate"]


def bin_similarity(sim: np.ndarray) -> np.ndarray:
    """Index into BIN_LABELS. The bands are the ones a reader can act on: >=0.9
    is effectively a variant of something already measured, <0.5 is novel
    sequence space, and a single pooled score mixes the two."""
    return np.clip(np.digitize(np.asarray(sim, float), BINS) - 1,
                   0, len(BIN_LABELS) - 1)
