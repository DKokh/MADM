"""Turning the feature extractor's output into model input.

Not to be confused with `features3d.py`, which is the extractor itself: that one
reads PDB files and needs biotite, anarci, propka and mdtraj; this one is pure
numpy and only reshapes what the extractor returns.

    features3d.extract_features_batch(pdb) -> dict -> features.combine_channels -> X

Two steps, both pure numpy, both previously living in notebook cells:

  1. CHANNEL ASSEMBLY.  `features3d.extract_features_batch` returns one
     array per named feature block. Generate_Embedding_clean.ipynb cell 12
     concatenates every per-residue block into one [N, L, F] tensor and records
     the column names. That rule is `combine_channels` here, so the tensor and the
     names are built in one place and can be written into the cache together.

  2. IMGT FRAME.  The extractor pads from the N-terminus, so column j of the conv
     input is a different residue in different antibodies. Model_TwoStream.ipynb
     cell 22 scatters each molecule onto a fixed frame of IMGT positions instead;
     `infer_twostream.align_features_to_frame` reproduces that frame at inference
     from `imgt_positions` stored in the bundle. Both are here.

The frame is built from TRAINING molecules only and carried with the model. A
frame rebuilt at inference from whatever happens to be in the prediction table
would put the same residue in a different column.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np

# Keys returned by extract_features_batch that are not per-residue feature
# channels. Everything else with ndim == 3 and a matching length axis is one.
METADATA_KEYS = {
    "CA", "mask", "IMGT_vh", "IMGT_vl", "B2A", "A2B",
    "pI_and_charge", "pdb_paths", "Ab_ID", "vh", "vl", "source",
}

# Column order of pI_and_charge, as assembled in Features3D_v1.extract_features_single.
GLOBAL_FEATURE_NAMES = ["pI_folded", "pI_unfolded", "charge_pH7.5", "charge_pH6.0",
                        "is_lambda", "vh_charge", "vl_charge"]


def combine_channels(feats: dict, max_len: int) -> Tuple[np.ndarray, List[str]]:
    """[N, L, F] per-residue tensor + one name per column.

    Blocks are taken in the extractor's own key order, which is the declaration
    order of its `batch` dict -- stable across calls, and the order every
    published feature-importance table is indexed by. NaN is mapped to 0, matching
    `np.nan_to_num(d["combined"])` in MultiTask.ipynb and in
    infer_twostream.predict_from_npz; without it one bad propka or DSSP value
    turns a whole channel's mean and sd into NaN during standardisation.
    """
    keys = [k for k, v in feats.items()
            if k not in METADATA_KEYS and isinstance(v, np.ndarray)
            and v.ndim == 3 and v.shape[1] == max_len]
    if not keys:
        raise ValueError(
            "no per-residue feature blocks found: expected arrays of shape "
            f"[N, {max_len}, C]. Got "
            f"{ {k: getattr(v, 'shape', type(v).__name__) for k, v in feats.items()} }")

    cols, names = [], []
    for k in keys:
        a = np.asarray(feats[k], dtype=np.float32)
        cols.append(a)
        c = a.shape[-1]
        names += [k] if c == 1 else [f"{k}_{i + 1}" for i in range(c)]

    X = np.nan_to_num(np.concatenate(cols, axis=-1), nan=0.0,
                      posinf=0.0, neginf=0.0).astype(np.float32)
    if X.shape[-1] != len(names):
        raise AssertionError(f"channel/name mismatch: {X.shape[-1]} vs {len(names)}")
    return X, names


def global_features(feats: dict) -> Tuple[np.ndarray, List[str]]:
    """[N, G] global (per-molecule) features, NaN-free, plus their names."""
    if "pI_and_charge" not in feats:
        raise KeyError("pI_and_charge missing from the feature extractor output")
    Xg = np.asarray(feats["pI_and_charge"], dtype=np.float32)
    if Xg.ndim == 1:
        Xg = Xg[:, None]
    Xg = np.nan_to_num(Xg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    names = (GLOBAL_FEATURE_NAMES if Xg.shape[-1] == len(GLOBAL_FEATURE_NAMES)
             else [f"global_{i}" for i in range(Xg.shape[-1])])
    return Xg, names


# --------------------------------------------------------------------------- #
# IMGT frame
# --------------------------------------------------------------------------- #
def _sort_imgt_key(s: str):
    chain, pos = s.split("_", 1)
    m = re.match(r"(\d+)([A-Z]*)", pos)
    if m is None:                       # unparseable label -> sort last, stably
        return (2, 10 ** 9, pos)
    return (0 if chain == "H" else 1, int(m.group(1)), m.group(2))


def _labels(imgt_vh, imgt_vl) -> List[str]:
    """One 'H_<pos>' / 'L_<pos>' label per numbered residue, heavy chain first.

    Accepts both the extractor's list-of-labels and the space-joined string form
    written by Generate_Embedding_clean.ipynb cell 10.
    """
    def split(x):
        if isinstance(x, str):
            return x.split()
        return [str(p) for p in np.atleast_1d(x)]
    return [f"H_{p}" for p in split(imgt_vh)] + [f"L_{p}" for p in split(imgt_vl)]


def build_frame(imgt_vh_list: Sequence, imgt_vl_list: Sequence) -> List[str]:
    """The sorted union of IMGT positions over the given molecules.

    Build this from TRAINING rows only, then save it: it defines what column j
    means, and a frame rebuilt from a different set of molecules is a different
    coordinate system.
    """
    all_pos = set()
    for vhn, vln in zip(imgt_vh_list, imgt_vl_list):
        all_pos.update(_labels(vhn, vln))
    return sorted(all_pos, key=_sort_imgt_key)


def align_to_frame(X: np.ndarray, mask: np.ndarray, imgt_vh_list: Sequence,
                   imgt_vl_list: Sequence, frame: Sequence[str],
                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter [N, L_raw, F] onto the fixed frame -> [N, L_frame, F].

    Residues at positions outside the frame are dropped; frame positions with no
    residue stay zero and masked out.

    Returns (X_aligned, mask_aligned, ok) where `ok` is False for molecules whose
    IMGT numbering length disagrees with their residue mask. Those rows cannot be
    placed and are returned as all-zero -- the caller drops them rather than
    training on an empty tensor, which is what the notebook's `continue` quietly
    did not do.
    """
    X = np.asarray(X, dtype=np.float32)
    mask = np.asarray(mask).astype(bool)
    n, _, n_feat = X.shape
    L = len(frame)
    idx_of = {p: j for j, p in enumerate(frame)}

    Xa = np.zeros((n, L, n_feat), dtype=np.float32)
    ma = np.zeros((n, L), dtype=bool)
    ok = np.ones(n, dtype=bool)

    for i in range(n):
        labels = _labels(imgt_vh_list[i], imgt_vl_list[i])
        valid = np.where(mask[i])[0]
        if len(labels) != len(valid):
            ok[i] = False
            continue
        for lab, pos in zip(labels, valid):
            j = idx_of.get(lab)
            if j is not None:
                Xa[i, j, :] = X[i, pos, :]
                ma[i, j] = True
    if verbose and not ok.all():
        print(f"[frame] {int((~ok).sum())}/{n} molecule(s) dropped: IMGT numbering "
              "length does not match the residue mask")
    return Xa, ma.astype(np.float32), ok
