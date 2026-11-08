"""Structures, features and embeddings -- built once per unique Fv, cached on disk.

Everything is keyed on sha1(clean(VH) + "|" + clean(VL)). Keying on Ab_ID or on
source means the same antibody is built twice when it appears in two datasets, and
renaming a source silently invalidates the whole cache.

Only what the configured streams need is built:

    streams = [feat]     structures + 3D features. AbLang is never loaded.
    streams = [ablang]   embeddings only. Structure prediction, which dominates
                         the runtime, is skipped entirely.
    streams = both       everything.

STRUCTURES YOU ALREADY HAVE.  `build_structures(pdb_dir=...)` (config
`paths.structures_in`, CLI `--pdb-dir`) uses your own .pdb files where they exist
and predicts only the rest. Because every feature is read off the STRUCTURE and
never off VH/VL, a supplied file is checked against the row's sequences first --
see `check_pdb_matches` for why that check is not optional in spirit.

CACHE CONTENTS.  A feature cache holds the assembled per-residue tensor `X`, the
global features `Xg`, the mask, the IMGT numbering, the column names, and the
`features:` block that produced them. Everything a model needs to check that the
tensor it is being served is the tensor it was trained on.

CACHE NAMING.  Callers are given the exact path by `feature_cache_path` /
`embedding_cache_path`. Never search the cache directory for "the newest file":
the name is a content hash, so the newest file by any ordering is unrelated to
the table in hand.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Config, FeaturesCfg
from .features import align_to_frame, combine_channels, global_features
from .table import clean_seq, unique_fvs

CACHE_FORMAT = "madm-features-2"


def fv_hash(vh: str, vl: str) -> str:
    return hashlib.sha1((clean_seq(vh) + "|" + clean_seq(vl)).encode()).hexdigest()[:16]


def needs(cfg: Config) -> Dict[str, bool]:
    return {"feat": "feat" in cfg.streams, "ablang": "ablang" in cfg.streams}


def _key_tag(keys: Sequence[str]) -> str:
    return hashlib.sha1("".join(sorted(keys)).encode()).hexdigest()[:12]


def feature_cache_path(cache: str | Path, keys: Sequence[str], fcfg: FeaturesCfg) -> Path:
    """Exact cache file for this set of molecules AND these settings.

    The settings signature is part of the name, so changing k_neighbors produces a
    new file instead of silently reusing a tensor with a different channel count.
    """
    return Path(cache) / f"features_{_key_tag(keys)}_{fcfg.signature()}.npz"


# Per-residue language models that can feed the `ablang` stream. The stream is
# named for what it was first, not for what it can hold: the encoder reads its
# width off the cache, so any of these drops in without touching the model.
#
# ESM sees camelid VHHs in UniRef; AbLang2-paired is trained on paired human
# VH/VL, so a single domain is outside its distribution -- which is the reason
# this switch exists, and why the embedder has to be recorded rather than assumed.
EMBEDDERS = {
    "ablang2":    None,                             # via the ablang2 package
    "esm2-35m":   "facebook/esm2_t12_35M_UR50D",    # 480-d, for a quick check
    "esm2-150m":  "facebook/esm2_t30_150M_UR50D",   # 640-d
    "esm2-650m":  "facebook/esm2_t33_650M_UR50D",   # 1280-d, the usual choice
}


def embedding_cache_path(cache: str | Path, keys: Sequence[str], max_len: int,
                         embedder: str = "ablang2", layer: int = -1) -> Path:
    # ablang2 keeps its historical name so caches built before this switch
    # existed are still found -- they cost GPU-hours and should not be orphaned
    # by a rename. Everything else carries its embedder, so two embedders for
    # the same molecules can never collide on one filename.
    stem = "ablang" if embedder == "ablang2" else embedder.replace("-", "_")
    # A different layer of the same model is a different representation, so it
    # gets its own file. -1 keeps the bare name, so caches built before the layer
    # switch existed are still found.
    if embedder != "ablang2" and layer not in (-1, None):
        stem = f"{stem}_L{layer}"
    return Path(cache) / f"{stem}_{_key_tag(keys)}_{max_len}.npz"


# --------------------------------------------------------------------------- #
# 1. structures
# --------------------------------------------------------------------------- #
PDB_SUFFIXES = (".pdb", ".ent", ".pdb1")

# PDB three-letter -> one-letter, including the protonation-state variants that
# PDB2PQR / propka write. Anything else is read as X and will not match, which is
# the safe direction: an unrecognised residue should raise a question, not pass.
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
    "HID": "H", "HIE": "H", "HIP": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "CYX": "C", "CYM": "C", "ASH": "D", "GLH": "E", "LYN": "K", "MSE": "M",
}
# Heavy chain first, light second. The feature extractor renames A -> H and B -> L
# before doing anything else, so those two spellings are equivalent here.
_HEAVY_CHAINS, _LIGHT_CHAINS = ("H", "A"), ("L", "B")


def pdb_chain_sequences(p: str | Path) -> Dict[str, str]:
    """{chain_id: one-letter sequence} from ATOM records, in file order.

    Deliberately not biotite/BioPython: this runs before the feature extractor is
    imported, on files the user may have hand-made, and the answer needed is just
    "which residues are in here". Reading CA atoms is exact for that and cannot
    itself fail on an unusual het-atom.
    """
    seqs: Dict[str, list] = {}
    seen: set = set()
    with open(p, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                if line.startswith("ENDMDL"):
                    break            # NMR / ensemble file: the first model only
                continue
            if line[12:16].strip() != "CA":
                continue
            chain = line[21].strip() or "_"
            resid = line[22:27]      # number + insertion code
            if (chain, resid) in seen:
                continue             # altloc B, C, ... of the same residue
            seen.add((chain, resid))
            seqs.setdefault(chain, []).append(
                _THREE_TO_ONE.get(line[17:20].strip().upper(), "X"))
    return {c: "".join(v) for c, v in seqs.items()}


def check_pdb_matches(p: str | Path, vh: str, vl: str,
                      min_identity: float = 0.90) -> Optional[str]:
    """None if the structure is this molecule; otherwise why it is not.

    This exists because every feature is read off the STRUCTURE -- the VH/VL
    strings are carried along for bookkeeping and never enter the tensor. A file
    under the wrong name therefore attaches one antibody's geometry to another's
    measurement, and no metric anywhere downstream would show it: the run simply
    learns something slightly wrong and reports a slightly worse number.

    Missing density is tolerated. A crystal structure with a disordered CDR loop
    is still the right molecule, and refusing it would rule out exactly the
    structures worth supplying. What is refused is a sequence that is a different
    antibody.
    """
    import difflib

    chains = pdb_chain_sequences(p)
    if not chains:
        return "no CA atoms found -- is this a PDB file?"
    pick = {}
    for want, names in (("H", _HEAVY_CHAINS), ("L", _LIGHT_CHAINS)):
        hit = [c for c in names if chains.get(c)]
        if hit:
            pick[want] = chains[hit[0]]
    # A blank VL in the table says single-domain, so a one-chain file is what we
    # expect rather than a truncated Fv. Only the heavy chain is then required,
    # and an extra L chain in the file would be a different molecule.
    single = not clean_seq(vl)
    need = 1 if single else 2
    if len(pick) < need:
        return (f"chains are {sorted(chains)}; need a heavy chain named H or A"
                + ("" if single else " and a light chain named L or B")
                + ". Rename the chains, or split the file -- MADM models an Fv "
                  "or a single VHH domain, not a whole IgG.")
    if single and "L" in pick:
        return (f"the table has no VL but the file has chains {sorted(chains)}; "
                "a single-domain molecule should be a one-chain structure.")

    worst = []
    pairs = [("VH", clean_seq(vh))] if single else [("VH", clean_seq(vh)),
                                                    ("VL", clean_seq(vl))]
    for want, seq in pairs:
        got = pick["H" if want == "VH" else "L"]
        r = difflib.SequenceMatcher(None, got, seq, autojunk=False).ratio()
        if r < min_identity:
            worst.append(f"{want}: {r:.0%} identity to the structure's "
                         f"{'H' if want == 'VH' else 'L'} chain "
                         f"({len(got)} residues modelled vs {len(seq)} in the table)")
    return "; ".join(worst) if worst else None


def _find_supplied(r, pdb_dir: Optional[Path], h: str) -> Optional[Path]:
    """The user's own structure for this molecule, if there is one.

    Three ways to name it, in order of how specific they are:
      1. a `pdb` column in the table -- explicit, wins over everything. Use this
                                        when file names are not unique across
                                        datasets; a full path settles it.
      2. <pdb_dir>/<Ab_ID>.pdb       -- one flat directory
      3. <pdb_dir>/<sha1(VH|VL)>.pdb -- MADM's own convention, so a cache
                                        directory can be handed over as-is

    pdb_dir is never searched recursively: Ab_IDs collide across datasets, and a
    recursive glob would settle that by filesystem order.
    """
    # pd.isna first: a partially-filled `pdb` column arrives as NaN, and NaN is
    # truthy -- str(nan) == "nan" would become a filename.
    raw = getattr(r, "pdb", None)
    col = "" if raw is None or pd.isna(raw) else str(raw).strip()
    if col:
        q = Path(col)
        if not q.is_absolute() and pdb_dir is not None and not q.exists():
            q = pdb_dir / q
        return q, True                # explicit: returned even if absent, so a
                                      # path written by hand and got wrong is
                                      # reported rather than predicted over
    if pdb_dir is None:
        return None, False

    for stem in (str(r.Ab_ID), h):
        for suf in PDB_SUFFIXES:
            q = pdb_dir / f"{stem}{suf}"
            if q.exists():
                return q, False
    return None, False


def build_structures(df: pd.DataFrame, out_dir: str | Path,
                     overwrite: bool = False,
                     pdb_dir: Optional[str | Path] = None,
                     check_sequence: bool = True) -> pd.DataFrame:
    """One .pdb per unique Fv: yours where you have one, predicted where you don't.

    Adds `pdb` (path, "" if unavailable) and `pdb_source` ("supplied" |
    "predicted" | "cached" | ""), so a later run can tell which numbers rest on
    experimental geometry and which on ABodyBuilder2.

    Resolution order per molecule -- the first that yields a usable file wins:
      1. a structure you supplied (`pdb` column, or `pdb_dir`)
      2. one already in `out_dir` from an earlier run
      3. ABodyBuilder2

    ImmuneBuilder is imported only if something actually has to be predicted, so a
    complete set of your own structures needs no structure-prediction install at
    all.

    Per-antibody failures are recorded, not raised: one unbuildable antibody
    should not stop a batch of a thousand. Rows with pdb == "" are dropped from
    the feature stream later, with a count.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pdb_dir = Path(pdb_dir) if pdb_dir else None
    if pdb_dir is not None and not pdb_dir.is_dir():
        raise FileNotFoundError(
            f"paths.structures_in: {pdb_dir.resolve()} is not a directory. "
            "Point it at the folder holding your .pdb files, or set it to null "
            "to predict every structure.")

    uniq = unique_fvs(df)
    has_col = "pdb" in df.columns and bool(
        (df["pdb"].notna() & (df["pdb"].astype(str).str.strip() != "")).any())
    predictor = nano = None
    paths: Dict[str, str] = {}
    source: Dict[str, str] = {}
    failed, rebuilt, mismatched, missing_named = [], 0, [], []
    n_supplied = skipped_named = 0

    import time
    t0, n_built, last = time.time(), 0, 0.0

    for n_done, (_, r) in enumerate(uniq.iterrows()):
        # Progress on the SLOW path only. Structure prediction is seconds per
        # molecule and silent, so a panel of a few thousand looks like a hang for
        # hours -- and the natural response to a hang is to kill it, losing every
        # structure built so far that was not yet cached to disk.
        if n_built and (time.time() - last > 30):
            rate = n_built / (time.time() - t0)
            left = (len(uniq) - n_done) / rate if rate else 0
            print(f"[structures] {n_done}/{len(uniq)}  built {n_built} "
                  f"({rate * 60:.1f}/min, ~{left / 60:.0f} min left)")
            last = time.time()

        h = fv_hash(r.VH, r.VL)
        paths[r.key], source[r.key] = "", ""

        # --- 1. a structure the user supplied --------------------------------
        q, explicit = _find_supplied(r, pdb_dir, h)
        if q is not None:
            bad = None
            if not q.exists():
                bad = "file does not exist"
                missing_named.append((r.Ab_ID, str(q)))
            elif not _usable_pdb(q):
                bad = "no coordinates"
                mismatched.append((r.Ab_ID, f"{q.name}: no coordinates"))
            else:
                why = check_pdb_matches(q, r.VH, r.VL) if check_sequence else None
                if why:
                    bad = why
                    mismatched.append((r.Ab_ID, f"{q.name}: {why}"))
            if bad is None:
                paths[r.key], source[r.key] = str(q), "supplied"
                n_supplied += 1
                continue
            if explicit:
                # A `pdb` cell is an assertion that THIS file is the structure for
                # THIS molecule. When it turns out not to be, quietly predicting a
                # different one hides the mistake behind a plausible result -- and
                # drags in a structure predictor the user was not asking for. Warn
                # and move on; the molecule leaves with pdb == "" and is handled
                # exactly like one whose structure could not be built.
                print(f"[structures] WARNING {r.Ab_ID}: {bad} -- skipped ({q})")
                skipped_named += 1
                continue
            # Found by NAME in --pdb-dir rather than asserted by the table, so the
            # user has claimed nothing about it: fall through and predict.

        # --- 2. built by an earlier run --------------------------------------
        p = out / f"{h}.pdb"
        if p.exists() and not overwrite:
            if _usable_pdb(p):
                paths[r.key], source[r.key] = str(p), "cached"
                continue
            # A run killed mid-write leaves a truncated or empty file. It passes
            # exists() and then fails inside the feature extractor, where the
            # error points at the wrong thing -- so rebuild it here instead.
            p.unlink(missing_ok=True)
            rebuilt += 1

        # --- 3. predicted ----------------------------------------------------
        # A VHH row needs a different network. Both are built lazily and cached
        # separately, so a mixed table pays for each only if it contains one.
        vhh = not clean_seq(r.VL)
        if (nano if vhh else predictor) is None:
            n_todo = len(uniq) - len(paths) + 1
            who = "NanoBodyBuilder2" if vhh else "ABodyBuilder2"
            print(f"[structures] predicting up to {n_todo} structure(s) with "
                  f"{who}. This is the slow step -- seconds per molecule, "
                  "and much slower on CPU than GPU.")
            if n_todo > 500:
                print("[structures] For a panel this size consider --pdb-dir with "
                      "structures you already have, or --model ablang, which "
                      "needs none.")
            if vhh:
                nano = _nanobodybuilder()
            else:
                predictor = _abodybuilder()
            last = time.time()
        try:
            n_built += 1
            ab = _predict_structure(nano if vhh else predictor, r.VH, r.VL)
            ab.save(str(p))
            if not _usable_pdb(p):        # wrote, but produced nothing usable
                raise ValueError("predictor wrote an empty/incomplete PDB")
            paths[r.key], source[r.key] = str(p), "predicted"
        except Exception as e:            # noqa: BLE001 - one bad Fv must not stop the batch
            p.unlink(missing_ok=True)     # never leave a partial file behind
            failed.append((r.Ab_ID, str(e)[:80]))

    df = df.copy()
    df["pdb"] = df.key.map(paths).fillna("")
    df["pdb_source"] = df.key.map(source).fillna("")
    n_ok = int((df.pdb != "").sum())
    print(f"[structures] {n_ok}/{len(df)} available ({len(uniq)} unique Fv)")
    if pdb_dir is not None or has_col:
        by = df.loc[df.pdb != "", "pdb_source"].value_counts().to_dict()
        print(f"[structures] provenance (molecules): {by or '{}'}")
    if n_supplied and not check_sequence:
        print(f"[structures] WARNING sequence checking is OFF: the {n_supplied} "
              "supplied structure(s) were used without confirming they are the "
              "molecules the table names them as.")
    if missing_named:
        print(f"[structures] WARNING {len(missing_named)} row(s) name a .pdb that "
              f"does not exist, e.g. {missing_named[:3]}")
    if mismatched:
        print(f"[structures] WARNING {len(mismatched)} supplied structure(s) were "
              f"REJECTED, e.g. {mismatched[:3]}")
        print("[structures] every feature is read off the structure, so a file "
              "under the wrong name would attach the wrong geometry to the label.")
    if skipped_named:
        print(f"[structures] {skipped_named} molecule(s) named a .pdb in the table "
              "that could not be used: SKIPPED, not predicted. A `pdb` cell asserts "
              "which structure belongs to that row, so substituting a predicted one "
              "would hide the mistake. Fix the paths, or blank the cells to have "
              "them predicted.")
    if rebuilt:
        print(f"[structures] {rebuilt} cached file(s) were truncated and were "
              "discarded, then rebuilt (or counted as failures below)")
    if failed:
        print(f"[structures] {len(failed)} failed, e.g. {failed[:3]}")
        print("[structures] these molecules carry pdb == '' and are dropped from the "
              "feature stream; `madm predict` still returns a row for each of them.")
    return df


def _nanobodybuilder(attempts: int = 4, first_wait: float = 5.0):
    """NanoBodyBuilder2, for rows whose VL is empty. Same retry logic as below."""
    return _abodybuilder(attempts, first_wait, single_domain=True)


def _predict_structure(predictor, vh: str, vl: str):
    """One structure, whichever builder was handed in.

    NanoBodyBuilder2 takes only a heavy chain. Its argument form has varied
    between ImmuneBuilder releases -- a {"H": seq} dict in some, a bare string in
    others -- so try the dict and fall back, rather than pinning a version for
    one call.
    """
    if clean_seq(vl):
        return predictor.predict({"H": clean_seq(vh), "L": clean_seq(vl)})
    try:
        return predictor.predict({"H": clean_seq(vh)})
    except (TypeError, KeyError, AttributeError):
        return predictor.predict(clean_seq(vh))


def _abodybuilder(attempts: int = 4, first_wait: float = 5.0,
                  single_domain: bool = False):
    """ABodyBuilder2 -- or NanoBodyBuilder2 -- imported and built on first use.

    Deferred so that a run whose structures are ALL supplied never needs
    ImmuneBuilder installed -- which is most of the point of supplying them.

    CONSTRUCTING it is a network operation the first time: ImmuneBuilder fetches
    four sets of weights from Zenodo into its own package directory. Zenodo
    returns 502/504 under load often enough that a single attempt is not a fair
    test, and failing here discards a run that has already loaded and screened the
    table. So transient HTTP and connection errors are retried with backoff; an
    HTTP 404, a permission error or a full disk are not, because waiting does not
    fix those.
    """
    import time

    try:
        if single_domain:
            from ImmuneBuilder import NanoBodyBuilder2 as _Builder
        else:
            from ImmuneBuilder import ABodyBuilder2 as _Builder
    except ImportError as e:                  # environment, not data
        raise ImportError(_install_hint(e)) from e

    last = None
    for i in range(attempts):
        try:
            return _Builder()
        except Exception as e:                # noqa: BLE001
            if not _is_transient(e):
                raise RuntimeError(_weights_hint(e, transient=False)) from e
            last = e
            if i < attempts - 1:
                wait = first_wait * (2 ** i)
                print(f"[structures] weights download failed ({type(e).__name__}: "
                      f"{str(e)[:90]}); retrying in {wait:.0f}s "
                      f"[{i + 2}/{attempts}]")
                time.sleep(wait)
    raise RuntimeError(_weights_hint(last, transient=True)) from last


def _is_transient(e: Exception) -> bool:
    """Is this worth waiting out?

    Deliberately conservative: a 5xx, a 429, or a socket-level failure is the
    server or the network having a moment. A 404 means the URL is wrong and no
    amount of retrying will produce the file -- retrying that just turns a clear
    failure into a slow one.
    """
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status is not None:
        return status in (408, 425, 429) or 500 <= status < 600
    name = type(e).__name__
    return name in ("ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout",
                    "ChunkedEncodingError", "IncompleteRead", "SSLError",
                    "URLError", "TimeoutError", "socket.timeout")


def weights_files() -> list:
    """Where ImmuneBuilder has cached its weights, if it has.

    Globbed rather than hardcoded: the subdirectory is ImmuneBuilder's private
    business and has moved between versions, so asserting a path here would make
    MADM wrong every time they rearrange it. An empty list means "not cached",
    not "broken".
    """
    try:
        import ImmuneBuilder
        root = Path(ImmuneBuilder.__file__).resolve().parent
        return sorted(p for p in root.rglob("*antibody_model*") if p.is_file())
    except Exception:                         # noqa: BLE001
        return []


def _weights_hint(e, transient: bool) -> str:
    have = weights_files()
    head = (
        "ABodyBuilder2's weights could not be downloaded from Zenodo"
        + (" after several attempts." if transient else ".")
        + ("\nThis usually means Zenodo is overloaded rather than anything wrong "
           "on your side -- 502 and 504 there are common and clear up on their own."
           if transient else ""))
    return (
        head
        + f"\n\nweights currently cached: {len(have)} file(s)"
        + (f" in {have[0].parent}" if have else " (none)")
        + "\n\nWhat to do:\n"
        "  * wait and re-run -- everything already done is cached, so a re-run "
        "resumes rather than restarting\n"
        "  * check outbound HTTPS to zenodo.org is open (a locked-down instance "
        "or a proxy will fail here every time, not intermittently):\n"
        "        curl -sSI https://zenodo.org/record/7258553 | head -1\n"
        "  * pre-fetch the weights once, then run offline:\n"
        "        python -c \"from ImmuneBuilder import ABodyBuilder2; "
        "ABodyBuilder2()\"\n"
        "  * --pdb-dir <dir>   use structures you already have; nothing is "
        "predicted, so no download happens\n"
        "  * --model ablang    skip the structure stream entirely\n"
        f"\noriginal error: {type(e).__name__}: {e}")


# Packages ImmuneBuilder imports that pip cannot supply. openmm ships compiled
# CUDA/OpenCL kernels and pdbfixer is not published to PyPI at all, so both come
# from conda-forge -- telling someone to `pip install` them sends them in a circle.
CONDA_ONLY = {"pdbfixer": "conda install -c conda-forge openmm pdbfixer",
              "openmm": "conda install -c conda-forge openmm pdbfixer",
              "anarci": "conda install -c bioconda anarci"}


def _install_hint(e: ImportError) -> str:
    """Name the module that ACTUALLY failed, and the channel that has it.

    This fires in two quite different situations and they need different fixes:
    ImmuneBuilder is absent, or ImmuneBuilder is installed and one of ITS imports
    failed. Reporting the second as the first is how someone ends up re-running a
    pip command that was never going to help -- which is what the previous version
    of this message did.
    """
    missing = (getattr(e, "name", None) or "").split(".")[0]
    if missing and missing != "ImmuneBuilder":
        fix = CONDA_ONLY.get(missing, f"pip install {missing}")
        head = (f"ImmuneBuilder is installed, but it needs `{missing}`, which is "
                f"missing:\n    {fix}")
        if missing in CONDA_ONLY:
            head += ("\n  (pip cannot supply this one -- it is a conda-forge "
                     "package, so `pip install -e '.[structure]'` will not fix it.)")
    else:
        head = "structures need ImmuneBuilder:\n    pip install -e '.[structure]'"
    return (
        head + "\n\nTwo ways to proceed without installing anything:\n"
        "  * --pdb-dir <dir>   use structures you already have; nothing is "
        "predicted, so ImmuneBuilder is never imported\n"
        "  * --model ablang    skip the structure stream entirely\n"
        f"\noriginal error: {e}")


def _usable_pdb(p: Path, min_atoms: int = 100) -> bool:
    """A cached structure is only reusable if it actually holds coordinates.

    `exists()` is not enough: an interrupted write leaves a zero-byte or half-
    written file that the extractor later fails on with an error that points at
    parsing rather than at the real cause. Counting ATOM records is cheap
    (structures are a few hundred kB) and catches both cases.
    """
    try:
        if p.stat().st_size == 0:
            return False
        n = 0
        with open(p, "r", errors="ignore") as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    n += 1
                    if n >= min_atoms:
                        return True
        return False
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# 2. per-residue + global features
# --------------------------------------------------------------------------- #
def _feature_extractor():
    """`madm.features3d.extract_features_batch` -- vendored, not a sibling directory.

    This used to import `Scripts.Features3D_v1` from two levels above the package,
    which meant `pip install madm` produced something that trained on embeddings
    and died at the first structure. The extractor is part of what MADM does, so
    it lives inside the package and travels with it.
    """
    try:
        from .features3d import extract_features_batch
        return extract_features_batch
    except ImportError as e:
        missing = (getattr(e, "name", "") or "").split(".")[-1]
        if missing and missing not in ("features3d",):
            # features3d is present; one of ITS third-party imports is not. Naming
            # the actual module matters -- biotite, hydride, anarci, mdtraj and
            # propka fail in different ways and are installed from different places.
            raise ImportError(
                f"the 3D feature extractor needs `{missing}`, which is not "
                f"installed:\n    pip install -e '.[structure]'\n"
                + (f"    {CONDA_ONLY[missing]}\n" if missing in CONDA_ONLY else "")
                + "Only the `feat` stream reads it: --model ablang runs without it."
                f"\noriginal error: {e}") from e
        raise ImportError(
            "madm/features3d.py is missing from the installation.\n"
            f"  expected at: {Path(__file__).resolve().parent / 'features3d.py'}\n"
            "It is the 3D feature extractor and is part of the package; an "
            "installation without it is incomplete. Reinstall, or copy it back "
            "from the source tree.\n"
            "Only the `feat` stream reads it: --model ablang runs without it."
        ) from e


def build_features(df: pd.DataFrame, cache: str | Path, reference_tables: str | Path,
                   fcfg: FeaturesCfg, overwrite: bool = False) -> Path:
    """Run Scripts/Features3D_v1.extract_features_batch and cache MODEL INPUT.

    What lands in the .npz is the assembled tensor, not the extractor's raw output
    dict: `X` [N, L, F], `Xg` [N, G], `mask` [N, L], the IMGT numbering needed to
    build the alignment frame, the column names, and the `features:` settings.
    """
    extract_features_batch = _feature_extractor()

    # One structure per distinct Fv: an antibody appearing twice is expensive to
    # build twice, and duplicate keys in the cache make the key->row map ambiguous.
    have = unique_fvs(df[df.pdb != ""]).reset_index(drop=True)
    if have.empty:
        raise ValueError("no molecules with a built structure -- nothing to featurise")

    path = feature_cache_path(cache, have.key, fcfg)
    Path(cache).mkdir(parents=True, exist_ok=True)
    prov = _provenance(have)
    if path.exists() and not overwrite:
        _check_provenance(path, prov)
        print(f"[features] cached -> {path.name}")
        return path

    ref = Path(reference_tables)
    for f in ("Hidrophobisicy_score.xlsx", "Residue_max_SASA.xlsx"):
        if not (ref / f).exists():
            raise FileNotFoundError(
                f"{ref / f} not found. These ship inside the package, so this "
                "usually means an incomplete install (reinstall with `pip install "
                "-e .`) -- or set paths.reference_tables to the directory "
                "holding Hidrophobisicy_score.xlsx and Residue_max_SASA.xlsx.")

    # Module-level rather than an argument: the extractor is vendored verbatim
    # from Scripts/Features3D_v1.py and its signature is kept identical, so the
    # relaxation budget is set on the module it reads it from.
    from . import features3d as _f3d
    _f3d.RELAX_TIMEOUT_S = fcfg.relax_timeout_s
    _f3d.RELAX_ITERATIONS = fcfg.relax_iterations
    if not fcfg.relax_timeout_s:
        print("[features] hydrogen relaxation OFF (features.relax_timeout_s = 0): "
              "ideal-geometry hydrogens, bounded runtime, less accurate H-bond counts")

    feats = extract_features_batch(
        pdb_paths=have.pdb.tolist(), Ab=have.Ab_ID.tolist(),
        # clean_seq, not the raw column: a single-domain row carries NaN in VL,
        # which reaches `build_ablang_index_map` as a float and fails on len().
        # Cleaning here rather than in the extractor keeps that file identical to
        # Scripts/Features3D_v1.py, which the parity test requires.
        vh=have.VH.map(clean_seq).tolist(), vl=have.VL.map(clean_seq).tolist(),
        source_list=["-"] * len(have), DIR_DATA_PSR=str(ref),
        Score=fcfg.score, max_len=fcfg.max_len, k_neighbors=fcfg.k_neighbors,
        contact_thresholds=fcfg.thresholds())

    X, feature_names = combine_channels(feats, fcfg.max_len)
    Xg, global_names = global_features(feats)
    mask = np.asarray(feats["mask"], dtype=np.float32)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    np.savez_compressed(
        path,
        format=np.array(CACHE_FORMAT),
        keys=have.key.to_numpy(),
        X=X, Xg=Xg, mask=mask,
        IMGT_vh=np.array([" ".join(str(p) for p in v) for v in feats["IMGT_vh"]], dtype=object),
        IMGT_vl=np.array([" ".join(str(p) for p in v) for v in feats["IMGT_vl"]], dtype=object),
        feature_names=np.array(feature_names),
        global_feature_names=np.array(global_names),
        params=np.array(json.dumps(_params_dict(fcfg))),
        pdb_provenance=np.array(prov, dtype=object),
    )
    print(f"[features] {len(have)} molecules -> {path.name}   "
          f"X {X.shape}  Xg {Xg.shape}")
    return path


def _params_dict(fcfg: FeaturesCfg) -> dict:
    from dataclasses import asdict
    return asdict(fcfg)


def _provenance(have: pd.DataFrame) -> list:
    """Per molecule: which structure the features were computed from.

    "predicted" for anything ABodyBuilder2 built -- whether this run or an earlier
    one, since those are the same file. For a supplied structure the FILE NAME is
    recorded too, so swapping one crystal structure for another is visible.
    """
    if "pdb_source" not in have.columns:
        return ["predicted"] * len(have)
    return [f"supplied:{Path(p).name}" if s == "supplied" else "predicted"
            for s, p in zip(have.pdb_source, have.pdb)]


def _check_provenance(path: Path, want: list) -> None:
    """Refuse a cache whose features came from different structures than this run.

    The cache name is keyed on the MOLECULES and the feature settings, neither of
    which changes when you add a directory of crystal structures. So without this,
    the first thing that happens after you supply your own structures is that the
    old predicted-geometry features are served back and nothing says a word.
    """
    try:
        d = np.load(path, allow_pickle=True)
        got = [str(s) for s in d["pdb_provenance"]] if "pdb_provenance" in d.files else None
    except Exception:                                    # noqa: BLE001
        return                                           # unreadable -> rebuilt anyway
    if got is None:
        # Written before provenance was recorded. Those runs predicted everything,
        # so the cache is only wrong if THIS run supplies something.
        got = ["predicted"] * len(want)
    if got == list(want):
        return
    diff = [(w, g) for w, g in zip(want, got) if w != g]
    raise ValueError(
        f"{path.name}: this cache was built from different structures.\n"
        f"  {len(diff)} of {len(want)} molecule(s) differ, e.g. "
        f"now={diff[0][0]!r} cached={diff[0][1]!r}\n"
        "The cache name covers the molecules and the `features:` settings, not "
        "which .pdb each one came from -- so reusing it would serve the old "
        "geometry under the new configuration.\n"
        "Rebuild with `madm prepare --overwrite` (or drop the file).")


# --------------------------------------------------------------------------- #
# 3. AbLang2 embeddings
# --------------------------------------------------------------------------- #
def _ablang_rows(model, pairs) -> list:
    """[L_i, D] per molecule, L_i = len(VH) + len(VL).

    AbLang2 returns either a list of per-molecule arrays (already at the
    molecule's own length) or ONE PADDED TENSOR for the batch. In the padded
    case the rows must be trimmed; taking the row whole marks the batch's
    padding as valid residues, and the pooled embedding then depends on which
    other molecules happened to share the batch.
    """
    rep = model([list(p) for p in pairs], mode="rescoding")
    if isinstance(rep, (list, tuple)):
        return [np.asarray(r, dtype=np.float32) for r in rep]
    arr = np.asarray(rep, dtype=np.float32)
    return [arr[j, :len(vh) + len(vl)] for j, (vh, vl) in enumerate(pairs)]


def _esm_rows(model, tok, device, pairs, layer: int = -1) -> list:
    """The same contract, from ESM-2.

    Each CHAIN is encoded separately and the two are concatenated, so the rows
    line up with the `[VH | VL]` layout that the mask and the IMGT frame assume.
    Encoding the pair as one string with a linker would shift every VL residue
    relative to the feature tensor -- the features and the embedding are indexed
    by the same residue axis, and only one of them would move.

    BOS and EOS are dropped, so the offset against len(VH)+len(VL) is exactly 0
    rather than a constant the caller has to notice and subtract.

    layer : which transformer block to read. -1 is the last, which is the obvious
            choice and usually the wrong one: ESM's final layer is specialised
            for the masked-token prediction it was trained on, and property
            prediction typically transfers better from the middle of the stack.
            0 is the embedding layer, before any attention. Indices follow
            `hidden_states`, so a 33-layer model offers 0..33.
    """
    import torch

    flat, owner = [], []
    for i, (vh, vl) in enumerate(pairs):
        for s in (vh, vl):
            if s:
                flat.append(s); owner.append(i)
    if not flat:
        return [np.zeros((0, model.config.hidden_size), np.float32) for _ in pairs]

    enc = tok(flat, return_tensors="pt", padding=True, add_special_tokens=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        if layer in (-1, None):
            h = model(**enc).last_hidden_state
        else:
            hs = model(**enc, output_hidden_states=True).hidden_states
            if not -len(hs) <= layer < len(hs):
                raise ValueError(f"embedder_layer {layer} outside 0..{len(hs) - 1} "
                                 f"for this model ({len(hs) - 1} blocks)")
            h = hs[layer]
        h = h.float().cpu().numpy()

    out = [[] for _ in pairs]
    for j, (s, i) in enumerate(zip(flat, owner)):
        out[i].append(h[j, 1:1 + len(s)])          # drop BOS, stop before EOS/pad
    return [np.concatenate(v, axis=0).astype(np.float32) if v
            else np.zeros((0, h.shape[-1]), np.float32) for v in out]


def _esm_model(name: str, device: str):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:                       # environment, not data
        raise ImportError(
            "ESM embeddings need the `transformers` package:\n"
            "    pip install transformers\n"
            "or set model.embedder: ablang2 to use AbLang2 instead.") from e
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval().to(device)
    for p in model.parameters():                   # frozen, like AbLang2
        p.requires_grad_(False)
    return model, tok


def build_embeddings(df: pd.DataFrame, cache: str | Path, max_len: int = 290,
                     overwrite: bool = False, batch: int = 16,
                     embedder: str = "ablang2", layer: int = -1) -> Path:
    """Frozen per-residue language-model embeddings, cached as one .npz.

    The encoder is never fine-tuned, so this is computed once and reused for every
    model, every fold and every seed.

    `embedder` selects the language model (see EMBEDDERS). Each writes to its own
    cache file and stamps its name into the .npz, because the two tensors are not
    interchangeable and nothing downstream could tell them apart from the numbers
    alone -- a 480-d AbLang2 cache and a 480-d ESM2-35M cache have the same shape
    and mean entirely different things.

    The observed token length is compared against len(VH) + len(VL) and reported.
    A constant offset means the model is emitting extra tokens -- a separator or
    start/end token -- which would otherwise be pooled in as if it were a residue.
    """
    if embedder not in EMBEDDERS:
        raise ValueError(f"model.embedder must be one of {sorted(EMBEDDERS)}, "
                         f"got {embedder!r}")
    tag = "ablang" if embedder == "ablang2" else embedder

    uniq = unique_fvs(df)
    path = embedding_cache_path(cache, uniq.key, max_len, embedder, layer)
    Path(cache).mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"[{tag}] cached -> {path.name}")
        return path

    try:                                   # both backends pull in torch; this is
        import torch                       # only for picking a device
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    if embedder == "ablang2":
        import ablang2
        model = ablang2.pretrained(model_to_use="ablang2-paired",
                                   random_init=False, device=device)
        model.freeze()
        rows_for = lambda pairs: _ablang_rows(model, pairs)          # noqa: E731
    else:
        esm, tok = _esm_model(EMBEDDERS[embedder], device)
        print(f"[{tag}] {EMBEDDERS[embedder]} on {device}")
        rows_for = lambda pairs: _esm_rows(esm, tok, device, pairs, layer)  # noqa: E731

    embs, masks, offsets, truncated = [], [], set(), 0
    for s in range(0, len(uniq), batch):
        chunk = uniq.iloc[s:s + batch]
        pairs = [(clean_seq(h), clean_seq(l)) for h, l in zip(chunk.VH, chunk.VL)]
        rows = rows_for(pairs)

        for r, (vh, vl) in zip(rows, pairs):
            offsets.add(int(r.shape[0]) - (len(vh) + len(vl)))
            L, D = r.shape
            e = np.zeros((max_len, D), dtype=np.float32)
            m = np.zeros(max_len, dtype=np.float32)
            n = min(L, max_len)
            truncated += int(L > max_len)
            e[:n], m[:n] = np.nan_to_num(r[:n], nan=0.0, posinf=0.0, neginf=0.0), 1.0
            embs.append(e); masks.append(m)

    np.savez_compressed(path, keys=uniq.key.to_numpy(),
                        X_ab=np.stack(embs), mask_ab=np.stack(masks),
                        token_offset=np.array(sorted(offsets)),
                        # Stamped so a bundle can refuse the wrong tensor. Shape
                        # alone cannot distinguish two 480-d embedders.
                        embedder=np.array(embedder), embedder_layer=np.array(layer),
                        embedder_model=np.array(EMBEDDERS[embedder] or "ablang2-paired"))
    print(f"[{tag}] {len(uniq)} molecules -> {path.name}  (dim {embs[0].shape[-1]})")
    if offsets != {0}:
        print(f"[{tag}] token length - (len(VH)+len(VL)) = {sorted(offsets)}. "
              "A non-zero constant means the model is emitting extra tokens "
              "(separator / start / end); they are being pooled in as residues.")
    if truncated:
        print(f"[{tag}] {truncated} molecule(s) longer than max_len={max_len} were "
              "truncated -- their C-terminal residues are not in the embedding")
    return path


# --------------------------------------------------------------------------- #
# loading back, aligned to a table
# --------------------------------------------------------------------------- #
def load_arrays(df: pd.DataFrame, cfg: Config,
                feat_path: Optional[Path] = None,
                emb_path: Optional[Path] = None,
                frame: Optional[Sequence[str]] = None,
                ) -> Tuple[dict, dict, pd.DataFrame]:
    """Arrays aligned row-for-row with the returned (possibly shrunk) table.

    Returns (arrays, meta, df). `meta` carries the provenance a bundle needs:
    feature names, global names, the feature-generation parameters and the frame
    actually used.

    frame : IMGT positions to scatter onto. Pass None during training to build the
            frame from these molecules; pass the bundle's frame at inference, so
            column j means the same residue it meant during training.

    Molecules missing any required array are dropped and counted, so training
    never silently runs on a different set than the table implies.
    """
    want = needs(cfg)
    arrays: dict = {}
    meta: dict = {"frame": None, "feature_names": None, "global_feature_names": None,
                  "feature_params": None}
    keep = np.ones(len(df), dtype=bool)

    if want["feat"]:
        if feat_path is None:
            raise ValueError("streams include 'feat' but no feature cache was given")
        d = np.load(feat_path, allow_pickle=True)
        missing = [k for k in ("X", "mask", "keys") if k not in d.files]
        if missing:
            raise ValueError(
                f"{feat_path}: not a {CACHE_FORMAT} cache (missing {missing}). "
                "Caches written before the X/Xg schema hold the extractor's raw "
                "output; rebuild with `madm prepare --overwrite`.")

        pos = {k: i for i, k in enumerate(d["keys"])}
        idx = np.array([pos.get(k, -1) for k in df.key])
        keep &= idx >= 0
        safe = np.where(idx >= 0, idx, 0)
        X = d["X"][safe].astype(np.float32)
        mask = d["mask"][safe].astype(np.float32)

        if "feature_names" in d.files:
            meta["feature_names"] = [str(s) for s in d["feature_names"]]
        if "global_feature_names" in d.files:
            meta["global_feature_names"] = [str(s) for s in d["global_feature_names"]]
        if "params" in d.files:
            meta["feature_params"] = json.loads(str(d["params"]))
            _check_params(meta["feature_params"], cfg, feat_path)

        if cfg.model.frame == "imgt":
            if "IMGT_vh" not in d.files:
                raise ValueError(
                    f"{feat_path}: model.frame is 'imgt' but the cache has no IMGT "
                    "numbering. Rebuild with `madm prepare --overwrite`, or set "
                    "model.frame: raw.")
            ivh = d["IMGT_vh"][safe]
            ivl = d["IMGT_vl"][safe]
            if frame is None:
                frame = build_frame_from(ivh, ivl)
            X, mask, ok = align_to_frame(X, mask, ivh, ivl, frame)
            keep &= ok
            meta["frame"] = list(frame)

        arrays["X"] = X
        arrays["mask"] = mask
        if cfg.model.use_global:
            if "Xg" not in d.files:
                raise ValueError(
                    f"{feat_path}: model.use_global is true but the cache has no "
                    "global features (Xg). Rebuild the cache, or set "
                    "model.use_global: false.")
            arrays["Xg"] = d["Xg"][safe].astype(np.float32)

    if want["ablang"]:
        if emb_path is None:
            raise ValueError("streams include 'ablang' but no embedding cache was given")
        d = np.load(emb_path, allow_pickle=True)
        pos = {k: i for i, k in enumerate(d["keys"])}
        idx = np.array([pos.get(k, -1) for k in df.key])
        keep &= idx >= 0
        safe = np.where(idx >= 0, idx, 0)
        arrays["X_ab"] = d["X_ab"][safe].astype(np.float32)
        arrays["mask_ab"] = d["mask_ab"][safe].astype(np.float32)
        # Caches written before the embedder switch carry no stamp; they are
        # AbLang2 by construction, since nothing else could have written them.
        meta["embedder"] = (str(d["embedder"]) if "embedder" in d.files
                            else "ablang2")
        meta["embedder_layer"] = (int(d["embedder_layer"]) if "embedder_layer" in d.files
                                  else -1)
        if meta["embedder_layer"] != cfg.model.embedder_layer:
            raise ValueError(
                f"{emb_path} holds layer {meta['embedder_layer']} but "
                f"model.embedder_layer is {cfg.model.embedder_layer}.")
        if meta["embedder"] != cfg.model.embedder:
            raise ValueError(
                f"{emb_path} holds {meta['embedder']!r} embeddings but "
                f"model.embedder is {cfg.model.embedder!r}. These are different "
                "representations of the same molecules, not interchangeable ones.")

    if not keep.all():
        print(f"[arrays] dropped {int((~keep).sum())} molecule(s) with no cached "
              f"{'features' if want['feat'] else 'embeddings'}")
    if not keep.any():
        raise ValueError(
            "every molecule was dropped: none of the table's (VH,VL) pairs are in "
            "the cache. Run `madm prepare` for this table first.")
    arrays = {k: v[keep] for k, v in arrays.items()}
    return arrays, meta, df[keep].reset_index(drop=True)


def build_frame_from(imgt_vh, imgt_vl):
    from .features import build_frame
    return build_frame(imgt_vh, imgt_vl)


def _check_params(cached: dict, cfg: Config, path) -> None:
    """Refuse a cache built with different feature settings than the config asks
    for. The filename already carries the signature, so this only fires when a
    file has been renamed or hand-copied -- which is exactly when it matters."""
    from dataclasses import asdict
    want = asdict(cfg.features)
    diff = {k: (cached.get(k), v) for k, v in want.items() if cached.get(k) != v}
    if diff:
        lines = "\n  ".join(f"{k}: cache={c!r} config={w!r}" for k, (c, w) in diff.items())
        raise ValueError(f"{path}: feature settings differ from config.yaml:\n  {lines}")
