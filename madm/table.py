"""Input tables.

Two shapes:

  TRAINING   Ab_ID, VH, VL, <one column per assay>
             A missing assay value means "not measured". Nothing else is implied.

  INFERENCE  VH, VL  (Ab_ID optional -- generated if absent)
             Any assay column present is treated as ground truth and used to
             validate the predictions rather than as an input.

Two optional columns are carried through both shapes: `source`, which drives the
per-source breakdown at validation time, and `pdb`, which names a structure the
caller already has for that molecule (see prepare.build_structures).

EVERY USABLE SEQUENCE IS KEPT AND PREDICTED. Duplicates are reported, not
removed: which of two disagreeing measurements to keep is a decision about the
data, and silently dropping a row the caller asked about is worse than telling
them it repeats. Repeated Ab_IDs are suffixed so the output can still be joined
back row for row.

A sequence that cannot be folded or embedded -- empty, too short, non-standard
residues -- is WARNED about and skipped in training, and warned about but KEPT in
prediction, where one row per input row is part of the contract. Only a table
where nothing at all survives raises; so does a missing required column, or too
few labelled molecules to train on. Those cannot be worked around by reporting
them.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_seq(s) -> str:
    """Uppercase, letters only. Also the identity used for caching and grouping.

    A missing cell becomes the EMPTY string, not the text of whatever Python
    calls it. `str(nan)` is "nan" and `str(None)` is "None", and letters-only
    would keep those as three- and four-residue sequences -- so a blank VL, which
    is how a single-domain molecule is written in a spreadsheet, would arrive as
    a short peptide and be dropped as malformed instead of read as a VHH. Common
    text placeholders go the same way for the same reason.
    """
    if s is None or (isinstance(s, float) and s != s):        # None / NaN
        return ""
    t = "".join(c for c in str(s).upper().strip() if "A" <= c <= "Z")
    # Safe to special-case: no usable chain is four residues long, so these can
    # only be placeholders.
    return "" if t in ("NAN", "NONE", "NA", "NULL", "NIL", "EMPTY") else t


def fv_key(vh, vl) -> str:
    return clean_seq(vh) + "|" + clean_seq(vl)


def _read(path: str | Path, what: str = "table") -> pd.DataFrame:
    """`what` names the role of the file, so a missing path says which input is
    missing and which setting points at it -- `_read` serves both the training
    table and the prediction table, and "file not found" alone leaves the caller
    to work out which one the run wanted."""
    p = Path(path)
    if not p.exists():
        # A relative `paths.table` is resolved against the WORKING DIRECTORY, not
        # against config.yaml -- so the same config works or fails depending on
        # where madm was started. Saying which directory was searched, and what is
        # in it, turns that into a one-line fix instead of a guess.
        here = p.parent if p.parent.as_posix() != "" else Path(".")
        near = sorted(q.name for q in here.glob("*.*")) if here.is_dir() else []
        raise FileNotFoundError(
            f"MISSING {what.upper()}: {p} does not exist.\n"
            f"  looked in : {here.resolve()}\n"
            f"  cwd       : {Path.cwd()}\n"
            + (f"  that directory holds: {', '.join(near[:12])}"
               + (" ..." if len(near) > 12 else "")
               if near else f"  {here.resolve()} does not exist or is empty")
            + "\nRelative paths in config.yaml are resolved from the working "
              "directory; use an absolute path if you run madm from elsewhere.")
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    return pd.read_csv(p, sep="\t" if p.suffix.lower() == ".tsv" else ",")


def screen_sequences(df: pd.DataFrame, min_vh: int = 90, min_vl: int = 80,
                     drop: bool = True, what: str = "table") -> pd.DataFrame:
    """Report molecules that cannot be folded or embedded; drop them if asked.

    One unusable row out of eight hundred should not stop a run -- the same
    argument as a structure that fails to build. What it must not do is vanish
    silently, so every offending molecule is named with its length, and the
    reason is stated in a form you can act on.

    drop=True   (training) the row is removed, with a WARNING. Nothing downstream
                can use a sequence it cannot featurise, and the alternative is a
                run that dies after the expensive part.
    drop=False  (prediction) the row is kept, because `predict` returns one row
                per input row. It fails featurisation later and comes back with
                an empty prediction and a status.

    min_vl=0    single-domain mode: an EMPTY VL is allowed through as a VHH, and
                only a non-empty but too-short VL is flagged. A paired Fv with a
                blank VL is indistinguishable from a nanobody at this level, so
                this is opt-in rather than inferred -- with the default min_vl a
                blank VL stays what it usually is, a broken row.

    Raises only if NOTHING survives -- that is not one bad row, it is the wrong
    file or the wrong columns.
    """
    n0 = len(df)
    bad = pd.Series(False, index=df.index)
    for col, minlen in (("VH", min_vh), ("VL", min_vl)):
        s = df[col].map(clean_seq)
        blank = s.str.len() == 0
        # min_vl=0 turns a blank light chain from a defect into a declaration:
        # this is a single-domain molecule and there is no VL to check.
        allow_blank = (col == "VL" and minlen == 0)
        if allow_blank and blank.any():
            print(f"[table] {int(blank.sum())} row(s) have no VL -- treated as "
                  "single-domain (VHH). Structures come from NanoBodyBuilder2 and "
                  "the light-chain feature channels are zero for them.")
        empty = blank & (not allow_blank)
        short = (s.str.len() < minlen) & ~blank
        odd = pd.Series([bool(set(x) - AA) for x in s], index=df.index)

        for flag, label in ((empty, "empty"),
                            (short, f"shorter than {minlen}"),
                            (odd, "non-standard residues")):
            if not flag.any():
                continue
            rows = df.index[flag].tolist()
            who = [f"{df.at[i, 'Ab_ID'] if 'Ab_ID' in df.columns else i}"
                   f" (len {len(s.at[i])})" for i in rows[:5]]
            extra = ""
            if label == "non-standard residues":
                extra = f" {sorted({c for i in rows for c in set(s.at[i]) - AA})}"
            print(f"[table] WARNING {col}: {int(flag.sum())} sequence(s) {label}"
                  f"{extra}: {', '.join(who)}"
                  + (" ..." if len(rows) > 5 else ""))
        bad |= empty | short | odd

    if not bad.any():
        return df
    if drop:
        print(f"[table] dropped {int(bad.sum())} of {n0} row(s) from the {what}: "
              "they cannot be folded or embedded. Check them if that is a surprise "
              "-- a length of 0 is usually a blank spreadsheet row, and a short VH "
              "is often a VHH or a truncated entry.")
        df = df.loc[~bad].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"every row of the {what} was unusable ({n0} rows). That is not a "
                "data problem but a file problem -- check that VH and VL hold "
                "sequences and not identifiers.")
        return df
    print(f"[table] {int(bad.sum())} row(s) cannot be featurised -- kept, and they "
          "will come back unpredicted")
    return df


def report_duplicates(df: pd.DataFrame, assays: Sequence[str] = ()) -> dict:
    """Warn about repeated sequences and ids. Never drops a row.

    Every sequence the caller supplied is kept and predicted. Duplicates are a
    property of the input worth knowing about, not an error:

      * repeated (VH,VL) with DIFFERENT measured values is a real disagreement --
        the spread is printed, because it bounds how well any model can do
      * repeated (VH,VL) with the same value is harmless, just redundant compute
      * repeated Ab_ID makes the output ambiguous to join back, so the ids are
        made unique (a "#2", "#3" suffix) while the sequences are left untouched

    Structure prediction and embedding are keyed on the sequence, so a duplicate
    costs nothing beyond one row in the output.
    """
    out = {"dup_keys": 0, "dup_ids": 0, "conflicting": []}

    dup = df.key.duplicated(keep=False)
    if dup.any():
        n_groups = df.loc[dup, "key"].nunique()
        out["dup_keys"] = int(dup.sum())
        print(f"[table] NOTE {int(dup.sum())} row(s) share an identical (VH,VL) "
              f"across {n_groups} distinct sequence(s); all are kept and predicted")
        for a in assays:
            if a not in df.columns:
                continue
            v = pd.to_numeric(df[a], errors="coerce")
            for k, g in df.loc[dup].groupby("key"):
                vals = v.loc[g.index].dropna()
                if len(vals) > 1 and float(vals.max() - vals.min()) > 0:
                    out["conflicting"].append((a, list(g.Ab_ID)[:4],
                                               float(vals.min()), float(vals.max())))
        if out["conflicting"]:
            print(f"[table] WARNING {len(out['conflicting'])} duplicate group(s) carry "
                  "DIFFERENT measured values for the same sequence:")
            for a, ids, lo, hi in out["conflicting"][:5]:
                print(f"[table]   {a}: {ids} span {lo:.4g} .. {hi:.4g}")
            print("[table]   the model cannot separate them; this is a floor on the "
                  "achievable error")

    dup_id = df.Ab_ID.duplicated(keep=False)
    if dup_id.any():
        out["dup_ids"] = int(dup_id.sum())
        print(f"[table] NOTE {int(dup_id.sum())} duplicate Ab_ID value(s) "
              f"(e.g. {df.loc[dup_id, 'Ab_ID'].tolist()[:4]}); suffixing to keep them "
              "distinguishable in the output")
        seen: dict = {}
        new = []
        for x in df.Ab_ID:
            seen[x] = seen.get(x, 0) + 1
            new.append(x if seen[x] == 1 else f"{x}#{seen[x]}")
        df["Ab_ID"] = new
    return out


def resolve_assay_columns(df: pd.DataFrame, assays: Sequence[str],
                          columns: Optional[dict] = None,
                          where: str = "") -> dict:
    """{assay -> column in `df`}, raising with the candidates if one is absent.

    Assay tables in this project name their value column inconsistently: `SMP`
    and `Ovalbumin` plainly, the rest as `<assay>_rescaled`, and `AC-SINS` as
    `AC_SINS_rescaled` because a hyphen is awkward in a column name. Requiring
    the column to equal the assay name means renaming the file for every run,
    which is how two tables that differ only in a column name end up both being
    called "the data".

    `columns` (from `columns:` in config.yaml) wins where given -- explicit is
    better, and it is recorded in the run's config, so the choice travels with
    the numbers. Where it is absent, the obvious spellings are tried and the one
    used is PRINTED, because picking a column silently is how a run ends up
    trained on `_rescaled` while the paper says otherwise.
    """
    columns = columns or {}
    out, unresolved = {}, []
    for a in assays:
        if a in columns:                       # explicit mapping, no guessing
            if columns[a] not in df.columns:
                raise ValueError(
                    f"{where}: columns['{a}'] = {columns[a]!r} is not in the table. "
                    f"Found: {list(df.columns)}")
            out[a] = columns[a]
            continue
        u = a.replace("-", "_")
        for cand in (a, u, f"{a}_rescaled", f"{u}_rescaled"):
            if cand in df.columns:
                out[a] = cand
                break
        else:
            lower = {c.lower(): c for c in df.columns}
            hit = next((lower[c] for c in lower
                        if c.startswith(u.lower()) and "source" not in c), None)
            if hit:
                out[a] = hit
            else:
                unresolved.append(a)
    if unresolved:
        raise ValueError(
            f"{where}: no column found for assay(s) {unresolved}.\n"
            f"  tried: <assay>, <assay with _ for ->, and both + '_rescaled'\n"
            f"  table has: {list(df.columns)}\n"
            "  Name them explicitly in config.yaml if they differ, e.g.\n"
            "    columns: {Insulin: Insulin_rescaled, BVP: BVP_rescaled}")
    renamed = {a: c for a, c in out.items() if c != a}
    if renamed:
        print("[table] assay -> column: "
              + ", ".join(f"{a} -> {c!r}" for a, c in renamed.items()))
    return out


def load_train(path: str | Path, target: str, aux: Sequence[str] = (),
               columns: Optional[dict] = None, min_vl: int = 80) -> pd.DataFrame:
    """Training table. Returns columns Ab_ID, VH, VL, key + one column per assay,
    each named after the assay whatever it was called in the file."""
    df = _read(path, what="training table (paths.table in config.yaml)")
    missing = [c for c in ("VH", "VL") if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}. "
                         f"Found: {list(df.columns)}")

    # Resolve every assay to a column, then rename to the assay names so that
    # nothing downstream has to know what the file called them.
    resolved = resolve_assay_columns(df, [target] + list(aux), columns, where=str(path))
    df = df.rename(columns={c: a for a, c in resolved.items() if c != a})

    if "Ab_ID" not in df.columns:
        df["Ab_ID"] = [f"Ab{i:05d}" for i in range(len(df))]
    df = df.reset_index(drop=True)
    # Training has no row-preservation contract: a molecule that cannot be folded
    # or embedded contributes nothing, so it is named and dropped rather than
    # stopping a run that would otherwise train on the other 800.
    df = screen_sequences(df, min_vl=min_vl, what="training table")

    df["key"] = [fv_key(h, l) for h, l in zip(df.VH, df.VL)]
    report_duplicates(df, assays=[target] + list(aux))

    assays = [target] + list(aux)
    for a in assays:
        df[a] = pd.to_numeric(df[a], errors="coerce")
    n_t = int(np.isfinite(df[target]).sum())
    if n_t < 20:
        raise ValueError(f"only {n_t} finite {target} values -- too few to train")

    # Two optional columns survive the selection because something downstream
    # reads them: `pdb` names a structure the caller supplied (build_structures),
    # and `source` drives the per-source panels in the cross-validation figure.
    # A pooled correlation across sources measured on differently calibrated
    # instruments is partly the offset between their scales, so losing `source`
    # here means the training figure cannot show the thing most worth seeing.
    # A multi-assay table usually carries one source column PER assay
    # (`source_BVP`, `source_HIC`), because the same molecule can come from
    # different publications for different assays. The target's own column is the
    # right one: it says where THIS run's labels came from. Fall back to a plain
    # `source`, then to any `source_*` rather than silently having none.
    cand = [f"source_{target}", f"source_{target.replace('-', '_')}", "source"]
    src = ([c for c in cand if c in df.columns]
           or [c for c in df.columns if c.lower().startswith("source_")])
    if src:
        df["source"] = df[src[0]].astype(str).str.strip()
        if src[0] != "source":
            print(f"[table] source column: {src[0]!r}")
    keep = (["Ab_ID", "VH", "VL", "key"] + assays
            + (["pdb"] if "pdb" in df.columns else [])
            + (["source"] if src else []))
    out = df[keep].copy()
    print(f"[table] {len(out)} molecules   {target}: {n_t} labelled")
    for a in aux:
        n = int(np.isfinite(out[a]).sum())
        both = int((np.isfinite(out[a]) & np.isfinite(out[target])).sum())
        print(f"[table]   aux {a}: {n} labelled, {both} co-measured with {target}")
    return out


def load_predict(path: str | Path, truth: Sequence[str] = (),
                 truth_column: Optional[str] = None,
                 min_vl: int = 80) -> pd.DataFrame:
    """Inference table: VH and VL required, everything else optional.

    `truth` names assay columns to carry through if the file has them. They are
    NOT model input -- they are measurements the caller already has, used to
    validate the predictions afterwards. Absent columns are skipped silently, so
    the same call works for a blind table and a benchmark one.

    `truth_column` says which column holds `truth[0]` when it is not named after
    the assay -- a table whose measurements sit in `PSR` while the model's target
    is `SMP`. Renaming a column to satisfy a tool is the wrong way round, and the
    alternative to saying it explicitly is guessing, which is the wrong thing to
    do with the numbers a model is about to be judged against.

    If the table ALSO has a column named after the target, `truth_column` wins
    and the other is dropped with a warning naming both. Naming the column is
    the caller answering "which of these two?"; refusing would only send them
    away to edit the file and say it again.

    Every row is kept, in the input order, duplicates included -- including rows
    with an empty VH or VL. Those cannot be featurised, so they come back with an
    empty prediction and `status = no_structure_or_embedding` rather than
    stopping the run or vanishing from the output.
    """
    df = _read(path, what="prediction table (--table)")
    # A single-domain table has nothing to put in a VL column, so requiring the
    # column would only make the caller add an empty one. Supplied here instead
    # -- but ONLY when min_vl=0 has declared the table single-domain, because a
    # paired table that lost its VL column is a mistake worth stopping for.
    if "VL" not in df.columns and min_vl == 0:
        print("[table] no VL column: every molecule is treated as single-domain")
        df["VL"] = ""
    missing = [c for c in ("VH", "VL") if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing column(s) {missing}. Found: {list(df.columns)}"
            + ("\n  For single-domain antibodies (VHH), set split.min_len_vl: 0 "
               "in the config and the VL column becomes optional."
               if missing == ["VL"] else ""))
    if truth_column:
        if truth_column not in df.columns:
            raise ValueError(
                f"{path}: --truth-column {truth_column!r} is not in the table.\n"
                f"  columns: {list(df.columns)}")
        if not truth:
            raise ValueError("--truth-column given but this model has no target "
                             "to validate against")
        target = truth[0]
        if truth_column != target:
            if target in df.columns:
                # --truth-column IS the answer to "which of these two?", so it
                # wins -- naming it is the caller stating intent, and refusing
                # would only make them edit the file to repeat what they said.
                # The discarded column is named loudly instead, because scoring
                # against the wrong one produces a plausible number and no error,
                # and the log is where that has to be recoverable afterwards.
                n_t = int(pd.to_numeric(df[target], errors="coerce").notna().sum())
                n_c = int(pd.to_numeric(df[truth_column], errors="coerce").notna().sum())
                print(f"[table] WARNING both {target!r} ({n_t} values) and "
                      f"{truth_column!r} ({n_c} values) are present. "
                      f"--truth-column selects {truth_column!r} as ground truth; "
                      f"the existing {target!r} column is DISCARDED.")
                df = df.drop(columns=[target])
            df = df.rename(columns={truth_column: target})
            print(f"[table] ground truth: {truth_column!r} -> {target}")

    if "Ab_ID" not in df.columns:
        df["Ab_ID"] = [f"Ab{i:05d}" for i in range(len(df))]
    df = df.reset_index(drop=True)
    # drop=False: predict returns one row per input row, so an unusable sequence
    # is reported and kept. It fails featurisation and surfaces in the output as
    # status = no_structure_or_embedding.
    df = screen_sequences(df, min_vl=min_vl, drop=False,
                          what="prediction table")
    df["key"] = [fv_key(h, l) for h, l in zip(df.VH, df.VL)]

    have = [c for c in truth if c in df.columns]
    report_duplicates(df, assays=have)
    for c in have:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    src = [c for c in df.columns if c.lower() == "source" or c.lower().startswith("source_")]
    print(f"[table] {len(df)} molecules to predict "
          f"({df.key.nunique()} unique sequences)"
          + (f"   ground truth: {have}" if have else "")
          + (f"   source column: {src[0]}" if src else ""))
    if src:
        df["source"] = df[src[0]].astype(str).str.strip()
    # `pdb`, if present, points at a structure the caller supplied for that row --
    # read by build_structures, so it must survive the column selection.
    return df[["Ab_ID", "VH", "VL", "key"] + have
              + (["source"] if src else [])
              + (["pdb"] if "pdb" in df.columns else [])].copy()


def unique_fvs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct (VH,VL). Used for structure and feature generation, so
    an antibody appearing twice is only ever built once."""
    return df.drop_duplicates(subset="key").reset_index(drop=True)


def targets_matrix(df: pd.DataFrame, tasks: List[str]) -> np.ndarray:
    """[n_molecules, n_tasks] with NaN for "not measured"."""
    return np.column_stack([df[t].to_numpy(dtype=float) for t in tasks])
