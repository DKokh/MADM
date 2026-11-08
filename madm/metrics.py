"""Scoring. One function, one convention, used everywhere.

Spearman and AUC are invariant to any monotone transform, so they read the same in
z-space or original units. Pearson is not: on a log-scaled target it is dominated
by the few largest values and can disagree with Spearman by 0.1 or more. Both are
reported, in original units, and Spearman is the primary number.

Everything is computed in float64. In float32, np.maximum(pred, floor) rounds the
clip point differently for predictions and truths, which silently perturbs the
ranking of values sitting exactly at a censoring floor.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score


# Said once per process. A degenerate fold or source is worth knowing about;
# the same line for every one of 40 fold x source combinations is not.
_CONSTANT_SEEN: list = []


def score(y_true, y_pred, threshold: float = 0.0, higher_is_bad: bool = True,
          noise_floor: Optional[float] = None,
          censor_side: str = "floor") -> Dict[str, float]:
    """Both metric spaces, both censoring levels, in original units.

    noise_floor : values below it are indistinguishable from zero, so both truth
                  and prediction are clipped there before the *_c metrics.
    """
    t = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    out: Dict[str, float] = {"n": int(t.size)}
    if t.size < 3:
        return {**out, "rho": np.nan, "pearson": np.nan, "auc": np.nan,
                "rho_c": np.nan, "pearson_c": np.nan, "auc_c": np.nan}

    # A constant input makes both correlations undefined, and scipy says so once
    # per call without saying WHICH array -- which is the only interesting part.
    # A constant PREDICTION is a collapsed model: it returns the same value for
    # every molecule, so there is no ranking to score. A constant TRUTH is a
    # degenerate subset: one class, or every measurement at the censoring floor.
    # The two mean entirely different things and neither is a warning worth
    # repeating for every fold and source in a run.
    flat = [n for n, v in (("prediction", p), ("measurement", t)) if np.ptp(v) == 0]
    if flat:
        if not _CONSTANT_SEEN:
            print(f"[metrics] {' and '.join(flat)} is constant on {t.size} "
                  "molecule(s): rho and pearson are undefined (nan) here. A "
                  "constant prediction means the model returned one value for "
                  "every molecule; a constant measurement means the subset holds "
                  "a single distinct value.")
            _CONSTANT_SEEN.append(True)
        out["rho"] = out["pearson"] = np.nan
    else:
        out["rho"] = float(spearmanr(t, p).correlation)
        out["pearson"] = float(pearsonr(t, p)[0])

    def _auc(tt, pp):
        lab = (tt > threshold) if higher_is_bad else (tt < threshold)
        if not 0 < lab.sum() < lab.size:
            return np.nan            # one class only: AUC undefined, not 0.5
        return float(roc_auc_score(lab, pp if higher_is_bad else -pp))

    out["auc"] = _auc(t, p)

    # Both conventions, always. Whichever one a caller reports, the other is one
    # dictionary key away -- so a finished run can be re-read under the opposite
    # convention without recomputing anything. `target_opts.report_censored`
    # decides which pair is printed and plotted; both are written to disk.
    if noise_floor is not None:
        f = float(noise_floor)
        # Clip TOWARDS the limit, on whichever side is censored: ordering among
        # values past it is ordering on noise, so collapsing them is the point.
        clip = np.minimum if censor_side == "ceiling" else np.maximum
        tc, pc = clip(t, f), clip(p, f)
        out["rho_c"] = float(spearmanr(tc, pc).correlation)
        out["pearson_c"] = float(pearsonr(tc, pc)[0])
        # Usually equal to `auc`: clipping only reorders values below the floor,
        # and those sit on one side of a threshold that is normally above it. It
        # differs when noise_floor >= threshold, which is worth seeing rather than
        # assuming away.
        out["auc_c"] = _auc(tc, pc)
    else:
        out["rho_c"], out["pearson_c"], out["auc_c"] = (
            out["rho"], out["pearson"], out["auc"])
    return out


def summarise(rows, keys=("rho", "pearson", "auc")) -> str:
    import pandas as pd
    df = pd.DataFrame(rows)
    return "   ".join(f"{k} {df[k].mean():.3f} +/- {df[k].std():.3f}"
                      for k in keys if k in df)


def paired_delta(rows_a, rows_b, key: str = "rho", label: str = "") -> Dict[str, float]:
    """Mean of the per-run DIFFERENCES, not the difference of two means.

    Both arms see the same folds and seeds, so fold difficulty cancels in the
    paired difference. The difference of two means has no usable standard error.
    """
    import pandas as pd
    a = pd.DataFrame(rows_a)[key].to_numpy(dtype=float)
    b = pd.DataFrame(rows_b)[key].to_numpy(dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"unpaired: {a.shape} vs {b.shape}")
    d = a - b
    out = {"label": label, "n": int(d.size), "mean": float(np.nanmean(d)),
           "sd": float(np.nanstd(d)), "wins": int((d > 0).sum())}
    try:
        from scipy.stats import wilcoxon
        if np.nansum(np.abs(d)) > 0:
            out["p"] = float(wilcoxon(a, b).pvalue)
    except Exception:
        pass
    return out
