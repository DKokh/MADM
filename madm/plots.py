"""Validation figures for predictions that have measurements to compare against.

Drawn only when the inference table carries the target column. Mirrors
`Scripts/Plots.plot_validation_v1`: one row per source plus a pooled row, each
with a truth-vs-prediction scatter and, when a threshold is set, an ROC curve.

Two conventions worth stating, because they are easy to get wrong and the figure
is where the mistake becomes visible:

  * every number on the figure is computed by `madm.metrics.score`, the same
    function the training loop reports, so a plot and a printed table can never
    disagree about what "Spearman" means.
  * CENSORING IS A REPORTING CHOICE, not implied by having a noise floor.
    `censor_metrics=False` (default) reports the raw measurement in original
    units; True clips truth and prediction at the floor first, on the grounds
    that ordering below the detection limit is ordering on noise. The floor is
    shaded on the scatter either way, and the panel states which convention it
    used rather than leaving the reader to guess.

    Model SELECTION is separate and always censored (madm.train.evaluate), since
    early-stopping on sub-floor ordering would stop on noise.

matplotlib is imported lazily so the package still installs and runs headless
without it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .metrics import score


def _fig(nrows: int, ncols: int, size=(4.2, 3.6)):
    import matplotlib
    matplotlib.use("Agg")                      # never needs a display
    import matplotlib.pyplot as plt
    _style()
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(size[0] * ncols, size[1] * nrows))
    return plt, fig, axes


def _style() -> None:
    """A seaborn-like look in plain matplotlib.

    Deliberately not seaborn: it would be one more optional dependency, one more
    thing to install on a fresh machine, and two rendering paths that can drift
    apart. Everything that made the figures worth looking at is a handful of
    rcParams and a bootstrap -- both below.

    Set per figure rather than globally at import, so importing madm.plots never
    changes the style of a caller's own plots.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.grid": True, "grid.color": "#d0d0d0",
        "grid.alpha": 0.5, "grid.linewidth": 0.6,
        # A closed box on every panel. Two open spines leave a scatter looking
        # unbounded -- you cannot see where the data range ends and the padding
        # begins -- and mixing open and closed panels in one figure reads as an
        # accident.
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.edgecolor": "#4d4d4d", "axes.linewidth": 0.9,
        "axes.axisbelow": True,               # grid behind the data, never over it
        # Plain white, not a tinted panel: the tint competes with the shaded
        # noise-floor region and with anything drawn at low alpha.
        "figure.facecolor": "white", "axes.facecolor": "white",
        # Sized to survive being dropped into a slide or shrunk to a journal
        # column -- the default 9pt is legible on screen at full size and
        # illegible at either of the places a figure actually ends up.
        "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "figure.titlesize": 14, "legend.frameon": True,
    })


def _trend(ax, x, y, color="#c0392b", n_boot: int = 1000):
    """Least-squares line with a 95% bootstrap confidence band.

    The band is the point. A slope drawn on eleven molecules looks exactly like a
    slope drawn on eight hundred, and the whole reason to plot per source is that
    those groups are small -- so the uncertainty has to be on the figure, not left
    to the reader to remember.

    Percentile bootstrap over resampled pairs: no normality assumption, and it
    widens where the data is thin, which is the behaviour wanted at the ends of
    the range.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return
    rng = np.random.default_rng(0)             # reproducible band
    xs = np.linspace(float(x.min()), float(x.max()), 100)

    b, a = np.polyfit(x, y, 1)
    ax.plot(xs, a + b * xs, color=color, lw=1.6, zorder=4)

    fits = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) == 0:                # degenerate resample, no slope
            continue
        bb, aa = np.polyfit(x[idx], y[idx], 1)
        fits.append(aa + bb * xs)
    if fits:
        lo, hi = np.percentile(np.vstack(fits), [2.5, 97.5], axis=0)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)


def validation_figure(truth, pred, out_path, title: str = "",
                      threshold: Optional[float] = None, higher_is_bad: bool = True,
                      noise_floor: Optional[float] = None, censor_metrics: bool = False,
                      sources=None, seed_preds=None, min_n: int = 5,
                      censor_side: str = "floor") -> dict:
    """Scatter (+ ROC) per source and pooled. Returns {group: metrics}.

    noise_floor    : shades the sub-detection region on the scatter. Does NOT by
                     itself change the reported number -- see censor_metrics.
    censor_metrics : clip truth and prediction at `noise_floor` before the rank
                     metrics. Default False: report the raw measurement. Set True
                     for the conservative reading, since ordering below the
                     detection limit is ordering on noise. The panel states which
                     convention it used either way, so a figure is never ambiguous.
    seed_preds     : [n_seeds, n] per-seed predictions, drawn as a faint cloud
                     behind the ensemble point, so the reader can see how much of
                     a position is initialisation noise rather than signal.
    """
    t = np.asarray(truth, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    sp = None if seed_preds is None else np.asarray(seed_preds, dtype=float)

    groups = [("ALL", np.ones(len(t), bool))]
    if sources is not None:
        s = np.asarray(sources).ravel()
        order = sorted(set(s), key=lambda x: -(s == x).sum())
        named = [(g, s == g) for g in order if (s == g).sum() >= min_n]
        # With a single source, "ALL" is the same rows drawn twice. Keep the
        # named one: it says WHICH source, and the pooled-vs-per-source
        # distinction only exists when there is more than one to pool.
        groups = named if len(named) == 1 else named + groups

    keep = [(g, m) for g, m in groups if int((m & np.isfinite(t) & np.isfinite(p)).sum()) >= min_n]
    if not keep:
        print("[plots] nothing to plot: fewer than "
              f"{min_n} molecules with both a prediction and a measurement")
        return {}

    ncols = 2 if threshold else 1
    plt, fig, axes = _fig(len(keep), ncols)
    out: Dict[str, dict] = {}
    scale_note: list = []

    for r, (name, m) in enumerate(keep):
        ok = m & np.isfinite(t) & np.isfinite(p)
        tt, pp = t[ok], p[ok]
        mt = score(tt, pp, censor_side=censor_side, threshold=threshold or 0.0,
                   higher_is_bad=higher_is_bad, noise_floor=noise_floor)
        out[name] = mt

        ax = axes[r][0]
        if sp is not None:
            for row in sp:
                ax.scatter(np.asarray(row)[ok], tt, s=12, alpha=0.10,
                           color="grey", edgecolor="none", zorder=1)
        # The ENSEMBLE point: large, black-edged, so it reads as the estimate and
        # the per-seed cloud behind it reads as the spread around it. Drawn after
        # the trend band would hide points under it, so zorder puts it on top.
        ax.scatter(pp, tt, s=85, alpha=0.9, color="#4a7fb5",
                   edgecolor="black", linewidth=0.8, zorder=5)
        # INDEPENDENT axes. A shared range forces predictions into a vertical
        # stripe whenever the model shrinks -- which is always, to some degree --
        # and then nothing about the ranking is visible. Each axis gets its own
        # data's range.
        xlo, xhi = float(pp.min()), float(pp.max())
        ylo, yhi = float(tt.min()), float(tt.max())
        xpad = 0.05 * (xhi - xlo + 1e-9); ypad = 0.05 * (yhi - ylo + 1e-9)
        xlo, xhi = xlo - xpad, xhi + xpad
        ylo, yhi = ylo - ypad, yhi + ypad

        # y = x only where it would actually appear. With predictions at 0-3 and
        # measurements at 0-160 it is a line in the corner that says nothing, and
        # drawing it implies the two axes are comparable when they are not.
        overlap = min(xhi, yhi) - max(xlo, ylo)
        comparable = overlap > 0.3 * max(xhi - xlo, yhi - ylo)
        if comparable:
            d0, d1 = min(xlo, ylo), max(xhi, yhi)
            ax.plot([d0, d1], [d0, d1], ls="--", c="grey", lw=0.9, zorder=3)
        _trend(ax, pp, tt)
        if threshold:
            ax.axhline(threshold, ls=":", c="grey", lw=0.8)
            ax.axvline(threshold, ls=":", c="grey", lw=0.8)
        if noise_floor is not None:
            # A property of the MEASUREMENT, so it shades the y axis -- and the
            # side that is censored, which is above the limit for a ceiling.
            lo, hi = ((noise_floor, yhi) if censor_side == "ceiling"
                      else (ylo, noise_floor))
            ax.axhspan(lo, hi, color="grey", alpha=0.08, zorder=0)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
        if not comparable:
            # To the CONSOLE, not the figure: the tick labels already show the two
            # ranges, so an annotation repeating them is clutter. What is worth
            # saying once is the interpretation -- excellent ranking with values on
            # the wrong scale is what validating against a different column from
            # the one the model trained on looks like.
            scale_note.append(
                f"{name}: predicted {pp.min():.3g}-{pp.max():.3g}, "
                f"actual {tt.min():.3g}-{tt.max():.3g}")
        rho = mt["rho_c"] if (censor_metrics and noise_floor is not None) else mt["rho"]
        ax.set_title(f"{name}  (n={mt['n']})\n"
                     f"Spearman {rho:.3f}   Pearson {mt['pearson']:.3f}")

        if threshold:
            ax2 = axes[r][1]
            lab = (tt > threshold) if higher_is_bad else (tt < threshold)
            if 0 < lab.sum() < lab.size:
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(lab, pp if higher_is_bad else -pp)
                ax2.plot(fpr, tpr, color="#c0392b", lw=1.8,
                         label=f"AUC = {mt['auc']:.3f}")
                ax2.legend(loc="lower right")
            else:
                ax2.text(0.5, 0.5, f"one class only\n({int(lab.sum())}/{lab.size} positive)",
                         ha="center", va="center", color="grey")
            ax2.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.9)
            ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
            ax2.set_xlabel("false positive rate"); ax2.set_ylabel("true positive rate")
            ax2.set_title(f"ROC - {name}")

    note = ("rank metrics censored at the noise floor"
            if (censor_metrics and noise_floor is not None)
            else "rank metrics in original units, uncensored")
    fig.suptitle(f"{title}\n{note}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] {out_path}")
    if scale_note:
        print("[plots] predictions and measurements are on different scales, so "
              "y = x is not drawn:")
        for s in scale_note:
            print(f"[plots]   {s}")
        print("[plots] Excellent ranking with the values on the wrong scale is "
              "what validating against a different column from the one the model "
              "was trained on looks like -- check `columns:` in the run's config.")
    return out


def seed_figure(truth, seed_preds, pred, out_path, title: str = "",
                threshold: Optional[float] = None, higher_is_bad: bool = True,
                noise_floor: Optional[float] = None, censor_metrics: bool = False,
                labels=None) -> dict:
    """One bar per seed model, plus the ensemble, on the same held-out set.

    A bundle is `n` models trained from different initialisations, and the number
    a release quotes belongs to the ENSEMBLE -- the score of the averaged
    predictions. That is not the mean of the per-seed scores and is usually above
    all of them, because averaging cancels initialisation noise that no single
    seed can. Plotting both makes two things visible that a single figure cannot:

      spread across the bars  how much of the headline is initialisation rather
                              than the model. Bars ranging 0.2-0.7 mean the next
                              retrain could land anywhere in that range.
      ensemble vs the bars    whether averaging is doing real work here. An
                              ensemble sitting at the top of a tight cluster of
                              bars is a stable model; one far above a scattered
                              set is a noisy model rescued by averaging, which is
                              worth knowing before quoting the number.

    Returns {"seeds": [per-seed metric dicts], "ensemble": metric dict}.
    """
    t = np.asarray(truth, float).ravel()
    e = np.asarray(pred, float).ravel()
    P = np.asarray(seed_preds, float)
    if P.ndim != 2 or P.shape[1] != t.size or len(P) < 2:
        return {}

    sfx = "_c" if (censor_metrics and noise_floor is not None) else ""
    kw = dict(threshold=threshold or 0.0, higher_is_bad=higher_is_bad,
              noise_floor=noise_floor)
    rows = [score(t, P[i], **kw) for i in range(len(P))]
    ens = score(t, e, **kw)

    keys = [("rho" + sfx, "Spearman"), ("pearson" + sfx, "Pearson")]
    if threshold is not None and np.isfinite(ens.get("auc" + sfx, np.nan)):
        keys.append(("auc" + sfx, "ROC-AUC"))
    names = list(labels) if labels is not None else [f"s{i + 1}" for i in range(len(P))]

    plt, fig, axes = _fig(1, len(keys), size=(0.42 * len(P) + 1.9, 3.6))
    for ax, (k, lab) in zip(axes[0], keys):
        v = np.array([r.get(k, np.nan) for r in rows], float)
        x = np.arange(len(v) + 1)
        ax.bar(x[:-1], v, width=0.7, color="#8fa6cc", edgecolor="#4d4d4d", lw=0.8,
               zorder=3)
        ax.bar(x[-1], ens.get(k, np.nan), width=0.7, color="#c0392b",
               edgecolor="black", lw=1.0, zorder=3)
        # The ensemble as a rule across the seed bars: the comparison this
        # figure exists for is vertical, and the eye does it badly between bars
        # that are far apart.
        ax.axhline(ens.get(k, np.nan), ls="--", c="#c0392b", lw=1.0, alpha=0.75,
                   zorder=4)
        for xi, vi in zip(x, list(v) + [ens.get(k, np.nan)]):
            if np.isfinite(vi):
                ax.annotate(f"{vi:.3f}", (xi, vi), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=9)
        ax.set_xticks(x, names + ["ens"])
        lo = float(np.nanmin(np.append(v, ens.get(k, np.nan))))
        ax.set_ylim(min(0.0, lo - 0.08), 1.0)
        ax.set_ylabel(lab)
        ax.set_title(f"{lab}   ensemble {ens.get(k, float('nan')):.3f}")
        ax.xaxis.grid(False)

    fig.suptitle(title + (f"   n={ens.get('n', 0)}" if title else ""))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] {out_path}")

    r = np.array([x.get("rho" + sfx, np.nan) for x in rows], float)
    if np.isfinite(r).sum() >= 2:
        print(f"  per-seed {('censored ' if sfx else '')}Spearman "
              f"{np.nanmin(r):.3f}-{np.nanmax(r):.3f} (mean {np.nanmean(r):.3f}); "
              f"ensemble {ens.get('rho' + sfx, float('nan')):.3f}. The ensemble is "
              "the score of the AVERAGED predictions, not the mean of these bars.")
    return {"seeds": rows, "ensemble": ens}


def calibration_figure(truth, pred, out_path, title: str = "",
                       noise_floor: Optional[float] = None) -> None:
    """Predicted vs actual distributions, and the residual against the actual
    value. A regularised model shrinks, and shrinkage is invisible in a rank
    metric -- these two panels are where it shows."""
    t = np.asarray(truth, float).ravel(); p = np.asarray(pred, float).ravel()
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    plt, fig, axes = _fig(1, 2, size=(4.6, 3.6))

    ax = axes[0][0]
    bins = np.histogram_bin_edges(np.concatenate([t, p]), bins=30)
    ax.hist(t, bins=bins, alpha=0.55, label=f"actual (sd {t.std():.3g})")
    ax.hist(p, bins=bins, alpha=0.55, label=f"predicted (sd {p.std():.3g})")
    if noise_floor is not None:
        ax.axvline(noise_floor, ls=":", c="grey", lw=1.0)
    ax.set_xlabel("value"); ax.set_ylabel("count"); ax.legend()
    ax.set_title(f"spread ratio {p.std() / (t.std() + 1e-12):.2f}")

    ax = axes[0][1]
    ax.scatter(t, p - t, s=20, alpha=0.7, color="#4a7fb5", edgecolor="white", linewidth=0.4)
    ax.axhline(0, ls="--", c="grey", lw=0.9)
    ax.set_xlabel("actual"); ax.set_ylabel("predicted - actual")
    ax.set_title("residual vs actual")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] {out_path}")
