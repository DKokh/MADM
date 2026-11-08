"""Command line: madm doctor | models | prepare | train | predict.

    madm doctor  --config config.yaml          # check the environment first
    madm train   --config config.yaml --model 2s
    madm predict --run runs/SMP_... --table new.csv --out p.csv

`doctor` exists because the expensive steps are the late ones. Structures and
embeddings are built hours in, so a partly-installed environment fails after the
costly part, with a traceback from inside someone else's package. It reports per
stream and exits non-zero only on what the given config actually needs.

ONE COMMAND IS THE WHOLE WORKFLOW. `train` and `predict` both build structures,
3D features and embeddings for whatever they are given, so `prepare` is a
convenience for doing the slow part separately, never a prerequisite.

`--model feat | ablang | 2s | mt` selects one of the four models without editing
the config. Anything but `mt` also clears `aux`: otherwise `--model 2s` on a
multi-task config stays multi-task and the two runs hash to the same directory.

`predict` validates when it can. A target column in the inference table is treated
as ground truth -- never as input -- and produces scores, a per-source breakdown
and figures alongside the predictions.

`--pdb-dir` (config `paths.structures_in`) supplies your own structures; only the
molecules you have no file for are predicted, and a complete set means
ImmuneBuilder is never imported. Each supplied file is checked against its row's
VH/VL first -- features come off the STRUCTURE, so a mis-named file would pair one
antibody's geometry with another's label and nothing downstream would show it.
At predict time the bundle's own `structures_in` is ignored: it described the
training machine, not this table.

Caching is by sequence hash AND feature-settings hash, so it is safe to re-run
after adding rows (existing molecules are not rebuilt) and impossible to reuse a
cache built with different settings. Neither hash covers WHICH .pdb a molecule
came from, so the cache records that separately and refuses itself if it changed.

Cache files are addressed, never searched for. `sorted(glob(...))[-1]` picks the
lexicographically largest content hash, which has nothing to do with the table in
hand -- at predict time it reliably selects the TRAINING cache and then drops
every new molecule.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# torch is imported inside the commands that need it, so `madm --help` and
# `madm prepare` work without a deep-learning install.
from .config import Config
from .data import (Calibrator, add_aux_only, cluster_sequences, make_folds,
                   training_indices)
from .metrics import score
from .plots import calibration_figure, seed_figure, validation_figure
from .prepare import (build_embeddings, build_features, build_structures,
                      embedding_cache_path, feature_cache_path, load_arrays, needs)
from .table import load_predict, load_train, targets_matrix, unique_fvs


MODELS = {
    "feat":   (["feat"],           False),   # structure features only
    "ablang": (["ablang"],         False),   # frozen AbLang2 embeddings only
    "2s":     (["feat", "ablang"], False),   # two-stream late fusion, single task
    "mt":     (["feat", "ablang"], True),    # ... plus the auxiliary assays
}


class _Tee:
    """Duplicate a stream to a file, buffering until the file is known.

    A run prints the things worth keeping -- which column each assay came from,
    which molecules were dropped and why, the structure provenance, the fold
    scores -- and then prints thousands of lines of training progress on top of
    them. By the time something looks wrong the explanation has scrolled away,
    and on a killed or backgrounded run it was never on screen at all.

    Buffered start, attached later: `train` only knows its run directory after
    the config is read and hashed, but the messages worth keeping begin before
    that. Everything printed in the meantime is held and written once the path
    exists, so the log has no gap at the top.
    """

    def __init__(self, stream):
        self.stream, self.pending, self.fh = stream, [], None

    def write(self, s):
        self.stream.write(s)
        if self.fh is not None:
            self.fh.write(s)
        else:
            self.pending.append(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        if self.fh is not None:
            self.fh.flush()

    def isatty(self):
        # tqdm and friends ask; the answer is about the CONSOLE, not the file.
        return getattr(self.stream, "isatty", lambda: False)()

    def attach(self, fh):
        self.fh = fh
        for s in self.pending:
            fh.write(s)
        self.pending.clear()
        fh.flush()


def log_to(path: str | Path) -> None:
    """Send everything printed from here on -- and everything already printed --
    to `path` as well as to the console. Quietly does nothing if stdout was not
    wrapped (library use), so importing madm never redirects a caller's output."""
    import sys
    outs = [s for s in (sys.stdout, sys.stderr) if isinstance(s, _Tee)]
    if not outs or any(s.fh is not None for s in outs):
        return                      # not wrapped, or already logging somewhere
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a", encoding="utf-8", buffering=1)
    fh.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S}  "
             f"{' '.join(sys.argv)} ===\n")
    for s in outs:
        s.attach(fh)                # one file, both streams: stdout and stderr
    print(f"[log] {p.resolve()}")   # interleaved as they happened


def _apply_overrides(cfg: Config, args) -> Config:
    """CLI flags beat the config file, so the four models in the paper can be run
    from one config. `--model` sets `streams`, and for anything but `mt` it also
    CLEARS `aux` -- otherwise `--model 2s` on a multi-task config would silently
    stay multi-task and the two runs would land in the same directory."""
    if getattr(args, "model", None):
        streams, keep_aux = MODELS[args.model]
        cfg.streams = list(streams)
        if not keep_aux and cfg.aux:
            print(f"[config] --model {args.model} is single-task: dropping aux {cfg.aux}")
            cfg.aux = []
    if getattr(args, "aux", None) is not None:
        cfg.aux = [a.strip() for a in args.aux.split(",") if a.strip()]
        if cfg.aux and getattr(args, "model", None) in ("feat", "ablang", "2s"):
            print(f"[config] --aux given with --model {args.model}: keeping the "
                  "auxiliaries, so this is a multi-task run on those streams")
    if getattr(args, "target", None):
        cfg.target = args.target
    if getattr(args, "table", None) and getattr(args, "_table_is_train", False):
        cfg.paths.table = args.table
    if getattr(args, "pdb_dir", None):
        cfg.paths.structures_in = args.pdb_dir
    if getattr(args, "check_pdb", None) is False:
        cfg.check_pdb_sequence = False
    cfg.validate()
    return cfg


def _prepare_inputs(cfg: Config, df: pd.DataFrame, overwrite: bool = False):
    """Structures, features and embeddings for `df`, built if absent.

    Called by train AND predict, so neither depends on `madm prepare` having been
    run first -- the whole workflow is reachable from one command. Everything is
    cached on the sequence hash, so a second call costs a directory listing.
    """
    want = needs(cfg)
    feat = emb = None
    if want["feat"]:
        df = build_structures(df, cfg.paths.structures, overwrite=overwrite,
                              pdb_dir=cfg.paths.structures_in,
                              check_sequence=cfg.check_pdb_sequence)
        feat = build_features(df, cfg.paths.features, cfg.reference_tables,
                              cfg.features, overwrite=overwrite)
    if want["ablang"]:
        emb = build_embeddings(df, cfg.paths.features, cfg.features.max_len,
                               embedder=cfg.model.embedder,
                               layer=cfg.model.embedder_layer,
                               overwrite=overwrite)
    return df, feat, emb


def _cache_paths(cfg: Config, df: pd.DataFrame, want: dict):
    """The exact caches for THIS table, or a clear error naming what to run."""
    feat = emb = None
    if want["feat"]:
        keys = unique_fvs(df[df.pdb != ""]).key if "pdb" in df.columns else df.key
        feat = feature_cache_path(cfg.paths.features, keys, cfg.features)
        if not feat.exists():
            raise FileNotFoundError(
                f"no feature cache for this table at {feat}.\n"
                "Run `madm prepare --config <cfg>` first. If you changed anything "
                "under `features:` the old cache no longer applies -- the settings "
                "hash is part of the filename.")
    if want["ablang"]:
        emb = embedding_cache_path(cfg.paths.features, unique_fvs(df).key,
                                   cfg.features.max_len, cfg.model.embedder,
                                   cfg.model.embedder_layer)
        if not emb.exists():
            raise FileNotFoundError(
                f"no embedding cache for this table at {emb}.\n"
                "Run `madm prepare --config <cfg>` first.")
    return feat, emb


# --------------------------------------------------------------------------- #
def cmd_prepare(args):
    cfg = _apply_overrides(Config.load(args.config), args)
    want = needs(cfg)
    df = load_train(cfg.paths.table, cfg.target, cfg.aux, columns=cfg.columns,
                    min_vl=cfg.split.min_len_vl)
    print(f"[prepare] model={args.model or '(config)'}  streams={cfg.streams} -> "
          f"structures/features: {want['feat']}, embeddings: {want['ablang']}")
    _prepare_inputs(cfg, df, overwrite=args.overwrite)
    print("[prepare] done")


def _cv_figures(cfg, oof_tab, seed_cols, run: Path, ens_m: dict) -> None:
    """Draw the cross-validation figures, or say why not.

    matplotlib is an optional dependency (`.[plots]`), and a missing plotting
    library must not lose a training run that has already finished. So this
    reports and returns rather than raising -- the numbers are in
    cv_metrics.csv and oof_predictions.csv either way, and the figures can be
    drawn later from those.
    """
    try:
        from .plots import calibration_figure, validation_figure
    except ImportError:
        print("\n[train] no matplotlib: skipping the CV figures. Everything they "
              "show is in oof_predictions.csv.\n        pip install -e '.[plots]'")
        return

    opt = cfg.target_opts
    cens = bool(opt.report_censored) and opt.noise_floor is not None
    t = oof_tab["true"].to_numpy(float)
    p = oof_tab["pred"].to_numpy(float)
    try:
        validation_figure(
            t, p, run / "cv_validation.png",
            title=f"{cfg.target} - out-of-fold ({cfg.split.cluster}/{cfg.split.folds}, "
                  f"{cfg.split.n_folds} folds)",
            threshold=opt.threshold, higher_is_bad=opt.higher_is_bad,
            noise_floor=opt.noise_floor, censor_metrics=cens,
            sources=oof_tab["source"].to_numpy() if "source" in oof_tab else None,
            # one line per seed, so the spread on the plot is initialisation
            # noise rather than anything about the data
            seed_preds=oof_tab[seed_cols].to_numpy().T if seed_cols else None)
        # The CALIBRATED column, deliberately: raw predictions are z-scored and
        # shrunk, so an uncalibrated plot shows a slope that no longer exists by
        # the time anyone uses the model.
        calibration_figure(
            t, oof_tab.get("pred_calibrated", oof_tab["pred"]).to_numpy(float),
            run / "cv_calibration.png",
            title=f"{cfg.target} - calibration ({cfg.target_opts.calibrate})",
            noise_floor=opt.noise_floor)
        # One panel PER FOLD. The pooled plot above is the honest headline --
        # every molecule scored once by a model that never saw it -- but it
        # averages away how unevenly the folds behave. With one cluster holding a
        # fifth of an assay, the fold that contains it is a different experiment
        # from the other four, and that shows here and nowhere else.
        if "fold" in oof_tab and oof_tab.fold.nunique() > 1:
            validation_figure(
                t, p, run / "cv_by_fold.png",
                title=f"{cfg.target} - out-of-fold, one panel per fold",
                threshold=opt.threshold, higher_is_bad=opt.higher_is_bad,
                noise_floor=opt.noise_floor, censor_metrics=cens,
                sources=np.array([f"fold {f}" for f in oof_tab.fold]))
    except Exception as e:                      # noqa: BLE001
        print(f"\n[train] WARNING could not draw the CV figures ({type(e).__name__}: "
              f"{str(e)[:80]}). The run is unaffected; see oof_predictions.csv.")
        return
    print("\n  cv_validation.png   out-of-fold truth vs prediction, per source + ROC")
    print("  cv_by_fold.png      the same, one panel per fold")
    print("  cv_calibration.png  predicted vs measured spread, and the residual")


# --------------------------------------------------------------------------- #

def cmd_train(args):
    import torch
    from .predict import save_bundle
    from .train import arch_kwargs, cross_validate, fit_final
    cfg = _apply_overrides(Config.load(args.config), args)
    device = cfg.device if torch.cuda.is_available() else "cpu"
    run = cfg.run_dir()
    cfg.save(run / "config.yaml")
    # Alongside config.yaml, so the run directory holds both what was asked for
    # and what happened. No-op when --log already named a file.
    log_to(run / "run.log")

    df = load_train(cfg.paths.table, cfg.target, cfg.aux, columns=cfg.columns,
                    min_vl=cfg.split.min_len_vl)
    want = needs(cfg)
    # build whatever is missing rather than requiring `madm prepare` first
    df, feat_path, emb_path = _prepare_inputs(cfg, df)
    # frame=None -> build the IMGT frame from these molecules and return it in meta
    arrays, meta, df = load_arrays(df, cfg, feat_path, emb_path)
    if meta["frame"]:
        print(f"[frame] IMGT frame: {len(meta['frame'])} positions")
    Y = targets_matrix(df, cfg.tasks)

    # has_t: scorable molecules -- they define the folds and every metric.
    # aux_only: auxiliary labels but no target value; trained on, never scored.
    has_t, aux_only = training_indices(Y)

    # Does the AUC threshold actually split this assay? It is in ORIGINAL units
    # and the default (0.3) is an SMP-scale number, so on an assay measured in
    # minutes or absorbance every molecule lands on one side and AUC is undefined
    # -- printed as `nan` on every fold, which reads like a failure rather than a
    # setting. Said once, with the numbers needed to fix it.
    _t = Y[has_t, 0]
    _pos = float(np.mean(_t > cfg.target_opts.threshold))
    if _pos in (0.0, 1.0):
        q = np.percentile(_t, [10, 50, 90])
        print(f"\n[train] WARNING every {cfg.target} value is on one side of "
              f"target_opts.threshold = {cfg.target_opts.threshold} (original "
              f"units), so AUC is undefined and will print as nan.")
        print(f"[train]   {cfg.target} spans {_t.min():.3g} to {_t.max():.3g}; "
              f"10th/50th/90th percentile = {q[0]:.3g} / {q[1]:.3g} / {q[2]:.3g}")
        print("[train]   Set a threshold that splits your panel. Spearman, Pearson "
              "and model selection are unaffected either way.")
    elif min(_pos, 1 - _pos) < 0.05:
        print(f"\n[train] NOTE threshold {cfg.target_opts.threshold} puts only "
              f"{min(_pos, 1 - _pos):.1%} of molecules in the smaller class; AUC "
              "will be unstable across folds.")

    clust = cluster_sequences(df.VH.iloc[has_t].tolist(), df.VL.iloc[has_t].tolist(),
                              kind=cfg.split.cluster, threshold=cfg.split.threshold,
                              k=cfg.split.k, min_len_vh=cfg.split.min_len_vh,
                              min_len_vl=cfg.split.min_len_vl)
    sizes = np.unique(clust, return_counts=True)[1]
    print(f"[split] cluster={cfg.split.cluster} folds={cfg.split.folds}: "
          f"{len(sizes)} clusters over {len(has_t)} molecules "
          f"(largest {sizes.max()}) at threshold {cfg.split.threshold}")
    if cfg.split.folds == "random":
        print("[split] RANDOM folds: near-duplicates span train and test. "
              "Scores are optimistic relative to grouped folds.")
    rounds_rel = make_folds(clust, cfg.split.n_folds, cfg.split.folds, cfg.split.seed)
    rounds = [(has_t[a], has_t[b]) for a, b in rounds_rel]

    # Aux-only molecules join every TRAINING fold. Not leakage: the molecules
    # being scored are still absent from training entirely, which `add_aux_only`
    # asserts per fold. Only `mt` benefits -- with no auxiliaries their rows carry
    # nothing `masked_loss` can use, so it skips them anyway.
    if len(aux_only):
        rounds = add_aux_only(rounds, aux_only)
        print(f"[split] + {len(aux_only)} molecule(s) with auxiliary labels but no "
              f"{cfg.target} value added to every training fold "
              f"({len(has_t)} scored, {len(has_t) + len(aux_only)} trained on)")
    elif cfg.aux:
        print(f"[split] no molecule carries an auxiliary label without a "
              f"{cfg.target} value -- nothing to add to the training folds")

    # the final models train on everything that carries ANY label, for the same
    # reason; `has_t` alone would discard the aux-only molecules a multi-task
    # model exists to exploit
    train_all = np.union1d(has_t, aux_only) if len(aux_only) else has_t

    print(f"\n=== {cfg.target}  streams={'+'.join(cfg.streams)}  "
          f"{'multi-task ' + str(cfg.aux) if cfg.aux else 'single-task'}  "
          f"{cfg.split.cluster}/{cfg.split.folds} folds ===")
    rows, oof = cross_validate(cfg, arrays, Y, rounds, device)
    cv = pd.DataFrame(rows)
    cv.to_csv(run / "cv_metrics.csv", index=False)

    # ---- per-seed and ensemble performance --------------------------------
    print("\nper seed (mean over folds):")
    for s, g in cv.groupby("seed"):
        print(f"  seed {s:<6} rho {g.rho.mean():.3f}   pearson {g.pearson.mean():.3f}"
              f"   auc {g.auc.mean():.3f}")
    print("\nacross all fold x seed runs:")
    for k in ("rho", "pearson", "auc"):
        print(f"  {k:<8} {cv[k].mean():.3f} +/- {cv[k].std():.3f}")
    if "w_feat" in cv:
        print(f"  w_feat   {cv.w_feat.mean():.3f}  (weight on the feature stream)")

    # ---- out-of-fold table: every molecule scored exactly once per seed ----
    o = pd.DataFrame(oof)
    wide = o.pivot_table(index="row", columns="seed", values="pred")
    wide.columns = [f"pred_seed{c}" for c in wide.columns]
    # `source` where the table has it: the CV figure draws one panel per source,
    # and a pooled correlation across sources measured on differently calibrated
    # instruments is partly the offset between their scales rather than ranking
    # within either. Without this column every molecule falls in one group and
    # the figure is a single panel that cannot show it.
    carry = ["Ab_ID", "VH", "VL"] + (["source"] if "source" in df.columns else [])
    oof_tab = pd.concat([
        df.loc[wide.index, carry].reset_index(drop=True),
        o.groupby("row").agg(fold=("fold", "first"), true=("true", "first")).reset_index(drop=True),
        wide.reset_index(drop=True)], axis=1)
    seed_cols = [c for c in oof_tab.columns if c.startswith("pred_seed")]
    oof_tab["pred"] = oof_tab[seed_cols].mean(axis=1)     # ensemble, out of fold
    oof_tab["pred_sd"] = oof_tab[seed_cols].std(axis=1)

    ens_m = score(oof_tab["true"], oof_tab["pred"],
                  threshold=cfg.target_opts.threshold,
                  higher_is_bad=cfg.target_opts.higher_is_bad,
                  noise_floor=cfg.target_opts.noise_floor,
                  censor_side=cfg.target_opts.censor_side)
    print(f"\nseed-averaged, pooled over all folds ({ens_m['n']} molecules):")
    print(f"  rho {ens_m['rho']:.3f}   pearson {ens_m['pearson']:.3f}   auc {ens_m['auc']:.3f}")
    print("  (this is what the saved ensemble does; the per-run numbers above are"
          " single models)")

    # ---- calibration: put predictions back on the training value scale -----
    # Fitted on OUT-OF-FOLD predictions, so it corrects the model's shrinkage
    # rather than its optimism. Monotone, so rho and auc cannot change.
    cal = Calibrator.fit(oof_tab["pred"], oof_tab["true"], cfg.target_opts.calibrate)
    oof_tab["pred_calibrated"] = cal.apply(oof_tab["pred"])
    oof_tab.to_csv(run / "oof_predictions.csv", index=False)
    if cal.kind != "none":
        cm = score(oof_tab["true"], oof_tab["pred_calibrated"],
                   threshold=cfg.target_opts.threshold,
                   higher_is_bad=cfg.target_opts.higher_is_bad)
        t_sd = float(np.std(oof_tab["true"]))
        print(f"\ncalibration ({cal.kind}), fitted out-of-fold:")
        print(f"  spread  true {t_sd:.3f} | raw {np.std(oof_tab['pred']):.3f} "
              f"-> calibrated {np.std(oof_tab['pred_calibrated']):.3f}")
        print(f"  rho     {ens_m['rho']:.3f} -> {cm['rho']:.3f}   (monotone: unchanged)")
        print(f"  pearson {ens_m['pearson']:.3f} -> {cm['pearson']:.3f}")
        print(f"  auc     {ens_m['auc']:.3f} -> {cm['auc']:.3f}   (monotone: unchanged)")

    # ---- the same two figures `predict` draws, on the OUT-OF-FOLD predictions -
    # Every molecule here was predicted by a model that never saw it, so this is
    # the honest picture of the run and the one to show. A single rho hides what
    # the scatter makes obvious: a source sitting on its own offset, a handful of
    # points carrying the correlation, or predictions collapsed to the mean --
    # which is invisible in a rank metric and is exactly what the calibration
    # panel is for.
    _cv_figures(cfg, oof_tab, seed_cols, run, ens_m)

    # ---- final models: one per seed, all labelled molecules, fixed epochs ---
    n_ep = _median_epoch(cv, "epoch", cfg.train.epochs)
    # Per stream, not one number for both. The streams converge at very different
    # rates -- frozen embeddings against 3D features -- so a shared budget
    # overtrains whichever stopped first and undertrains the other, and those are
    # the weights the fused model starts from.
    n_warm = {s: _median_epoch(cv, f"warm_epoch_{s}", cfg.train.warm_epochs)
              for s in cfg.streams} if cfg.train.warm_start else {}
    # The epoch budget is INHERITED from CV, where a genuine held-out fold
    # existed; there is none here, so nothing can be early-stopped. Two ways that
    # number stops being evidence, both silent until now:
    _ep = cv["epoch"] if "epoch" in cv else pd.Series(dtype=float)
    if not len(_ep) or (_ep >= 0).sum() == 0:
        print(f"[train] WARNING no valid best epoch came out of CV; falling back to "
              f"train.epochs={cfg.train.epochs}, which is a configured cap rather "
              "than a measurement")
    elif (_ep >= cfg.train.epochs - 2).mean() > 0.25:
        print(f"[train] WARNING {(_ep >= cfg.train.epochs - 2).mean():.0%} of CV runs "
              f"stopped at the epochs={cfg.train.epochs} ceiling -- the median is a "
              "property of the cap, not of convergence. Raise train.epochs.")

    print(f"\nfitting {len(cfg.fit_seeds)} final models on {len(train_all)} labelled "
          f"molecules ({len(has_t)} with a {cfg.target} value"
          + (f", {len(aux_only)} auxiliary-only" if len(aux_only) else "")
          + f") for {n_ep} epochs (median best epoch in CV"
          + (", warm start " + ", ".join(f"{s}={n}" for s, n in n_warm.items())
             if n_warm else "")
          + ", no early stopping)...")
    if len(set(n_warm.values())) > 1:
        # Worth seeing: it is the whole reason these are tracked separately, and
        # a large gap says the two streams are converging on different timescales.
        print(f"[train] warm-start budgets differ by stream {dict(n_warm)} -- each "
              "is the median of that stream's own stopping epoch in CV")
    models, tfs, fm, fs = [], None, None, None
    aux_weight_log = []
    for seed in cfg.fit_seeds:
        r = fit_final(cfg, arrays, Y, train_all, n_ep, seed, device, warm_epochs=n_warm)
        models.append(r["model"]); tfs, fm, fs = r["tfs"], r["feat_mean"], r["feat_sd"]
        am, asd = r.get("ab_mean"), r.get("ab_sd")
        # No per-seed .pt file: `bundle.pt` already carries every seed's
        # state_dict, and a second copy on disk is one more thing that can go
        # stale and be loaded by mistake.
        print(f"  seed {seed} fitted"
              + (f"   w_feat={r['blend'][cfg.target]:.3f}" if r["blend"] else ""))
        # Learned auxiliary weights. Printed per seed because an auxiliary the
        # model switches off (weight -> 0) is a result -- it says that donor
        # transfers nothing -- and because the target's share of the loss drifts
        # as they fall, so "half the loss" stops being true after the first epoch
        # and nothing else in the run records that.
        if r.get("aux_weights"):
            aw = {a: round(w, 3) for a, w in r["aux_weights"].items() if a != cfg.target}
            print(f"           learned aux weights: {aw}")
            aux_weight_log.append(r["aux_weights"])
    arch = arch_kwargs(cfg, arrays)
    save_bundle(run / "bundle.pt", models, cfg, tfs, fm, fs, arch,
                seeds=cfg.fit_seeds, epochs=n_ep, calibrator=cal, meta=meta,
                ab_mean=am, ab_sd=asd)
    (run / "summary.json").write_text(json.dumps(
        {"target": cfg.target, "streams": cfg.streams, "aux": cfg.aux,
         "split": {"cluster": cfg.split.cluster, "folds": cfg.split.folds,
                   "threshold": cfg.split.threshold, "n_clusters": int(len(sizes))},
         "frame": cfg.model.frame,
         "n_frame_positions": len(meta["frame"]) if meta["frame"] else None,
         # n_scored is what every metric refers to; n_trained includes the
         # auxiliary-only molecules, which are trained on but never scored
         "n": int(len(has_t)),
         "n_scored": int(len(has_t)),
         "n_trained": int(len(train_all)),
         "n_aux_only": int(len(aux_only)),
         "n_models": len(models), "seeds": cfg.fit_seeds,
         "cv_seeds": cfg.train.seeds,
         "final_epochs": n_ep, "warm_epochs": dict(n_warm),
         "epochs_from": ("cv_median" if "epoch" in cv and (cv["epoch"] >= 0).any()
                         else "config_fallback"),
         "single_model": {"rho": float(cv.rho.mean()), "rho_sd": float(cv.rho.std()),
                          "pearson": float(cv.pearson.mean()), "auc": float(cv.auc.mean())},
         "ensemble_oof": {"rho": ens_m["rho"], "pearson": ens_m["pearson"],
                          "auc": ens_m["auc"]},
         "calibrate": cal.kind,
         "aux_weighting": cfg.train.aux_weighting,
         "aux_weights_final": (
             {a: float(np.mean([w[a] for w in aux_weight_log]))
              for a in aux_weight_log[0]} if aux_weight_log else None)}, indent=2))
    print(f"\nrun -> {run}")
    print("  cv_metrics.csv      one row per fold x seed")
    print("  oof_predictions.csv one row per molecule, per-seed + ensemble")
    print("  cv_validation.png   out-of-fold truth vs prediction")
    print("  cv_calibration.png  spread and residual")
    print("  bundle.pt           all seeds + transforms + feature provenance")


def _fmt(x) -> str:
    """`n/a` rather than `nan` for a metric that could not be computed.

    A printed `nan` reads like a failure; AUC with one class present is simply
    not defined, which is a different thing and has a different fix."""
    return f"{x:.3f}" if np.isfinite(x) else "n/a"


def _median_epoch(cv: pd.DataFrame, col: str, fallback: int) -> int:
    """Median best epoch over CV runs, as a 1-based epoch COUNT.

    `epoch` is a 0-based index, so a model whose best epoch was 0 stopped after
    one epoch -- dropping those rows as "no data" biases the count upwards."""
    if col not in cv:
        return int(fallback)
    ep = cv[col][cv[col] >= 0]
    return int(np.median(ep)) + 1 if len(ep) else int(fallback)


# --------------------------------------------------------------------------- #
def cmd_predict(args):
    import torch
    from .predict import Ensemble
    run = Path(args.run)
    # Beside the predictions rather than in the model directory: a released model
    # may be read-only or shared, and this log describes THIS table.
    log_to(Path(args.out).with_suffix("").as_posix() + ".log")
    ens = Ensemble(run / "bundle.pt", device="cuda" if torch.cuda.is_available() else "cpu")
    print(ens)
    cfg = Config.load(run / "config.yaml")
    cfg.streams = ens.streams
    # The bundle's `structures_in` described where the TRAINING structures were,
    # on whoever's machine trained it -- a path that need not exist here, and that
    # holds none of these molecules anyway. Only --pdb-dir (or a `pdb` column)
    # supplies structures for a prediction table.
    cfg.paths.structures_in = getattr(args, "pdb_dir", None)
    if getattr(args, "check_pdb", None) is False:
        cfg.check_pdb_sequence = False

    # carry the target column through if the file has it -- it is ground truth to
    # validate against, never model input
    df_in = load_predict(args.table, truth=[ens.target],
                         truth_column=getattr(args, "truth_column", None),
                         min_vl=cfg.split.min_len_vl)
    has_truth = ens.target in df_in.columns
    if not has_truth:
        # Easy to do by accident -- a measured table scored as a blind one looks
        # like a success, just without the validation section.
        near = [c for c in df_in.columns
                if c not in ("Ab_ID", "VH", "VL", "key", "source", "pdb")]
        print(f"[predict] no {ens.target} column: predicting without validating."
              + (f" Other columns present: {near}. Use --truth-column NAME if one "
                 "of them holds the measurements." if near else ""))

    df, feat_path, emb_path = _prepare_inputs(cfg, df_in, overwrite=args.overwrite)
    # The bundle's frame, not a fresh one: column j must mean what it meant in training.
    arrays, meta, df = load_arrays(df, cfg, feat_path, emb_path,
                                   frame=ens.imgt_positions)
    ens.check_features(meta, arrays)

    scored = ens.predict_table(df, arrays)        # only the featurisable rows

    # EVERY input row comes back, in the order supplied. A molecule whose
    # structure could not be built is a NaN prediction with a reason, not a
    # missing line: the caller asked about these antibodies, a shorter file
    # silently breaks any positional join, and "absent" is indistinguishable
    # from "lost" once the console output has scrolled away.
    #
    # The measurements and the source come from the INPUT table, not from the
    # scored subset -- a row that could not be featurised still has whatever the
    # caller measured, and taking it from the merge would blank it out and then
    # hide the row from the "measured but unscored" count below.
    # Ab_ID is unique here (`report_duplicates` suffixes repeats inside
    # `load_predict`), so this merge is one-to-one.
    carry = ["Ab_ID", "VH", "VL"] + ([ens.target] if has_truth else []) \
            + (["source"] if "source" in df_in.columns else [])
    out = df_in[carry].copy()
    # `pdb_source` stays: it says whether the number rests on a structure the
    # caller supplied or on a predicted one, which is not recoverable afterwards.
    # The path itself goes -- it is long, machine-specific, and adds nothing.
    scored = scored.drop(columns=[c for c in ("VH", "VL", ens.target, "source", "pdb")
                                  if c in scored.columns])
    out = out.merge(scored, on="Ab_ID", how="left")

    pred_col = f"{ens.target}_pred"
    ok = out[pred_col].notna() if pred_col in out.columns else pd.Series(False, index=out.index)
    out["status"] = np.where(ok, "predicted", "no_structure_or_embedding")

    n_bad = int((~ok).sum())
    if n_bad:
        print(f"[predict] {n_bad} of {len(out)} row(s) could not be featurised; "
              "they are present with an empty prediction and status "
              "'no_structure_or_embedding'")
    out.to_csv(args.out, index=False)
    print(f"[predict] {len(out)} rows ({int(ok.sum())} predicted) -> {args.out}")

    # ---- validation, only when the caller supplied measurements ------------
    if not has_truth:
        print(f"[predict] no '{ens.target}' column in the table -- predictions only. "
              "Add one to get validation plots and scores.")
        return

    # Validation runs on rows that have BOTH a measurement and a prediction. The
    # output file keeps the unpredictable ones (see above), but scoring them
    # against nothing would silently shrink n or, worse, leak NaNs into the
    # figures.
    t_all = out[ens.target].to_numpy(float)
    # The UNCALIBRATED prediction, when there is one, for the same reason the CV
    # figures use it: quantile calibration maps predicted quantiles onto the
    # training values, and on a zero-inflated assay (AC-SINS, SMP -- most
    # antibodies simply do not self-associate) the bottom of that map is flat at
    # zero. Every prediction in that range lands on exactly zero, and those ties
    # cost Spearman. The map is monotone but not STRICTLY monotone, which is the
    # gap between "ranking is preserved" and "rank metrics are unchanged".
    #
    # Calibration is for reading values in assay units -- thresholds, flags, the
    # calibration figure. It is not for ranking, and scoring it here also made
    # this path disagree with the CV path, so the two numbers were not comparable.
    raw_col = f"{ens.target}_raw"
    score_col = raw_col if raw_col in out.columns else f"{ens.target}_pred"
    if score_col == raw_col:
        print(f"[predict] rank metrics on the uncalibrated prediction "
              f"({raw_col}); {ens.target}_pred keeps the calibrated values")
    p_all = out[score_col].to_numpy(float)
    both = np.isfinite(t_all) & np.isfinite(p_all)
    t, p = t_all[both], p_all[both]
    n_lab = int(both.sum())
    n_unscored = int((np.isfinite(t_all) & ~np.isfinite(p_all)).sum())
    if n_unscored:
        print(f"[predict] {n_unscored} measured molecule(s) have no prediction and "
              "are excluded from the scores and figures")
    if n_lab < 5:
        print(f"[predict] only {n_lab} molecule(s) with both a measurement and a "
              "prediction -- too few to validate")
        return

    opt = dict(ens.b["target_opts"])
    # The bundle carries the settings its run was REPORTED with, so a model is
    # scored the way its own numbers were produced. The AUC threshold is the one
    # that should not need a retrain to change: it is a reporting choice about
    # where pass/fail sits, not a property of the weights.
    if getattr(args, "threshold", None) is not None:
        print(f"[predict] AUC threshold {opt['threshold']} -> {args.threshold} "
              "(override; rho and pearson are unaffected)")
        opt["threshold"] = args.threshold
    m = score(t, p, threshold=opt["threshold"], higher_is_bad=opt["higher_is_bad"],
              noise_floor=opt["noise_floor"],
              censor_side=opt.get("censor_side", "floor"))
    lab = (t > opt["threshold"]) if opt["higher_is_bad"] else (t < opt["threshold"])
    if lab.all() or not lab.any():
        print(f"[predict] AUC is n/a: all {len(t)} measured molecules are on one "
              f"side of threshold {opt['threshold']} (measured range "
              f"{t.min():.3g} to {t.max():.3g}). Use --threshold to change it.")
    cens = bool(opt.get("report_censored", False)) and opt["noise_floor"] is not None
    rho = m["rho_c"] if cens else m["rho"]
    pear = m["pearson_c"] if cens else m["pearson"]
    print(f"\n=== validation on {n_lab} measured molecules ===")
    print(f"  Spearman {rho:.3f}   Pearson {pear:.3f}   AUC {_fmt(m['auc'])}"
          + ("   (rank metrics censored at the noise floor)" if cens
             else "   (original units, uncensored)"))
    if "source" in out.columns:
        print("\nper source:")
        aucs = [m["auc"]]
        for s, g in out[both].groupby("source"):
            gt = g[ens.target].to_numpy(float); gp = g[score_col].to_numpy(float)
            if int(np.isfinite(gt).sum()) < 5:
                continue
            gm = score(gt, gp, threshold=opt["threshold"],
                       higher_is_bad=opt["higher_is_bad"], noise_floor=opt["noise_floor"],
                       censor_side=opt.get("censor_side", "floor"))
            grho = gm["rho_c"] if cens else gm["rho"]
            gpear = gm["pearson_c"] if cens else gm["pearson"]
            aucs.append(gm["auc"])
            print(f"  {s:<28} n={gm['n']:<5} Spearman {grho:.3f}   "
                  f"Pearson {gpear:.3f}   AUC {_fmt(gm['auc'])}")
        # An AUC that is NaN everywhere is not a weak result, it is no result --
        # every molecule fell on one side of the threshold. Said once, with the
        # number to change, rather than printed as `nan` on every row.
        if aucs and not np.isfinite(aucs).any():
            print(f"\n  AUC is n/a: every molecule is on one side of "
                  f"target_opts.threshold = {opt['threshold']} (original units), so "
                  "there are no two classes to separate. Spearman and Pearson are "
                  "unaffected; set a threshold that splits your panel to get an AUC.")
        print("  (pooled numbers across sources partly measure the OFFSET between "
              "source scales, not ranking within either -- read the per-source rows)")

    stem = Path(args.out).with_suffix("")
    # Raw seed columns when the bundle calibrates, so the cloud shares an axis
    # with the ensemble points the figures score. `_seed*` without the suffix
    # stays calibrated, for anyone reading spread in assay units.
    _seed = [c for c in out.columns if c.startswith(f"{ens.target}_seed")]
    seed_cols = [c for c in _seed if c.endswith("_raw")] or _seed
    # every array the figure sees is subset by `both`, so the rows line up
    validation_figure(
        t, p, f"{stem}_validation.png",
        title=f"{ens.target} - {Path(args.run).name}",
        threshold=opt["threshold"], higher_is_bad=opt["higher_is_bad"],
        noise_floor=opt["noise_floor"], censor_metrics=cens,
        sources=out["source"].to_numpy()[both] if "source" in out.columns else None,
        seed_preds=out[seed_cols].to_numpy()[both].T if seed_cols else None)
    # the calibrated column here, deliberately: this figure exists to show
    # whether the values land on the training scale, which is what calibration
    # is for. Scoring uses the raw column above; plotting spread uses this one.
    calibration_figure(t, out[f"{ens.target}_pred"].to_numpy(float)[both],
                       f"{stem}_calibration.png",
                       title=f"{ens.target} - calibration",
                       noise_floor=opt["noise_floor"])
    # One bar per seed model next to the ensemble. The released number is the
    # ensemble; this says how much of it survives a different initialisation,
    # which a scatter of the ensemble alone cannot show.
    if seed_cols:
        seed_figure(
            t, out[seed_cols].to_numpy()[both].T, p, f"{stem}_seeds.png",
            title=f"{ens.target} - per-seed vs ensemble ({Path(args.run).name})",
            threshold=opt["threshold"], higher_is_bad=opt["higher_is_bad"],
            noise_floor=opt["noise_floor"], censor_metrics=cens,
            labels=[c.rsplit("seed", 1)[-1] for c in seed_cols])


# --------------------------------------------------------------------------- #
def cmd_models(args):
    """List the released models and what each one's number actually means.

    A released model is just a run directory (bundle.pt + config.yaml), so this
    command exists only to make the manifest readable and to flag entries whose
    numbers were never filled in -- an unfinished release should be visible, not
    assumed to be fine.
    """
    import yaml

    root = Path(args.dir) if args.dir else Path(__file__).resolve().parents[1] / "models"
    man = root / "MODELS.yaml"
    if not man.exists():
        raise FileNotFoundError(
            f"{man} not found. Pass --dir if the models folder is elsewhere.")
    doc = yaml.safe_load(man.read_text(encoding="utf-8")) or {}
    entries = doc.get("models") or []
    if not entries:
        print(f"[models] {man} lists no models")
        return

    print(f"{'id':<20} {'target':<11} {'model':<7} {'status':<9} "
          f"{'rho':>6} {'AUC':>6}  regime / exposure")
    def _num(v):                       # numbers to 3 dp, anything missing to --
        return f"{v:.3f}" if isinstance(v, (int, float)) else "--"

    incomplete = []
    for e in entries:
        mid = e.get("id", "?")
        cv = e.get("cv") or {}
        ex = e.get("exposure") or {}
        released = e.get("status") == "released"
        avail = (root / (e.get("dir") or mid) / "bundle.pt").exists()
        status = e.get("status", "?") + ("" if avail or not released else " (!)")
        exposure = (f"{e.get('regime', '?')}  "
                    f"T{ex.get('transductive', '?')}/A{ex.get('analogue', '?')}"
                    f"/S{ex.get('strict', '?')}%" if ex else e.get("regime", ""))
        print(f"{mid:<20} {e.get('target', ''):<11} {e.get('model', ''):<7} "
              f"{status:<9} {_num(cv.get('spearman')):>6} "
              f"{_num(cv.get('auc')):>6}  {exposure}")
        if released and (not cv or not ex):
            incomplete.append(mid)
        if released and not avail:
            incomplete.append(f"{mid} (no bundle.pt)")

    n_rel = sum(1 for e in entries if e.get("status") == "released")
    print(f"\n{n_rel}/{len(entries)} released.  "
          f"Use one with:  madm predict --run {root.name}/<id> --table <file> --out <file>")
    if incomplete:
        print(f"[models] incomplete entries: {', '.join(sorted(set(incomplete)))}")
    print("[models] `regime` is the strongest route by which auxiliary information "
          "reached the validation molecules (T/A/S = transductive/analogue/strict "
          "per cent). A transductive number describes assay imputation, not "
          "prediction for an antibody with no other measurements.")


# --------------------------------------------------------------------------- #
# What each part of the package needs, and where it comes from. `None` means the
# module is pip-installable under its own name via the matching extra.
#   (module, what it is for, how to install it)
DEPS = [
    ("core", [
        ("numpy", "arrays", None), ("pandas", "tables", None),
        ("scipy", "statistics", None), ("sklearn", "splits, metrics", None),
        ("yaml", "config", None), ("openpyxl", ".xlsx tables", None),
    ]),
    ("training and inference", [
        ("torch", "everything except `madm models`", "see pytorch.org for your CUDA version"),
    ]),
    ("feat stream (3D structures)", [
        ("ImmuneBuilder", "structure prediction", "pip install -e '.[structure]'"),
        ("openmm", "ImmuneBuilder's refinement", "conda install -c conda-forge openmm pdbfixer"),
        ("pdbfixer", "ImmuneBuilder's refinement", "conda install -c conda-forge openmm pdbfixer"),
        ("anarci", "IMGT numbering", "conda install -c bioconda anarci hmmer"),
        ("biotite", "PDB parsing", "pip install -e '.[structure]'"),
    ]),
    ("ablang stream (embeddings)", [
        ("ablang2", "AbLang2 embeddings", "pip install -e '.[ablang]'"),
    ]),
    ("figures", [
        ("matplotlib", "validation plots", "pip install -e '.[plots]'"),
    ]),
]

SUPPORTED_PY = (3, 10), (3, 12)


def _check_python() -> None:
    """Say so at the start if the interpreter is outside the tested range.

    `conda create -n madm` without `python=3.11` gives you whatever is newest,
    and the consequences arrive much later and disguised: a wheel that resolves
    to "no matching distribution", or an install that succeeds and then dies on
    the first structure. Naming it here costs one line of output and saves the
    hour spent reading the wrong traceback.

    A warning, not an error -- new versions do eventually work, and refusing to
    run on one that does would be its own obstruction.
    """
    import sys
    v = sys.version_info[:2]
    lo, hi = SUPPORTED_PY
    if lo <= v <= hi:
        return
    where = "older than" if v < lo else "newer than"
    print(f"[madm] WARNING Python {v[0]}.{v[1]} is {where} the tested range "
          f"{lo[0]}.{lo[1]}-{hi[0]}.{hi[1]}.")
    if v > hi:
        print("[madm]   torch, ablang2, openmm and anarci publish builds for a new "
              "interpreter months late, so some of them may be missing or absent "
              "from this environment even though `pip install` reported success.")
    print(f"[madm]   conda create -n madm python={hi[0]}.{hi[1]} -y && conda activate madm")


def _reachable(url: str, timeout: float = 6.0):
    """HTTP status for `url`, or a short string saying why there is none.

    Uses urllib rather than requests so `doctor` keeps working in an environment
    that is missing things -- which is the environment it exists for. Short
    timeout: this is a courtesy check, not a reason to sit for 30 seconds.
    """
    import urllib.error
    import urllib.request
    try:
        req = urllib.request.Request(url, method="GET",
                                     headers={"User-Agent": "madm-doctor"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code                       # 504 etc -- the server answered
    except Exception as e:                  # noqa: BLE001 - DNS, TLS, proxy, timeout
        return f"no response ({type(e).__name__})"


def cmd_doctor(args):
    """Report what is missing BEFORE a long run finds out the hard way.

    Written for a fresh machine. Structure prediction and embedding both sit
    hours into `madm train`, so an environment that is 90% installed fails late
    and expensively -- and with a traceback from inside somebody else's package,
    which names the symptom rather than the fix.

    Reports per stream, because "incomplete" is not the same as "broken": an
    ablang-only run needs none of the structure side, so a red line there is only
    a problem if you intend to use the feature stream. Exits non-zero when
    something the GIVEN CONFIG actually needs is absent, so this is usable in a
    provisioning script.
    """
    import importlib
    import sys

    print(f"python       {sys.version.split()[0]}   ({sys.executable})")
    lo, hi = SUPPORTED_PY
    ok_py = lo <= sys.version_info[:2] <= hi
    print(f"             {'ok' if ok_py else 'OUTSIDE'} tested range "
          f"{lo[0]}.{lo[1]}-{hi[0]}.{hi[1]}")
    print(f"madm         {Path(__file__).resolve().parents[1]}")

    want_feat = want_ab = True
    cfg = None
    if args.config:
        try:
            cfg = _apply_overrides(Config.load(args.config), args)
            want = needs(cfg)
            want_feat, want_ab = want["feat"], want["ablang"]
            print(f"config       {args.config}   target={cfg.target}  "
                  f"streams={cfg.streams}  aux={cfg.aux or 'none'}")
        except Exception as e:                                  # noqa: BLE001
            print(f"config       FAILED TO LOAD: {e}")

    missing_needed, missing_other = [], []
    for group, mods in DEPS:
        needed = (group != "feat stream (3D structures)" or want_feat) and \
                 (group != "ablang stream (embeddings)" or want_ab)
        print(f"\n{group}{'' if needed else '   (not needed for these streams)'}")
        for mod, why, fix in mods:
            try:
                m = importlib.import_module(mod)
                ver = getattr(m, "__version__", "")
                print(f"  [ok]      {mod:<15} {ver:<12} {why}")
            except Exception as e:                              # noqa: BLE001
                print(f"  [MISSING] {mod:<15} {'':<12} {why}")
                print(f"            -> {fix or 'pip install -e .'}")
                if str(e) and mod not in str(e):
                    print(f"            ({e})")                 # a broken dep, not an absent one
                (missing_needed if needed else missing_other).append(mod)

    # Importing ImmuneBuilder is not the same as being able to USE it: the
    # weights are fetched from Zenodo when ABodyBuilder2 is first constructed,
    # which is a network operation inside what otherwise looks like an offline
    # step. Reported, never downloaded here -- `doctor` should be quick and safe
    # to run anywhere, and this is hundreds of MB.
    # Vendored, so this is really a check that the install is complete and that
    # the extractor's own third-party imports resolve -- biotite, hydride, anarci,
    # mdtraj and propka all fail here rather than at `import madm`.
    if want_feat:
        from .prepare import _feature_extractor
        try:
            _feature_extractor()
            print("\n  [ok]      madm.features3d   3D feature extractor")
        except ImportError as e:
            print("\n  [MISSING] madm.features3d   3D feature extractor")
            for line in str(e).splitlines():
                print(f"            {line}")
            missing_needed.append("madm.features3d")

    if want_feat and "ImmuneBuilder" not in missing_needed:
        from .prepare import weights_files
        have = weights_files()
        if have:
            print(f"\n  [ok]      ABodyBuilder2 weights  {len(have)} file(s) in "
                  f"{have[0].parent}")
        else:
            print("\n  [note]    ABodyBuilder2 weights  not cached yet")
            print("            The first structure downloads them from zenodo.org "
                  "(~hundreds of MB).")
            # Only probed when the weights are ABSENT, so a fully provisioned
            # machine never makes a network call to run `doctor`. Zenodo goes down
            # for hours at a time; finding that out here costs two seconds, and
            # finding it out from build_structures costs the table load, the
            # screening and whatever else ran first.
            code = _reachable("https://zenodo.org/api/records/7258553")
            if code == 200:
                print("            zenodo.org is reachable. Fetch them now, so a "
                      "long run cannot die on an outage:")
            else:
                print(f"  [WARNING] zenodo.org answered {code} -- structure "
                      "prediction WILL fail until this clears.")
                print("            Zenodo has multi-hour outages; this is not "
                      "something your environment can fix.")
                print("            Use --pdb-dir with structures you already have, "
                      "or --model ablang, and fetch the weights later:")
            print("              python -c \"from ImmuneBuilder import "
                  "ABodyBuilder2; ABodyBuilder2()\"")

    if cfg is not None:
        print("\ninputs")
        q = Path(cfg.paths.table)
        print(f"  [{'ok' if q.exists() else 'MISSING':<7}] training table    {q.resolve()}")
        if want_feat:
            ref = cfg.reference_tables
            where = "packaged" if not cfg.paths.reference_tables else "configured"
            print(f"  [{'ok' if ref.is_dir() else 'MISSING':<7}] reference tables  "
                  f"{ref}  ({where})")
            for f in ("Hidrophobisicy_score.xlsx", "Residue_max_SASA.xlsx"):
                ok = (ref / f).exists()
                print(f"  [{'ok' if ok else 'MISSING':<7}] {f}")
                if not ok:
                    missing_needed.append(f)
        if cfg.paths.structures_in:
            d = Path(cfg.paths.structures_in)
            n = len(list(d.glob("*.pdb"))) if d.is_dir() else 0
            print(f"  [{'ok' if d.is_dir() else 'MISSING':<7}] supplied structures  "
                  f"{d.resolve()}  ({n} .pdb)")

    print()
    if missing_needed:
        print(f"[doctor] {len(missing_needed)} missing: {', '.join(missing_needed)}")
        if not want_feat:
            print("[doctor] (the structure side was not checked -- these streams "
                  "do not use it)")
        raise SystemExit(1)
    print("[doctor] everything this configuration needs is present."
          + ("" if want_feat else "  Structure prediction was not checked.")
          + ("" if ok_py else "  Python is outside the tested range; see above."))


def main(argv=None):
    _check_python()
    ap = argparse.ArgumentParser(prog="madm", description="MADM -- multi-assay antibody developability models")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _log_flag(p):
        p.add_argument("--log", metavar="PATH",
                       help="also write everything printed (and any traceback) "
                            "to this file. `train` already logs to "
                            "<run>/run.log and `predict` to <out>.log; this "
                            "overrides those.")

    def _model_flags(p, with_target=True):
        p.add_argument("--model", choices=sorted(MODELS),
                       help="feat = structure features only | ablang = embeddings only | "
                            "2s = both streams, single task | mt = both streams + auxiliaries. "
                            "Overrides `streams` in the config; anything but `mt` also clears `aux`.")
        p.add_argument("--aux", help="comma-separated auxiliary assays, e.g. "
                                     "'Ovalbumin,Insulin'. Overrides `aux` in the config; "
                                     "pass '' for none.")
        if with_target:
            p.add_argument("--target", help="override `target` in the config")
            p.add_argument("--table", help="override `paths.table` in the config")

    def _pdb_flags(p):
        p.add_argument("--pdb-dir", dest="pdb_dir",
                       help="directory of YOUR .pdb structures, named <Ab_ID>.pdb "
                            "or <sha1(VH|VL)[:16]>.pdb. Used where present; only "
                            "the rest are predicted. Overrides paths.structures_in.")
        p.add_argument("--no-check-pdb-sequence", dest="check_pdb", action="store_false",
                       default=None,
                       help="skip the check that a supplied .pdb is the molecule "
                            "the table names it as. Features come from the "
                            "structure, so this is how the wrong geometry gets "
                            "attached to a label without anything complaining.")

    p = sub.add_parser("doctor", help="check the environment and inputs before a long run")
    p.add_argument("--config", help="also check this config's table, reference "
                                    "tables and streams")
    _model_flags(p); _pdb_flags(p); _log_flag(p)
    p.set_defaults(func=cmd_doctor, _table_is_train=True)

    p = sub.add_parser("prepare", help="structures, features, embeddings (cached)")
    p.add_argument("--config", required=True)
    p.add_argument("--overwrite", action="store_true")
    _model_flags(p); _pdb_flags(p); _log_flag(p)
    p.set_defaults(func=cmd_prepare, _table_is_train=True)

    p = sub.add_parser("train", help="prepare if needed, cross-validate, fit, save a bundle")
    p.add_argument("--config", required=True)
    p.add_argument("--overwrite", action="store_true")
    _model_flags(p); _pdb_flags(p); _log_flag(p)
    p.set_defaults(func=cmd_train, _table_is_train=True)

    p = sub.add_parser("models", help="list the released models and their regimes")
    p.add_argument("--dir", help="models folder (default: the one shipped with MADM)")
    _log_flag(p)
    p.set_defaults(func=cmd_models, _table_is_train=False, model=None, aux=None,
                   target=None)

    p = sub.add_parser("predict", help="score a VH/VL table; validates if it has measurements")
    p.add_argument("--run", required=True,
                   help="a training run directory, or a released model: "
                        "models/<id> (see `madm models`)")
    p.add_argument("--table", required=True)
    p.add_argument("--out", default="predictions.csv")
    p.add_argument("--threshold", type=float,
                   help="AUC cutoff in ORIGINAL units, overriding the bundle's. "
                        "A reporting choice, not a model property -- Spearman and "
                        "Pearson are unaffected.")
    p.add_argument("--truth-column", dest="truth_column", metavar="NAME",
                   help="the column holding the MEASURED values, when it is not "
                        "named after the model's target -- e.g. --truth-column PSR "
                        "for a model whose target is SMP. Treated as ground truth, "
                        "never as input.")
    p.add_argument("--overwrite", action="store_true")
    _pdb_flags(p); _log_flag(p)
    p.set_defaults(func=cmd_predict, _table_is_train=False, model=None, aux=None, target=None)

    args = ap.parse_args(argv)

    # Wrap the streams before anything runs, so the log starts at the first line
    # rather than wherever the destination happened to become known. stderr goes
    # to the same file: a traceback belongs next to the output that led to it,
    # and the two interleave as they happened.
    import sys
    sys.stdout, sys.stderr = _Tee(sys.stdout), _Tee(sys.stderr)
    if getattr(args, "log", None):
        log_to(args.log)
    try:
        args.func(args)
    finally:
        # stdout and stderr share ONE handle, so flush both before closing it
        # once -- flushing a stream whose file is already closed raises.
        tees = [s for s in (sys.stdout, sys.stderr) if isinstance(s, _Tee)]
        for s in tees:
            s.flush()
        for fh in {id(s.fh): s.fh for s in tees if s.fh is not None}.values():
            fh.close()
        for s in tees:
            s.fh = None
        # the underlying console streams are left alone; the process is ending


if __name__ == "__main__":
    main()
