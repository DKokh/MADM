"""Inference from a VH/VL table alone.

A bundle carries everything inference needs and nothing it does not: weights, the
architecture arguments, the feature standardisation, the target transform, the
list of streams, and the PROVENANCE of the features -- the `features:` settings
they were built with, their column names, and the IMGT frame they were aligned to.

That provenance is not documentation. A channel-count mismatch shows up as a conv
shape error, but a channel-ORDER mismatch shows up as nothing at all: the model
happily consumes the wrong columns and returns confident numbers. The bundle
checks both before predicting.

Seeds and folds are averaged in Z-SPACE and inverted once. Averaging in original
units after an expm1 would weight the members by how large their predictions are.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from .data import Calibrator, TargetTransform, apply_feature_stats, make_loader
from .model import FvModel

FORMAT = "madm-bundle-2"


def save_bundle(path, models: List[FvModel], cfg, tfs, feat_mean, feat_sd, arch: dict,
                seeds=None, epochs: int = None, calibrator: Calibrator = None,
                meta: Optional[dict] = None, ab_mean=None, ab_sd=None) -> Path:
    path = Path(path)
    meta = meta or {}
    from dataclasses import asdict
    torch.save({
        "format": FORMAT,
        "target": cfg.target,
        "tasks": cfg.tasks,
        "streams": list(cfg.streams),
        "arch": arch,
        "seeds": list(seeds) if seeds is not None else list(cfg.train.seeds),
        "epochs": epochs,
        "state_dicts": [m.state_dict() for m in models],
        "target_transform": tfs[cfg.target].to_dict(),
        "calibrator": (calibrator or Calibrator()).to_dict(),
        "feat_mean": feat_mean,
        "feat_sd": feat_sd,
        # the embedding stream's statistics, for the same reason: a mid-stack ESM
        # tensor is unnormalised and must reach the model on the scale it trained on
        "ab_mean": ab_mean,
        "ab_sd": ab_sd,
        "target_opts": {"threshold": cfg.target_opts.threshold,
                        "higher_is_bad": cfg.target_opts.higher_is_bad,
                        "noise_floor": cfg.target_opts.noise_floor,
                        "censor_side": cfg.target_opts.censor_side,
                        # reporting convention travels with the model, so a
                        # prediction is scored the way its training run was
                        "report_censored": cfg.target_opts.report_censored},
        # ---- provenance: what the feature tensor has to look like -----------
        "frame_kind": cfg.model.frame,
        "imgt_positions": meta.get("frame"),
        "feature_names": meta.get("feature_names"),
        "global_feature_names": meta.get("global_feature_names"),
        "feature_params": meta.get("feature_params") or asdict(cfg.features),
        # Which language model produced X_ab. Shape cannot say: AbLang2 and
        # ESM2-35M are both 480-d and mean different things.
        "embedder": meta.get("embedder") or cfg.model.embedder,
        "embedder_layer": meta.get("embedder_layer", cfg.model.embedder_layer),
        "config_hash": cfg.hash(),
    }, path)
    print(f"[bundle] {len(models)} model(s) -> {path}")
    return path


class Ensemble:
    """Loaded bundle. `streams` tells you what inputs it needs."""

    def __init__(self, path, device: str = "cpu"):
        b = torch.load(path, map_location="cpu", weights_only=False)
        if b.get("format") != FORMAT:
            raise ValueError(
                f"{path}: not a {FORMAT} bundle (got {b.get('format')!r}). Bundles "
                "written before the provenance fields cannot be checked against a "
                "feature cache; retrain, or load them with the version that wrote them.")
        self.b = b
        self.device = device
        self.streams = b["streams"]
        self.target = b["target"]
        self.tasks = b["tasks"]
        self.frame_kind = b.get("frame_kind", "raw")
        self.imgt_positions = b.get("imgt_positions")
        self.embedder = b.get("embedder", "ablang2")
        self.embedder_layer = b.get("embedder_layer", -1)
        self.feature_names = b.get("feature_names")
        self.feature_params = b.get("feature_params")
        self.tf = TargetTransform.from_dict(b["target_transform"])
        self.cal = Calibrator.from_dict(b.get("calibrator") or {"kind": "none"})
        self.seeds = b.get("seeds") or list(range(len(b["state_dicts"])))
        if len(self.seeds) != len(b["state_dicts"]):
            raise ValueError(f"{path}: {len(b['state_dicts'])} model(s) but "
                             f"{len(self.seeds)} seed label(s)")
        self.models = []
        for sd in b["state_dicts"]:
            m = FvModel(tasks=b["tasks"], streams=b["streams"], **b["arch"])
            m.load_state_dict(sd)
            m.eval().to(device)
            self.models.append(m)

    def __repr__(self):
        return (f"Ensemble(target={self.target}, streams={'+'.join(self.streams)}, "
                f"n_models={len(self.models)}, frame={self.frame_kind}, "
                f"calibrate={self.cal.kind}, cfg={self.b['config_hash']})")

    def needs(self) -> dict:
        return {"feat": "feat" in self.streams, "ablang": "ablang" in self.streams}

    # ---- provenance -------------------------------------------------------
    def check_features(self, meta: dict, arrays: dict) -> None:
        """Refuse arrays that are not what this model was trained on."""
        if "ablang" in self.streams:
            got_l = meta.get("embedder_layer", -1)
            if got_l != self.embedder_layer:
                raise ValueError(
                    f"this model was trained on layer {self.embedder_layer} but "
                    f"the cache holds layer {got_l}.")
            got = meta.get("embedder", "ablang2")
            if got != self.embedder:
                raise ValueError(
                    f"this model was trained on {self.embedder!r} embeddings but "
                    f"the cache holds {got!r}. Set model.embedder in the run's "
                    "config.yaml to match, or rebuild the cache.")
            want_d = self.b["arch"].get("ablang_dim")
            got_d = arrays["X_ab"].shape[-1]
            if want_d and got_d != want_d:
                raise ValueError(f"model expects {want_d}-d embeddings, cache has {got_d}")
        if "feat" not in self.streams:
            return
        want_f = self.b["arch"]["n_feat"]
        got_f = arrays["X"].shape[-1]
        if got_f != want_f:
            raise ValueError(f"model expects {want_f} feature channels, cache has {got_f}")

        names = meta.get("feature_names")
        if self.feature_names and names and list(names) != list(self.feature_names):
            first = next((i for i, (a, b) in enumerate(zip(names, self.feature_names))
                          if a != b), 0)
            raise ValueError(
                "feature COLUMN ORDER differs from training -- the model would read "
                f"the wrong channels. First difference at column {first}: "
                f"cache={names[first]!r} bundle={self.feature_names[first]!r}")

        got_p = meta.get("feature_params")
        if self.feature_params and got_p:
            diff = {k: (v, got_p.get(k)) for k, v in self.feature_params.items()
                    if got_p.get(k) != v}
            if diff:
                lines = "\n  ".join(f"{k}: bundle={a!r} cache={b!r}"
                                    for k, (a, b) in diff.items())
                raise ValueError(f"features were built with different settings:\n  {lines}")

        if self.frame_kind == "imgt":
            if not self.imgt_positions:
                raise ValueError("bundle claims frame 'imgt' but carries no imgt_positions")
            if arrays["X"].shape[1] != len(self.imgt_positions):
                raise ValueError(
                    f"frame length {arrays['X'].shape[1]} != bundle frame "
                    f"{len(self.imgt_positions)} -- align with the bundle's frame")

    # ---- inference --------------------------------------------------------
    @torch.no_grad()
    def predict_per_seed(self, arrays: dict, batch_size: int = 32) -> np.ndarray:
        """[n_models, n_molecules] in Z-SPACE, one row per seed."""
        arrays = dict(arrays)
        if "feat" in self.streams and self.b["feat_mean"] is not None:
            arrays["X"] = apply_feature_stats(arrays["X"], arrays["mask"],
                                              self.b["feat_mean"], self.b["feat_sd"])
        if "ablang" in self.streams and self.b.get("ab_mean") is not None:
            arrays["X_ab"] = apply_feature_stats(arrays["X_ab"], arrays["mask_ab"],
                                                 self.b["ab_mean"], self.b["ab_sd"])
        n = len(next(iter(arrays.values())))
        loader = make_loader(np.arange(n), batch_size, False, **arrays)

        acc = []
        for m in self.models:
            out = []
            for x, mk, xa, ma, g, _, _ in loader:
                x, mk, xa, ma, g = [t.to(self.device) for t in (x, mk, xa, ma, g)]
                out.append(m(x, xa, self.target, mask=mk, mask_ab=ma,
                             x_global=g).cpu().numpy())
            acc.append(np.concatenate(out))
        return np.stack(acc)

    def predict(self, arrays: dict, batch_size: int = 32,
                calibrated: bool = True) -> np.ndarray:
        """Ensemble prediction on the scale of the original training values.

        Averaged in Z-SPACE and inverted once -- averaging after expm1 would weight
        the seeds by how large their predictions are, which is not an average --
        then rescaled by the calibrator so the spread matches the training values.
        """
        y = self.tf.inverse(self.predict_per_seed(arrays, batch_size).mean(axis=0))
        return self.cal.apply(y) if calibrated else y

    def predict_table(self, df: pd.DataFrame, arrays: dict,
                      per_seed: bool = True) -> pd.DataFrame:
        """Output table, ONE ROW PER INPUT ROW IN THE INPUT ORDER.

        A `<target>_rank` column carries the ordering (1 = worst by the assay's own
        convention) so nothing is lost by not sorting, and the caller can still
        join the result back onto their table by position.

        `*_sd` is disagreement between seeds, not a confidence interval. It says
        how much of the prediction is initialisation noise -- useful for spotting
        molecules the model is unsure about, but it does not cover systematic
        error, and a confident ensemble can still be confidently wrong.
        """
        Z = self.predict_per_seed(arrays)                 # [n_models, n]
        raw = self.tf.inverse(Z.mean(axis=0))
        y = self.cal.apply(raw)
        per = np.stack([self.cal.apply(self.tf.inverse(z)) for z in Z])

        out = df[["Ab_ID", "VH", "VL"]].copy()
        t = self.target
        hib = self.b["target_opts"]["higher_is_bad"]
        out[f"{t}_pred"] = y
        out[f"{t}_sd"] = per.std(axis=0)
        out[f"{t}_min"] = per.min(axis=0)
        out[f"{t}_max"] = per.max(axis=0)
        if self.cal.kind != "none":
            out[f"{t}_raw"] = raw       # before rescaling, for reference
        out[f"{t}_rank"] = pd.Series(y).rank(ascending=not hib,
                                             method="min").astype(int).to_numpy()

        thr = self.b["target_opts"]["threshold"]
        if thr:
            flag = y > thr if hib else y < thr
            out[f"{t}_flag"] = np.where(flag, "risk", "ok")
            # fraction of seeds that agree with the ensemble's call
            votes = (per > thr) if hib else (per < thr)
            out[f"{t}_agree"] = np.where(flag, votes.mean(0), 1 - votes.mean(0))

        if per_seed:
            for s, col in zip(self.seeds, per):
                out[f"{t}_seed{s}"] = col
            # ... and the same seeds BEFORE calibration. The figures score and
            # plot the uncalibrated ensemble, so a per-seed cloud drawn from the
            # calibrated columns would sit on a different axis from the points
            # it is meant to surround -- on a zero-inflated assay, a grey stripe
            # at zero behind ensemble points that are spread out.
            if self.cal.kind != "none":
                raw_per = np.stack([self.tf.inverse(z) for z in Z])
                for s, col in zip(self.seeds, raw_per):
                    out[f"{t}_seed{s}_raw"] = col
        return out.reset_index(drop=True)
