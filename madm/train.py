"""One training loop, for every stream and task combination.

A single-task model is the multi-task loss with one task; a single-stream model is
the two-stream model without the other stream. So there is one `train_model`, one
loss, one early-stopping rule, and one checkpoint format. Keeping separate loops
for the single-stream baselines is what makes their numbers quietly incomparable
with the fused ones -- different stopping criteria, different metric conventions.

EARLY STOPPING: every model stops once, on its own validation fold, by
`train.select_on` -- validation loss by default, as the notebooks use
(`select_on="mse"`). Warm-start models included. Same rule everywhere, so the
comparison is fair even though the models differ.

Spearman is available but is a poor stopping rule on its own: it is scale-
invariant, so a model that shrinks its predictions toward the mean pays no penalty
for doing so.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import Config
from .data import (apply_feature_stats, apply_transforms, fit_feature_stats,
                   fit_transforms, make_loader)
from .metrics import score
from .model import AuxUncertaintyWeighter, FvModel


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# loss
# --------------------------------------------------------------------------- #
def pointwise(kind: str):
    """The per-element loss on uncensored points.

    `mse` is what the notebooks train with: they call `train_simple` with
    `nonzero_weight=1`, which collapses `Scripts.Train_loader.weighted_mse_loss`
    to plain MSE. `smooth_l1` is available but is not what any published number
    used, and it changes how much the largest residuals matter.
    """
    if kind == "mse":
        return F.mse_loss
    if kind == "smooth_l1":
        return F.smooth_l1_loss
    raise ValueError(f"train.loss must be mse or smooth_l1, got {kind!r}")


def _tobit(pred, y, floor_z: float, sigma: float, obs_loss=F.mse_loss,
           side: str = "floor"):
    """Censored loss: past the limit only "is it past" is informative.

    Uncensored points get smooth L1; censored ones get the negative log-probability
    of falling below the floor. Without this, the model is pushed to fit the exact
    value of measurements that carry no quantitative information.

    `sigma` is the measurement-noise SD at the floor, IN Z-SPACE, and must be
    passed explicitly -- see TargetTransform.floor_sigma_z for why deriving it from
    floor_z is a category error.

    The two terms are weighted by their share of the batch and summed, which makes
    this exactly

        mean over all elements of [ censored ? -log CDF : obs_loss ]

    i.e. `Scripts.Train_loader.tobit_loss_floor` at `nonzero_weight=1`, the value
    every notebook passes. That function's `nonzero_weight=None` branch instead
    auto-balances the two classes from the batch; no notebook uses it, so it is
    not reproduced here.
    """
    if sigma is None or sigma <= 0:
        raise ValueError(f"tobit sigma must be positive, got {sigma!r}")
    # The ceiling case is the mirror image, and only the direction changes:
    #   floor    P(true <= cut | pred) = Phi((cut - pred) / sigma)
    #   ceiling  P(true >= cut | pred) = 1 - Phi((cut - pred) / sigma)
    #                                  = Phi((pred - cut) / sigma)
    # so one sign flip in z covers it. Written out rather than negating the
    # target, because every downstream convention -- higher_is_bad, the AUC
    # threshold, the flag column, calibration -- would otherwise have to be
    # negated too, and a single missed sign there is invisible.
    ceil = side == "ceiling"
    cens = (y >= floor_z) if ceil else (y <= floor_z)
    n = y.numel()
    loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
    if (~cens).any():
        w = (~cens).sum().float() / n
        loss = loss + obs_loss(pred[~cens], y[~cens]) * w
    if cens.any():
        w = cens.sum().float() / n
        z = ((pred[cens] - floor_z) if ceil else (floor_z - pred[cens])) / sigma
        cdf = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
        loss = loss - torch.log(cdf.clamp_min(1e-6)).mean() * w
    return loss


def masked_loss(model, batch, tasks: Sequence[str], target: str,
                target_weight: float, floor_z: Optional[float],
                floor_sigma: Optional[float], device,
                obs_loss=F.mse_loss, only: Optional[str] = None,
                censor_side: str = "floor", weighter=None) -> torch.Tensor:
    """Per task, only labelled rows contribute. An all-missing task adds nothing.

    `only` restricts the sum to one task and drops the task weighting, which is
    what validation-loss selection needs: the number that picks the epoch should
    be the target's objective, not a multi-task total whose scale moves with how
    many auxiliaries happened to be measured in the batch. `only` also bypasses
    `weighter` for the same reason -- a selection metric that moves with learned
    weights is not comparable across epochs.

    `weighter` (AuxUncertaintyWeighter) replaces the fixed auxiliary weights with
    learned ones; see `train.aux_weighting`.
    """
    x, m, xa, ma, g, y, ym = [t.to(device) for t in batch]
    per_task: Dict[str, torch.Tensor] = {}
    total, used = 0.0, 0
    for ti, task in enumerate(tasks):
        if only is not None and task != only:
            continue
        sel = ym[:, ti] > 0
        if sel.sum() < 2:
            continue
        pred = model(x[sel], xa[sel], task, mask=m[sel], mask_ab=ma[sel], x_global=g[sel])
        yt = y[sel, ti]
        if task == target and floor_z is not None:
            li = _tobit(pred, yt, floor_z, floor_sigma, obs_loss, censor_side)
        else:
            li = obs_loss(pred, yt)
        per_task[task] = li
        w = 1.0 if only is not None else (target_weight if task == target else 1.0)
        total = total + w * li
        used += 1
    if used == 0:
        return torch.zeros((), device=device, requires_grad=True)
    if weighter is not None and only is None and target in per_task:
        # The weighter pins the target, so a batch that happened to carry no
        # target-labelled rows falls back to the fixed sum rather than silently
        # optimising the auxiliaries alone.
        return weighter(per_task)
    return total


# --------------------------------------------------------------------------- #
# predict / train
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, loader, task: str, tasks: Sequence[str], device) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pred_z, true_z) for one task, in the loader's order."""
    model.eval()
    ti = list(tasks).index(task)
    P, T = [], []
    for x, m, xa, ma, g, y, ym in loader:
        x, m, xa, ma, g = [t.to(device) for t in (x, m, xa, ma, g)]
        p = model(x, xa, task, mask=m, mask_ab=ma, x_global=g)
        P.append(p.detach().cpu().numpy())
        t = y[:, ti].numpy().copy()
        t[ym[:, ti].numpy() == 0] = np.nan
        T.append(t)
    return np.concatenate(P), np.concatenate(T)


def selection_score(kind: str, rho: float, pearson: float, loss: float) -> float:
    """The number to MAXIMISE when picking an epoch. Mirrors
    Scripts.Train._selection_score.

    Spearman is scale-invariant, so a model that shrinks its predictions toward
    the mean pays no penalty for it -- which is how the SMP feature stream reached
    sd(pred) = 0.19 against sd(true) = 1.0 while still scoring rho = 0.57. `mse`
    penalises shrinkage and bias; `pearson` penalises shrinkage only.
    """
    if kind == "spearman":
        return rho
    if kind == "pearson":
        return pearson
    if kind == "mse":
        return -loss                       # lower loss is better -> negate
    raise ValueError(f"train.select_on must be mse, spearman or pearson, got {kind!r}")


@torch.no_grad()
def evaluate(model, loader, cfg: Config, tasks: Sequence[str], floor_z, floor_sigma,
             obs_loss, device, sel_floor_z=None) -> dict:
    """Validation numbers for one epoch: the target's objective, plus rho and
    pearson in z-space.

    Rank metrics are computed AFTER clipping both arrays at the noise floor, as
    Scripts.Train.train_simple._evaluate does: differences below the assay's
    detection limit are not measurable, so ranking on them is ranking on noise.
    """
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        b = int(batch[0].shape[0])
        total += float(masked_loss(model, batch, tasks, cfg.target, 1.0,
                                   floor_z, floor_sigma, device,
                                   obs_loss=obs_loss, only=cfg.target,
                                   censor_side=cfg.target_opts.censor_side)) * b
        n += b
    p, t = predict(model, loader, cfg.target, tasks, device)
    ok = np.isfinite(p) & np.isfinite(t)
    pc, tc = p[ok], t[ok]
    _f = sel_floor_z if sel_floor_z is not None else floor_z
    if _f is not None:
        pc, tc = np.maximum(pc, _f), np.maximum(tc, _f)
    m = score(tc, pc)
    return {"loss": total / max(n, 1), "rho": m["rho"], "pearson": m["pearson"]}


def train_model(model: FvModel, loader_tr, loader_va, cfg: Config,
                tasks: Sequence[str], epochs: Optional[int] = None,
                lr: Optional[float] = None, floor_z: Optional[float] = None,
                floor_sigma: Optional[float] = None, sel_floor_z: Optional[float] = None,
                device: str = "cpu", verbose: bool = False) -> Tuple[FvModel, dict]:
    """Fit for `epochs`, early-stopping on `loader_va` by Spearman on the target.

    `loader_va=None` disables early stopping AND checkpoint selection: the model
    returned is the one at the final epoch. Passing the training set as `loader_va`
    would select the epoch that best fits the data being fitted, which is the most
    overfit one -- so callers with no genuine held-out fold must pass None.
    """
    epochs = cfg.train.epochs if epochs is None else epochs
    lr = cfg.train.lr if lr is None else lr
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(device)

    # Adam, not AdamW, by default: the notebooks use Adam, which folds weight decay
    # into the gradient rather than decoupling it. At weight_decay=5e-3 the two are
    # not interchangeable, so the choice is configuration, not a detail.
    make_opt = (torch.optim.Adam if cfg.train.optimizer == "adam" else torch.optim.AdamW)
    tw = cfg.target_weight if len(tasks) > 1 else 1.0

    # Learned auxiliary weights. Single-task runs never get one: with no
    # auxiliaries there is nothing to weight, and constructing it anyway would
    # consume RNG draws and shift every subsequent weight init, so a single-task
    # baseline would stop being comparable across the two settings.
    weighter = None
    if cfg.train.aux_weighting == "uncertainty" and len(tasks) > 1:
        weighter = AuxUncertaintyWeighter(
            tasks, target=cfg.target, target_weight=tw,
            init_weight=cfg.train.aux_init_weight).to(device)

    groups = [{"params": list(model.parameters()),
               "weight_decay": cfg.train.weight_decay}]
    if weighter is not None:
        groups.append(weighter.param_group())     # weight decay OFF, see the class
    opt = make_opt(groups, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    obs_loss = pointwise(cfg.train.loss)

    best = {"score": -np.inf, "rho": np.nan, "epoch": -1, "state": None}
    history, bad = [], 0

    for ep in range(epochs):
        model.train()
        run, nb = 0.0, 0
        for batch in loader_tr:
            opt.zero_grad()
            loss = masked_loss(model, batch, tasks, cfg.target, tw,
                               floor_z, floor_sigma, device, obs_loss=obs_loss,
                               censor_side=cfg.target_opts.censor_side,
                               weighter=weighter)
            loss.backward()
            if cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            run += float(loss.detach()); nb += 1
        sched.step()

        row = {"epoch": ep, "loss": run / max(nb, 1),
               "val_loss": np.nan, "val_rho": np.nan, "val_pearson": np.nan}
        if loader_va is not None:
            va = evaluate(model, loader_va, cfg, tasks, floor_z, floor_sigma,
                          obs_loss, device,
                          sel_floor_z=sel_floor_z if sel_floor_z is not None else floor_z)
            row.update(val_loss=va["loss"], val_rho=va["rho"], val_pearson=va["pearson"])
            s = selection_score(cfg.train.select_on, va["rho"], va["pearson"], va["loss"])
            # select_best=False: run the full budget and keep the LAST epoch.
            # Checkpoint selection scores every epoch against the validation fold
            # and keeps the winner, so with a small fold the chosen epoch is
            # partly a draw from that fold's noise -- and the same molecules are
            # then scored. Fine on a normal fold of hundreds; the dominant effect
            # on a fold of twenty, where it can make an untrained model look best.
            if not getattr(cfg.train, "select_best", True):
                best = {"score": float(s) if np.isfinite(s) else -np.inf,
                        "rho": float(va["rho"]), "epoch": ep, "state": None}
            elif np.isfinite(s) and s > best["score"]:
                best = {"score": float(s), "rho": float(va["rho"]), "epoch": ep,
                        "state": copy.deepcopy(model.state_dict())}
                bad = 0
            else:
                bad += 1
        history.append(row)
        if verbose and ep % 20 == 0:
            print(f"      ep {ep:3d}  loss {row['loss']:.4f}  "
                  f"val_loss {row['val_loss']:.4f}  val_rho {row['val_rho']:.3f}")
        if loader_va is not None and bad >= cfg.train.patience and ep > 10:
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    # The learned weights are from the LAST epoch, not the selected one: only the
    # model is checkpointed, and the weighter affects training alone. Reported so
    # that the drift away from "the target holds half the loss" is visible -- it
    # is the one thing about the run a reader cannot infer from the config.
    return model, {"epoch": best["epoch"], "rho": best["rho"],
                   "score": best["score"], "select_on": cfg.train.select_on,
                   "aux_weights": (weighter.weights() if weighter is not None
                                   else None),
                   "history": history}


# --------------------------------------------------------------------------- #
# warm start
# --------------------------------------------------------------------------- #
def _build(cfg: Config, arrays: dict, tasks, streams) -> FvModel:
    """One construction site for FvModel, so CV, the final fit and the saved
    architecture cannot drift apart."""
    return FvModel(tasks=tasks, streams=streams, **arch_kwargs(cfg, arrays))


def arch_kwargs(cfg: Config, arrays: dict) -> dict:
    """Every constructor argument that is not `tasks` or `streams`. Saved verbatim
    in the bundle, so inference rebuilds exactly what training built."""
    return dict(
        n_feat=arrays["X"].shape[-1] if arrays.get("X") is not None else 1,
        ablang_dim=arrays["X_ab"].shape[-1] if arrays.get("X_ab") is not None else 480,
        n_global=arrays["Xg"].shape[-1] if arrays.get("Xg") is not None else 0,
        hidden=cfg.model.hidden, ab_hidden=cfg.model.ab_hidden,
        head=cfg.model.head, ab_head=cfg.model.ab_head,
        dropout=cfg.model.dropout, ab_mode=cfg.model.ab_mode,
        pool=cfg.model.pool, ab_pool=cfg.ab_pool,
        simple_head=cfg.model.simple_head,
        ab_simple_head=cfg.ab_simple_head,
        init_w=cfg.model.init_w, freeze_w=cfg.model.freeze_w)


def _fit_ab_stats(arrays: dict, idx):
    """Per-channel mean/sd of the embedding stream, fitted on `idx` only.

    Returns (None, None) when there is no embedding stream, so callers can
    store it unconditionally.
    """
    if arrays.get("X_ab") is None:
        return None, None
    am, asd = fit_feature_stats(arrays["X_ab"][idx], arrays["mask_ab"][idx],
                                what="embedding")
    arrays["X_ab"] = apply_feature_stats(arrays["X_ab"], arrays["mask_ab"], am, asd)
    return am, asd


def warm_start(cfg: Config, arrays: dict, tr_idx, va_idx, Z, seed: int, device: str,
               epochs: Optional[int | dict] = None,
               ) -> Tuple[List[FvModel], Dict[str, int]]:
    """Fit each requested stream on its own, single-task, for use as init.

    With both streams copied in, a late-fusion model starts exactly at the plain
    average of the two. Both arms of a comparison get the identical pair, so the
    warm models' own early stopping is common to both and cancels.

    `va_idx=None` -> no early stopping, run `epochs` and keep the last epoch. That
    is the only correct mode when there is no held-out fold; see `fit_final`.

    `epochs` is PER STREAM: a {stream: epochs} mapping, or one int for all of them
    (used when no CV evidence exists yet). The streams converge at very different
    rates -- frozen embeddings against 3D features -- so a single shared budget is
    two mistakes at once: one stream trained past where it stopped generalising,
    the other stopped while still improving.

    Returns the models and {stream: best epoch} (-1 where nothing stopped it).
    """
    out, stopped = [], {}
    for s in cfg.streams:
        if isinstance(epochs, dict):
            n = epochs.get(s) or cfg.train.warm_epochs
        else:
            n = epochs if epochs is not None else cfg.train.warm_epochs
        set_seed(seed)
        m = _build(cfg, arrays, [cfg.target], [s])
        Zt = Z[:, :1]
        ld_tr = make_loader(tr_idx, cfg.train.batch_size, True, Z=Zt, **arrays)
        ld_va = (make_loader(va_idx, cfg.train.batch_size, False, Z=Zt, **arrays)
                 if va_idx is not None else None)
        m, info = train_model(m, ld_tr, ld_va, cfg, [cfg.target],
                              epochs=n, device=device)
        out.append(m)
        stopped[s] = int(info["epoch"])
    return out, stopped


# --------------------------------------------------------------------------- #
# one fold
# --------------------------------------------------------------------------- #
def run_fold(cfg: Config, arrays_raw: dict, Y: np.ndarray, tr_idx, va_idx,
             seed: int, device: str) -> dict:
    """Fit on tr_idx, score on va_idx. All preprocessing fitted on tr_idx only."""
    tasks = cfg.tasks

    tfs = fit_transforms(Y, tr_idx, tasks, log=cfg.target_opts.log)
    Z = apply_transforms(Y, tfs, tasks)
    # `floor_z` drives the LOSS and is gated on censor_loss; `sel_floor_z` drives
    # the selection metric and is not. Early stopping always censors -- ordering
    # below the detection limit is ordering on noise -- while whether the loss
    # censors is the caller's choice.
    sel_floor_z = tfs[cfg.target].floor_z(cfg.target_opts.noise_floor)
    floor_z = sel_floor_z if cfg.target_opts.censor_loss else None
    floor_sigma = tfs[cfg.target].floor_sigma_z(cfg.target_opts.noise_floor,
                                                cfg.target_opts.floor_sigma_frac)

    arrays = dict(arrays_raw)
    if arrays.get("X") is not None:
        mean, sd = fit_feature_stats(arrays["X"][tr_idx], arrays["mask"][tr_idx])
        arrays["X"] = apply_feature_stats(arrays["X"], arrays["mask"], mean, sd)
    else:
        mean = sd = None
    # The EMBEDDING stream needs the same treatment, and for a stronger reason.
    # AbLang2 comes out near unit scale so this is a no-op there, but a mid-stack
    # ESM block is unnormalised (only `last_hidden_state` is past the final
    # LayerNorm) and carries the massive-activation channels transformers are
    # known for -- sd ~9 with individual values near 500. Unstandardised, a
    # handful of channels dominate the trunk and the stream learns nothing.
    ab_mean, ab_sd = _fit_ab_stats(arrays, tr_idx)

    warm, warm_ep = ([], {})
    if cfg.train.warm_start:
        warm, warm_ep = warm_start(cfg, arrays, tr_idx, va_idx, Z, seed, device)

    set_seed(seed)
    model = _build(cfg, arrays, tasks, cfg.streams)
    if warm:
        model.load_from(*warm)

    ld_tr = make_loader(tr_idx, cfg.train.batch_size, True, Z=Z, **arrays)
    ld_va = make_loader(va_idx, cfg.train.batch_size, False, Z=Z, **arrays)
    model, info = train_model(model, ld_tr, ld_va, cfg, tasks, floor_z=floor_z,
                              floor_sigma=floor_sigma, sel_floor_z=sel_floor_z,
                              device=device)

    pz, tz = predict(model, ld_va, cfg.target, tasks, device)
    tf = tfs[cfg.target]
    y_pred, y_true = tf.inverse(pz), tf.inverse(tz)
    m = score(y_true, y_pred,
              threshold=cfg.target_opts.threshold,
              higher_is_bad=cfg.target_opts.higher_is_bad,
              noise_floor=cfg.target_opts.noise_floor)
    return {"metrics": m, "model": model, "tfs": tfs, "feat_mean": mean, "feat_sd": sd,
            "ab_mean": ab_mean, "ab_sd": ab_sd,
            "epoch": info["epoch"], "warm_epochs": warm_ep,
            "blend": model.blend_weights(), "aux_weights": info.get("aux_weights"),
            "va_idx": np.asarray(va_idx), "pred": y_pred, "true": y_true}


def fit_final(cfg: Config, arrays_raw: dict, Y: np.ndarray, rows_idx, epochs: int,
              seed: int, device: str,
              warm_epochs: Optional[int | dict] = None) -> dict:
    """Fit on EVERY labelled molecule for a fixed number of epochs.

    There is no validation set here, so there is nothing to early-stop on -- and
    carving one out of the training data would both shrink it and stop on data the
    model is fitting. `epochs` comes from the median best epoch observed in CV,
    where a genuine held-out fold was available. It is the epoch of the FUSED
    model, which has a single stopping curve, so there is nothing per-stream about
    it.

    The WARM MODELS obey the same rule. Passing `rows_idx` as their validation set
    -- as this function used to -- makes `train_model` select the epoch that best
    fits the training data, i.e. the most overfit one, and then copies those
    weights into the final model.

    `warm_epochs` is PER STREAM ({stream: epochs}), for the same reason `epochs`
    is inherited at all: each warm model's budget should be the one ITS OWN
    stopping curve justified. A single shared number -- which this took until the
    per-stream change -- is the median of two, so it overtrains the stream that
    stopped early and undertrains the one that was still improving, and those
    weights are then the initialisation the final model builds on.
    """
    tasks = cfg.tasks
    tfs = fit_transforms(Y, rows_idx, tasks, log=cfg.target_opts.log)
    Z = apply_transforms(Y, tfs, tasks)
    # `floor_z` drives the LOSS and is gated on censor_loss; `sel_floor_z` drives
    # the selection metric and is not. Early stopping always censors -- ordering
    # below the detection limit is ordering on noise -- while whether the loss
    # censors is the caller's choice.
    sel_floor_z = tfs[cfg.target].floor_z(cfg.target_opts.noise_floor)
    floor_z = sel_floor_z if cfg.target_opts.censor_loss else None
    floor_sigma = tfs[cfg.target].floor_sigma_z(cfg.target_opts.noise_floor,
                                                cfg.target_opts.floor_sigma_frac)

    arrays = dict(arrays_raw)
    mean = sd = None
    if arrays.get("X") is not None:
        mean, sd = fit_feature_stats(arrays["X"][rows_idx], arrays["mask"][rows_idx])
        arrays["X"] = apply_feature_stats(arrays["X"], arrays["mask"], mean, sd)
    ab_mean, ab_sd = _fit_ab_stats(arrays, rows_idx)

    warm = []
    if cfg.train.warm_start:
        # va_idx=None -> no early stopping, keep the last epoch.
        warm, _ = warm_start(cfg, arrays, rows_idx, None, Z, seed, device,
                             epochs=warm_epochs if warm_epochs
                             else cfg.train.warm_epochs)

    set_seed(seed)
    model = _build(cfg, arrays, tasks, cfg.streams)
    if warm:
        model.load_from(*warm)

    ld = make_loader(rows_idx, cfg.train.batch_size, True, Z=Z, **arrays)
    model, _info = train_model(model, ld, None, cfg, tasks, epochs=epochs,
                               floor_z=floor_z, floor_sigma=floor_sigma,
                               sel_floor_z=sel_floor_z, device=device)
    return {"model": model, "tfs": tfs, "feat_mean": mean, "feat_sd": sd,
            "ab_mean": ab_mean, "ab_sd": ab_sd,
            "blend": model.blend_weights(), "aux_weights": _info.get("aux_weights")}


def cross_validate(cfg: Config, arrays_raw: dict, Y: np.ndarray,
                   rounds, device: str = "cpu"):
    """Returns (metric_rows, oof) -- oof holds one out-of-fold prediction per
    molecule per seed, so the per-seed spread is visible molecule by molecule."""
    rows, oof = [], []
    for fold, (tr, va) in enumerate(rounds):
        for seed in cfg.train.seeds:
            r = run_fold(cfg, arrays_raw, Y, tr, va, seed, device)
            # One column PER STREAM. The old single `warm_epoch` was the median
            # across streams, i.e. the mean of two numbers that are neither --
            # and it was then applied to both, training one stream past its
            # stopping point and the other short of it.
            row = {"fold": fold, "seed": seed, "epoch": r["epoch"], **r["metrics"]}
            for s in cfg.streams:
                row[f"warm_epoch_{s}"] = int(r["warm_epochs"].get(s, -1))
            if r["blend"]:
                row["w_feat"] = round(r["blend"][cfg.target], 3)
            rows.append(row)
            for i, p, t in zip(r["va_idx"], r["pred"], r["true"]):
                oof.append({"row": int(i), "fold": fold, "seed": seed,
                            "true": float(t), "pred": float(p)})
            print(f"  fold {fold} seed {seed}: rho={row['rho']:.3f} "
                  f"pearson={row['pearson']:.3f} auc={row['auc']:.3f} (ep {row['epoch']})")
    return rows, oof
