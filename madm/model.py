"""One model. Two switches.

    streams = ["feat"]              structural features only
              ["ablang"]            AbLang2 embeddings only
              ["feat", "ablang"]    late fusion, per-assay blend weight

    tasks   = [target]              single-task
              [target] + aux        multi-task, shared encoders

The four models in the paper are the four combinations. They are not four classes:
a single-stream model is a two-stream model without the other stream, and a
single-task model is a multi-task model with one task. Keeping them as one class
is what lets there be one training loop, one loss, one checkpoint format.

LATE FUSION, not concatenation:

    p_a = w_a * feat_head_a(z_feat, z_glob) + (1 - w_a) * ab_head_a(z_ab)
    w_a = sigmoid(w_logit_a)      one learned scalar per assay

The streams never mix before the output. A shared head over concatenated features
can lower the training loss by leaning on whichever stream fits more easily, which
destroys the error independence that makes combining worth doing at all.

Note `w_a` is a model parameter, so it is fitted on the TRAINING objective: it
measures which stream fits training better, not which generalises better. On a
held-out source from unseen sequence space that distinction matters.

Global features enter the FEATURE head only. The AbLang stream never sees them.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# pooling
# --------------------------------------------------------------------------- #
class MaskedPool(nn.Module):
    """[N,C,L] -> [N,2C], ignoring padded positions.

    attn     -> [attention-weighted sum | max]
    meanmax  -> [mean | max]

    Matched to Scripts.models_shared._Pooler, including two details that are not
    cosmetic:

      * the attention score is a Linear over the CHANNEL axis, ZERO-INITIALISED,
        so attention starts as exact mean pooling and has to earn any departure
        from it. A default-initialised score starts at an arbitrary weighting.
      * a fully-masked row gives 0, not NaN. Softmax over all -inf is NaN, and the
        meanmax denominator is clamped at 1.0 rather than 1e-6, which would turn
        an empty row into ~1e6.
    """

    def __init__(self, channels: int, mode: str = "attn"):
        super().__init__()
        if mode not in ("attn", "meanmax"):
            raise ValueError(f"pool must be attn or meanmax, got {mode!r}")
        self.mode = mode
        if mode == "attn":
            self.score = nn.Linear(channels, 1)
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None):
        neg = torch.finfo(h.dtype).min
        if mask is not None:
            m = mask.unsqueeze(1) if mask.dim() == 2 else mask     # [N,1,L]
            m = m[..., :h.shape[-1]].float()
        else:
            m = torch.ones(h.shape[0], 1, h.shape[-1], device=h.device, dtype=h.dtype)
        mb = m > 0

        mx = h.masked_fill(~mb, neg).max(dim=-1).values
        mx = torch.where(mb.any(-1), mx, torch.zeros_like(mx))     # all-masked -> 0

        if self.mode == "meanmax":
            avg = (h * m).sum(-1) / m.sum(-1).clamp(min=1.0)
        else:
            s = self.score(h.transpose(1, 2)).squeeze(-1)          # [N,L]
            s = s.masked_fill(~mb.squeeze(1), neg)
            a = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
            avg = (a.unsqueeze(1) * h).sum(-1)
        return torch.cat([avg, mx], dim=-1)


def _head(in_dim: int, width: int, dropout: float, simple: bool = True) -> nn.Sequential:
    """simple -> Dropout -> Linear, which is what the per-assay configurations in
    MultiTask.ipynb use for SMP, Ovalbumin, DNA, Insulin and BVP.
    otherwise -> Linear -> LayerNorm -> GELU -> Dropout -> Linear (HIC, AC-SINS).

    Both match Scripts.models_shared._head exactly, so a warm start from a
    single-stream model is a weight copy rather than an approximation."""
    if simple:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, 1))
    return nn.Sequential(
        nn.Linear(in_dim, width), nn.LayerNorm(width),
        nn.GELU(), nn.Dropout(dropout),
        nn.Linear(width, 1))


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #
class FeatureEncoder(nn.Module):
    """Per-residue structural features [N,L,F] -> [N,2*hidden].

    conv(k5) -> pool/2 -> conv(k5)+residual -> conv(k3)+residual -> masked pool.
    """

    def __init__(self, n_feat: int, hidden: int = 64, dropout: float = 0.3,
                 pool: str = "attn"):
        super().__init__()
        assert hidden % 8 == 0, "hidden must be divisible by 8 (GroupNorm)"

        def blk(cin, k):
            return nn.Sequential(
                nn.Conv1d(cin, hidden, k, padding=k // 2),
                nn.GroupNorm(8, hidden, eps=1e-4), nn.GELU(), nn.Dropout1d(dropout))

        self.conv1, self.conv2, self.conv3 = blk(n_feat, 5), blk(hidden, 5), blk(hidden, 3)
        self.pooler = MaskedPool(hidden, pool)
        self.norm = nn.LayerNorm(2 * hidden)
        self.out_dim = 2 * hidden

    def forward(self, x, mask=None):
        h = F.max_pool1d(self.conv1(x.transpose(1, 2)), 2, 2)   # [N,C,L/2]
        h = self.conv2(h) + h
        h = self.conv3(h) + h
        m = F.max_pool1d(mask.unsqueeze(1).float(), 2, 2) if mask is not None else None
        return self.norm(self.pooler(h, m))


class AbLangEncoder(nn.Module):
    """Frozen AbLang2 embeddings [N,L,D] -> [N,2*hidden].

    small -> one linear projection
    large -> 3-layer MLP
    conv  -> linear to 2*hidden, then two conv blocks each halving the length
    """

    def __init__(self, ablang_dim: int = 480, hidden: int = 64, dropout: float = 0.3,
                 mode: str = "small", pool: str = "attn"):
        super().__init__()
        assert hidden % 8 == 0, "hidden must be divisible by 8 (GroupNorm)"
        if mode not in ("small", "large", "conv"):
            raise ValueError(f"ab_mode must be small/large/conv, got {mode!r}")
        self.mode = mode

        if mode == "small":
            self.proj = nn.Sequential(nn.Linear(ablang_dim, hidden),
                                      nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        elif mode == "large":
            self.proj = nn.Sequential(
                nn.Linear(ablang_dim, hidden * 2), nn.LayerNorm(hidden * 2),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden * 2), nn.LayerNorm(hidden * 2),
                nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden))
        else:
            self.proj = nn.Sequential(nn.Linear(ablang_dim, hidden * 2),
                                      nn.LayerNorm(hidden * 2), nn.GELU(), nn.Dropout(dropout))

            def blk(cin, k):
                return nn.Sequential(
                    nn.Conv1d(cin, hidden, k, padding=k // 2),
                    nn.GroupNorm(8, hidden, eps=1e-4), nn.GELU(),
                    nn.Dropout1d(dropout), nn.MaxPool1d(2, 2))

            self.block1, self.block2 = blk(hidden * 2, 3), blk(hidden, 5)

        self.pooler = MaskedPool(hidden, pool)
        self.norm = nn.LayerNorm(2 * hidden)
        self.out_dim = 2 * hidden

    def forward(self, x_ab, mask=None):
        h = self.proj(x_ab).transpose(1, 2)
        m = mask.unsqueeze(1).float() if mask is not None else None
        if self.mode == "conv":
            h = self.block2(self.block1(h))
            if m is not None:
                m = F.max_pool1d(F.max_pool1d(m, 2, 2), 2, 2)
        return self.norm(self.pooler(h, m))


# --------------------------------------------------------------------------- #
# auxiliary task weighting
# --------------------------------------------------------------------------- #
class AuxUncertaintyWeighter(nn.Module):
    """Homoscedastic-uncertainty weighting of the AUXILIARIES only.

    Ported from Scripts.models_shared.UncertaintyWeighter, which the MultiTask
    notebook uses. It is the DEFAULT here (`train.aux_weighting: uncertainty`)
    because it is what produced the published multi-task numbers; `fixed` keeps
    every auxiliary at 1.0 for the whole run and is available for ablation.

    Kendall, Gal & Cipolla (2018) learn one log-variance s_t per task and combine

        L = sum_t  0.5 * exp(-s_t) * L_t  +  0.5 * s_t

    The `+ 0.5 * s_t` term is what makes it learnable: without it the minimiser
    drives every weight to zero.

    WHY THE TARGET IS EXCLUDED. Plain uncertainty weighting BALANCES tasks -- it
    is derived for the case where all of them matter equally. The learned weight
    is inversely proportional to a task's fitted noise, so a NOISY TARGET gets
    down-weighted and capacity goes to the cleaner auxiliaries. For a
    target-focused run that is backwards, and it silently overrides the derived
    `max(len(aux), 1)` target weight. Here the target is pinned at
    `target_weight` and only the auxiliaries are learned, so the model can decide
    which donor assays are worth listening to -- switching off ones that transfer
    negatively -- while the target's weight is never at risk.

    CONSEQUENCE FOR REPORTING. The target's SHARE of the loss then drifts upward
    as auxiliaries are down-weighted, so "the target holds half the loss" is true
    only at initialisation. `weights()` returns the end-of-training values; the
    CLI logs them, because that drift is the one thing a reader cannot infer from
    the configuration.

    init_weight
        Effective weight of each learned task at initialisation. 0.5 reproduces
        the plain Kendall form at s_t = 0; 1.0 (the default here) starts a run at
        exactly the fixed-weight baseline, so training can only move away from it
        deliberately -- the same argument as warm-starting a fused model at the
        plain average of its two streams.

    Weight decay must NOT be applied to the log-variances: decaying s_t pulls
    every weight back toward `init_weight`, a hidden prior fighting the quantity
    being learned. Use `param_group()`.
    """

    def __init__(self, tasks: Sequence[str], target: str, target_weight: float,
                 init_weight: float = 1.0):
        super().__init__()
        self.tasks = list(tasks)
        if target not in self.tasks:
            raise ValueError(f"target {target!r} not in tasks {self.tasks}")
        if init_weight <= 0:
            raise ValueError(f"aux_init_weight must be positive, got {init_weight}")
        self.target = target
        self.target_weight = float(target_weight)
        s0 = math.log(1.0 / (2.0 * float(init_weight)))   # 0.5*exp(-s0) == init_weight
        # ParameterDict keys must be identifier-like; assay names carry '-' and '.'
        self._key = {t: t.replace("-", "_").replace(".", "_") for t in self.tasks}
        self.learned = [t for t in self.tasks if t != target]
        self.log_var = nn.ParameterDict(
            {self._key[t]: nn.Parameter(torch.full((), s0)) for t in self.learned})

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for t, L in losses.items():
            if t not in self._key:
                raise KeyError(f"loss for unknown task {t!r}; weighter knows {self.tasks}")
            if t == self.target:
                term = self.target_weight * L
            else:
                s = self.log_var[self._key[t]]
                term = 0.5 * torch.exp(-s) * L + 0.5 * s
            total = term if total is None else total + term
        return total

    def param_group(self, lr=None) -> dict:
        g = {"params": list(self.parameters()), "weight_decay": 0.0}
        if lr is not None:
            g["lr"] = lr
        return g

    @torch.no_grad()
    def weights(self) -> Dict[str, float]:
        """{task -> effective weight on its loss}. The target's is fixed."""
        w = {t: 0.5 * float(torch.exp(-self.log_var[self._key[t]])) for t in self.learned}
        w[self.target] = self.target_weight
        return w


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class FvModel(nn.Module):
    def __init__(self, n_feat: int, tasks: Sequence[str],
                 streams: Sequence[str] = ("feat", "ablang"),
                 ablang_dim: int = 480, n_global: int = 0,
                 hidden: int = 64, ab_hidden: int = 64,
                 head: int = 16, ab_head: int = 32,
                 dropout: float = 0.25, ab_mode: str = "small",
                 pool: str = "attn", ab_pool: Optional[str] = None,
                 simple_head: bool = True,
                 ab_simple_head: Optional[bool] = None, init_w: float = 0.5,
                 freeze_w: bool = False):
        super().__init__()
        self.tasks = list(tasks)
        self.streams = tuple(streams)
        self.n_global = n_global
        if not self.streams:
            raise ValueError("at least one stream required")
        ab_simple_head = simple_head if ab_simple_head is None else bool(ab_simple_head)
        # The two streams pool independently, like their heads do. The notebooks
        # use attention on the feature trunk and the per-assay `Abl_pool` on the
        # AbLang trunk -- meanmax for SMP -- and a single shared setting cannot
        # express that. None follows `pool`, so old configs behave as before.
        ab_pool = pool if ab_pool is None else str(ab_pool)

        if "feat" in self.streams:
            self.f_encoder = FeatureEncoder(n_feat, hidden, dropout, pool)
            self.feat_heads = nn.ModuleDict(
                {t: _head(self.f_encoder.out_dim + n_global, head, dropout, simple_head)
                 for t in self.tasks})
        if "ablang" in self.streams:
            self.ab_encoder = AbLangEncoder(ablang_dim, ab_hidden, dropout, ab_mode, ab_pool)
            self.ab_heads = nn.ModuleDict(
                {t: _head(self.ab_encoder.out_dim, ab_head, dropout, ab_simple_head)
                 for t in self.tasks})

        if len(self.streams) == 2:
            # `freeze_w` pins the blend at `init_w` instead of learning it. With
            # init_w = 0.5 that is the plain average of the two streams, which is
            # the control for "does a LEARNED blend help, or just ensembling?" --
            # the two are indistinguishable from the fused score alone.
            init_w = float(min(max(init_w, 1e-4), 1 - 1e-4))
            w0 = math.log(init_w / (1.0 - init_w))
            self.w_logit = nn.ParameterDict(
                {t: nn.Parameter(torch.tensor(w0), requires_grad=not freeze_w)
                 for t in self.tasks})

    # ---- introspection ----------------------------------------------------
    def blend_weights(self) -> Dict[str, float]:
        """{assay: weight on the FEATURE stream}. Empty for single-stream models.
        Near 0 or 1 means that assay is effectively using one stream."""
        if len(self.streams) < 2:
            return {}
        return {t: float(torch.sigmoid(p).detach()) for t, p in self.w_logit.items()}

    # ---- forward ----------------------------------------------------------
    def forward(self, x, x_ab, task: str, mask=None, mask_ab=None, x_global=None,
                return_streams: bool = False):
        p_f = p_a = None
        if "feat" in self.streams:
            z = self.f_encoder(x, mask)
            if self.n_global and x_global is not None:
                z = torch.cat([z, x_global], dim=-1)
            p_f = self.feat_heads[task](z).squeeze(-1)
        if "ablang" in self.streams:
            p_a = self.ab_heads[task](self.ab_encoder(x_ab, mask_ab)).squeeze(-1)

        if len(self.streams) == 2:
            w = torch.sigmoid(self.w_logit[task])
            p = w * p_f + (1.0 - w) * p_a
        else:
            p = p_f if p_f is not None else p_a
        return (p, p_f, p_a) if return_streams else p

    # ---- warm start -------------------------------------------------------
    def load_from(self, *singles: "FvModel") -> "FvModel":
        """Copy encoder AND head weights from single-stream models.

        With both streams copied, a late-fusion model starts at exactly the plain
        average of the two, so training can only move it away from that baseline
        deliberately. Heads are copied only for tasks the source model has -- an
        auxiliary head has no single-stream counterpart and stays at init.
        """
        for src in singles:
            for s_ in src.streams:
                if s_ not in self.streams:
                    continue
                if s_ == "feat":
                    self.f_encoder.load_state_dict(src.f_encoder.state_dict())
                    for t in self.tasks:
                        if t in src.feat_heads:
                            self.feat_heads[t].load_state_dict(src.feat_heads[t].state_dict())
                else:
                    self.ab_encoder.load_state_dict(src.ab_encoder.state_dict())
                    for t in self.tasks:
                        if t in src.ab_heads:
                            self.ab_heads[t].load_state_dict(src.ab_heads[t].state_dict())
        return self

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        w = self.blend_weights()
        return (f"FvModel streams={'+'.join(self.streams)} tasks={self.tasks} "
                f"params={n:,}" + (f" w_feat={ {k: round(v,3) for k,v in w.items()} }" if w else ""))
