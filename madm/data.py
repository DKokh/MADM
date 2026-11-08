"""Targets, splits, loaders.

TARGET TRANSFORM.  Assay values are right-skewed, so training happens on
log1p-then-z-scored values. The transform is FITTED ON TRAINING ROWS ONLY and
carried with the model, because inference has no labels to fit it from.

SPLITS.  Two independent choices: how molecules are CLUSTERED, and how clusters
become FOLDS. Both come from `split:` in the config; see SplitCfg for what each
value means and which notebook switch it corresponds to.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .table import clean_seq

# torch is imported lazily, at the bottom of this file, so that the target
# transforms, clustering and fold logic can be used (and tested) without a
# deep-learning install.


# --------------------------------------------------------------------------- #
# target transform
# --------------------------------------------------------------------------- #
@dataclass
class TargetTransform:
    mu: float
    sd: float
    log: bool

    def forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.log:
            y = np.log1p(np.maximum(y, 0.0))
        return (y - self.mu) / self.sd

    def inverse(self, z: np.ndarray, cap: float = 50.0) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        y = z * self.sd + self.mu
        if self.log:
            y = np.expm1(np.clip(y, -cap, cap))
        return y

    @classmethod
    def fit(cls, y: np.ndarray, log: bool = True) -> "TargetTransform":
        y = np.asarray(y, dtype=np.float64)
        y = y[np.isfinite(y)]
        if y.size == 0:
            raise ValueError("cannot fit a target transform on zero finite values")
        if log:
            y = np.log1p(np.maximum(y, 0.0))
        sd = float(y.std())
        return cls(mu=float(y.mean()), sd=sd if sd > 1e-8 else 1.0, log=log)

    def floor_z(self, floor: Optional[float]) -> Optional[float]:
        """A noise floor given in ORIGINAL units, expressed in z-space."""
        if floor is None:
            return None
        return float(self.forward(np.array([floor]))[0])

    def floor_sigma_z(self, floor: Optional[float], frac: float = 0.5) -> Optional[float]:
        """Measurement-noise SD at the floor, in the target's z-space.

        The censored loss needs a SCALE, and `floor_z` is not one: it is the
        floor's DISTANCE FROM THE LABEL MEAN. A floor that happens to sit near the
        mean drives floor_z -> 0, so a sigma derived from it collapses and turns
        -log CDF((floor_z - pred)/sigma) into a near-vertical hinge. Measured on
        the real assays, that mistake is wrong by 0.2x to 17x, in both directions.

        Instead: take the floor's MAGNITUDE in the space the labels were
        standardised in, scale it by `frac`, and divide by the label sd. Mirrors
        Scripts.Splitter.floor_sigma_z.
        """
        if floor is None:
            return None
        f = float(floor)
        if self.log:
            f = float(np.log1p(max(f, 0.0)))
        sigma = abs(f) * float(frac) / self.sd
        return float(sigma) if sigma > 1e-6 else 1e-6

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TargetTransform":
        return cls(**d)


def fit_transforms(Y: np.ndarray, rows: np.ndarray, tasks: Sequence[str],
                   log: bool = True) -> Dict[str, TargetTransform]:
    """One transform per task, fitted on `rows` only."""
    return {t: TargetTransform.fit(Y[rows, j], log=log) for j, t in enumerate(tasks)}


def apply_transforms(Y: np.ndarray, tfs: Dict[str, TargetTransform],
                     tasks: Sequence[str]) -> np.ndarray:
    Z = np.full_like(Y, np.nan, dtype=np.float64)
    for j, t in enumerate(tasks):
        ok = np.isfinite(Y[:, j])
        Z[ok, j] = tfs[t].forward(Y[ok, j])
    return Z


# --------------------------------------------------------------------------- #
# output calibration
# --------------------------------------------------------------------------- #
@dataclass
class Calibrator:
    """Rescales predictions onto the scale of the original training values.

    A regularised model shrinks: its predictions have less spread than the truth.
    Inverting the z-score gives back the right UNITS but not the right RANGE, and
    because expm1 is convex, a mild linear shrinkage in z-space becomes a bent,
    compressed relationship in assay units. Predictions then look systematically
    "too average" -- a real value of 22 comes back as 8.

    Every mode here is monotone, so the ORDERING is preserved. That is not the
    same as leaving Spearman and AUC untouched: `quantile` is monotone but not
    STRICTLY monotone, and where the training values have an atom -- a zero-
    inflated assay like AC-SINS or SMP, where most antibodies simply do not
    self-associate -- the map is flat there and collapses a whole range of
    predictions onto one value. Those ties cost rank metrics. So score the
    UNCALIBRATED prediction and keep the calibrated one for assay units;
    `madm predict` and the CV figures both do this.

    Fitted on OUT-OF-FOLD predictions, never on the training fit, otherwise it
    learns the model's optimism instead of its bias.

        none      leave predictions as they are
        linear    match the mean and standard deviation of the training values
        quantile  map predicted quantiles onto training-value quantiles, so the
                  predicted distribution matches the training distribution

    None of this makes the ranking better. It makes the numbers usable -- for
    thresholds, for plots against measured values, for anything reported in assay
    units rather than as a rank.
    """
    kind: str = "none"
    x: Optional[np.ndarray] = None       # predicted quantile points
    y: Optional[np.ndarray] = None       # matching target quantile points
    scale: float = 1.0
    shift: float = 0.0

    @classmethod
    def fit(cls, pred, true, kind: str = "quantile", n_points: int = 101) -> "Calibrator":
        p = np.asarray(pred, float); t = np.asarray(true, float)
        ok = np.isfinite(p) & np.isfinite(t)
        p, t = p[ok], t[ok]
        if kind == "none" or p.size < 10:
            return cls(kind="none")
        if kind == "linear":
            sp = p.std() or 1.0
            sc = float(t.std() / sp)
            return cls(kind="linear", scale=sc, shift=float(t.mean() - sc * p.mean()))
        if kind != "quantile":
            raise ValueError(f"calibrate must be none/linear/quantile, got {kind!r}")
        q = np.linspace(0.0, 1.0, n_points)
        x, y = np.quantile(p, q), np.quantile(t, q)
        x = np.maximum.accumulate(x)     # strictly usable by np.interp
        y = np.maximum.accumulate(y)
        # A flat stretch in y is an atom in the training values, and every
        # prediction landing there comes back identical. Worth saying once: the
        # effect is visible in a scatter as a stripe, and silently drags Spearman
        # down without anything looking wrong.
        flat = float(np.mean(np.diff(y) <= 0))
        if flat > 0.15:
            print(f"[calibrate] WARNING {flat:.0%} of the quantile map is flat -- "
                  "the training values are concentrated on a few levels, so that "
                  "share of predictions will collapse onto identical values. Rank "
                  "metrics should use the uncalibrated prediction; consider "
                  "target_opts.calibrate: linear for this assay.")
        return cls(kind="quantile", x=x, y=y)

    def apply(self, p) -> np.ndarray:
        p = np.asarray(p, float)
        if self.kind == "none":
            return p
        if self.kind == "linear":
            return self.scale * p + self.shift
        out = np.interp(p, self.x, self.y)
        # linear extrapolation beyond the fitted range, so a molecule more extreme
        # than anything seen in training is not silently clipped to the maximum
        lo, hi = self.x[0], self.x[-1]
        if self.x[1] > lo:
            s = (self.y[1] - self.y[0]) / (self.x[1] - lo)
            out = np.where(p < lo, self.y[0] + s * (p - lo), out)
        if hi > self.x[-2]:
            s = (self.y[-1] - self.y[-2]) / (hi - self.x[-2])
            out = np.where(p > hi, self.y[-1] + s * (p - hi), out)
        return out

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("x", "y"):
            d[k] = None if d[k] is None else np.asarray(d[k]).tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        d = dict(d)
        for k in ("x", "y"):
            if d.get(k) is not None:
                d[k] = np.asarray(d[k], dtype=float)
        return cls(**d)


# --------------------------------------------------------------------------- #
# feature standardisation
# --------------------------------------------------------------------------- #
def fit_feature_stats(X: np.ndarray, mask: np.ndarray,
                      what: str = "features") -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/sd over VALID residues only. X [N,L,F], mask [N,L].

    An all-false mask is fatal rather than a warning. numpy would return NaN for
    every channel, `apply_feature_stats` would then produce NaN everywhere and
    immediately overwrite it with 0 (because the same all-false mask selects the
    whole tensor) -- so the stream arrives at the model as a block of zeros. The
    model trains, converges, and reports a plausible number produced by the OTHER
    stream alone. Nothing downstream can tell that apart from a real two-stream
    result, which is why this stops here.
    """
    m = mask.astype(bool)
    if not m.any():
        raise ValueError(
            f"every residue is masked out in the {what} array for these "
            f"{len(X)} molecule(s), so the {what} stream would reach the model "
            "as zeros and the run would silently become single-stream.\n"
            "  Usual causes: the cache was built for a different molecule set, "
            "or (for the embedding stream) the language model returned no rows "
            "-- check the '[arrays] dropped ... molecule(s)' line and the "
            "token-offset report from the embedding build.")
    flat = X[m]                                   # [n_valid_residues, F]
    mean = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return mean.astype(np.float32), sd.astype(np.float32)


def apply_feature_stats(X: np.ndarray, mask: np.ndarray,
                        mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    Xs = (X - mean) / sd
    Xs[~mask.astype(bool)] = 0.0
    return Xs.astype(np.float32)


# --------------------------------------------------------------------------- #
# clustering + folds
# --------------------------------------------------------------------------- #
def _kmers(seq: str, k: int, min_len: int = 0) -> frozenset:
    """k-mer set, or empty for sequences below `min_len`.

    An empty set has Jaccard 0 with everything, so a truncated or missing chain
    becomes a forced singleton rather than accidentally matching another stub.
    """
    s = clean_seq(seq)
    if len(s) < max(k, min_len):
        return frozenset()
    return frozenset(s[i:i + k] for i in range(len(s) - k + 1))


def _jaccard(a, b) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def _greedy_chain(seqs: Sequence[str], threshold: float, k: int,
                  min_len: int) -> np.ndarray:
    """CD-HIT-like greedy clustering of ONE chain. Mirrors
    Scripts.Splitter.greedy_cdhit_like_cluster: longest sequences first, each
    becomes a representative unless it matches an existing one."""
    ks = [_kmers(s, k, min_len) for s in seqs]
    order = sorted(range(len(seqs)), key=lambda i: -len(clean_seq(seqs[i])))
    reps: List[frozenset] = []
    out = np.full(len(seqs), -1, dtype=int)
    for i in order:
        for cid, rk in enumerate(reps):
            if _jaccard(ks[i], rk) >= threshold:
                out[i] = cid
                break
        else:
            reps.append(ks[i])
            out[i] = len(reps) - 1
    return out


def cluster_greedy(vh: Sequence[str], vl: Sequence[str], threshold: float = 0.8,
                   k: int = 4, min_len_vh: int = 90, min_len_vl: int = 80) -> np.ndarray:
    """The notebook default (GRAPH_CLUST = False).

    VH and VL are clustered separately and the label is the PAIR (H_id, L_id), so
    two molecules share a cluster only when BOTH chains landed on the same
    representative.

    WEAKER LEAKAGE GUARANTEE THAN `graph`. Requiring both chains to
    match makes the partition FINER, not safer: two antibodies with an identical
    VH but different VL get different labels and can be split across folds. An
    identical heavy chain across train and test is a large leak, so this mode does
    NOT guarantee that molecules in different clusters are dissimilar. It is
    provided to reproduce the notebook's GRAPH_CLUST = False runs; prefer `graph`
    for a leakage-safe estimate.
    """
    h = _greedy_chain(vh, threshold, k, min_len_vh)
    l = _greedy_chain(vl, threshold, k, min_len_vl)
    return _relabel([f"({a},{b})" for a, b in zip(h, l)])


def cluster_graph(vh: Sequence[str], vl: Sequence[str], threshold: float = 0.8,
                  k: int = 4, min_len_vh: int = 90, min_len_vl: int = 80) -> np.ndarray:
    """Connected components of "Jaccard(VH) >= t OR Jaccard(VL) >= t"
    (GRAPH_CLUST = True; Scripts.similarity_cluster with chain_rule="either").

    Blocked k-mer intersection via a sparse membership matrix, so this is O(n^2)
    in memory only in the number of EDGES, not in n^2 python-level comparisons.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(vh)
    src, dst = [], []
    for seqs, min_len in ((vh, min_len_vh), (vl, min_len_vl)):
        M, sizes = _kmer_matrix(seqs, k, min_len)
        inter = (M @ M.T).toarray()
        union = sizes[:, None] + sizes[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            J = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        i, j = np.where(np.triu(J, 1) >= threshold)
        src.append(i); dst.append(j)

    src = np.concatenate(src) if src else np.zeros(0, int)
    dst = np.concatenate(dst) if dst else np.zeros(0, int)
    adj = csr_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    return _relabel(lab)


def _kmer_matrix(seqs: Sequence[str], k: int, min_len: int):
    from scipy.sparse import csr_matrix
    vocab: Dict[str, int] = {}
    rows, cols, sizes = [], [], np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        ks = _kmers(s, k, min_len)
        sizes[i] = len(ks)
        for km in ks:
            rows.append(i)
            cols.append(vocab.setdefault(km, len(vocab)))
    M = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                   shape=(len(seqs), max(len(vocab), 1)))
    return M, sizes


def _cluster_linkage_reference(vh: Sequence[str], vl: Sequence[str],
                               threshold: float = 0.8, k: int = 4,
                               min_len_vh: int = 90, min_len_vl: int = 80) -> np.ndarray:
    """Single-linkage components by brute-force union-find. NOT a config option.

    This computes exactly what `cluster_graph` computes -- single linkage at a
    threshold IS the connected components of the threshold graph, so the two
    partitions are equal by construction, not by coincidence. It was offered as
    `split.cluster: linkage` for a while, which was a choice between two spellings
    of one thing.

    It survives as the TEST ORACLE for `cluster_graph`. That function gets its
    speed from a blocked sparse k-mer intersection -- a sparse matrix product,
    then Jaccard from the intersection and the row sums -- which is the kind of
    code that can be subtly wrong while looking right, and it decides what counts
    as leakage between folds. Checking it against an O(n^2) loop that computes
    each Jaccard directly is cheap insurance on a correctness-critical function.
    """
    n = len(vh)
    kh = [_kmers(s, k, min_len_vh) for s in vh]
    kl = [_kmers(s, k, min_len_vl) for s in vl]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            if _jaccard(kh[i], kh[j]) >= threshold or _jaccard(kl[i], kl[j]) >= threshold:
                parent[rj] = ri
    return _relabel([find(i) for i in range(n)])


def _relabel(labels) -> np.ndarray:
    """Arbitrary hashable labels -> 0..K-1, in order of first appearance."""
    seen: Dict[object, int] = {}
    return np.array([seen.setdefault(x, len(seen)) for x in labels], dtype=int)


CLUSTERERS = {"greedy": cluster_greedy, "graph": cluster_graph}


def cluster_sequences(vh: Sequence[str], vl: Sequence[str], kind: str = "greedy",
                      threshold: float = 0.8, k: int = 4,
                      min_len_vh: int = 90, min_len_vl: int = 80) -> np.ndarray:
    """Dispatch on `split.cluster`. `none` gives singletons."""
    if kind == "none":
        return np.arange(len(vh))
    if kind not in CLUSTERERS:
        raise ValueError(f"unknown split.cluster {kind!r}; "
                         f"allowed: {sorted(CLUSTERERS)} + 'none'")
    return CLUSTERERS[kind](vh, vl, threshold, k, min_len_vh, min_len_vl)


def training_indices(Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(scored, aux_only) row indices, given Y with the target in column 0.

    scored    has a target value: the only molecules a metric can be computed on,
              so the only ones that ever enter a validation fold.
    aux_only  has at least one auxiliary value and no target value. These cannot
              be scored, but the auxiliary heads and the shared encoder can learn
              from them at no cost in target labels -- which is what training
              multi-task is for. Training on them is not leakage: nothing that
              gets scored is in the training set.

    Returning both separately, rather than one combined index, is deliberate: the
    counts differ and a reported `n` that silently included unscored molecules
    would overstate the evidence behind every metric.
    """
    Y = np.asarray(Y)
    has_t = np.isfinite(Y[:, 0])
    has_aux = (np.isfinite(Y[:, 1:]).any(axis=1) if Y.shape[1] > 1
               else np.zeros(len(Y), bool))
    return np.where(has_t)[0], np.where(~has_t & has_aux)[0]


def add_aux_only(rounds, aux_only) -> list:
    """Put the aux-only molecules in every training fold, leaving folds untouched.

    Asserts the invariant that matters: a molecule being scored in a fold is
    absent from that fold's training set. Cheap, and it is the one mistake here
    that would inflate every number without failing anything.
    """
    aux_only = np.asarray(aux_only, dtype=int)
    if not len(aux_only):
        return list(rounds)
    out = []
    for tr, va in rounds:
        tr2 = np.union1d(np.asarray(tr, dtype=int), aux_only)
        if np.intersect1d(tr2, np.asarray(va, dtype=int)).size:
            raise AssertionError(
                "a scored molecule reached the training set -- an aux-only index "
                "overlapped a validation fold, which means the target column was "
                "not what column 0 of Y claims")
        out.append((tr2, np.asarray(va, dtype=int)))
    return out


def make_folds(clust: np.ndarray, n_folds: int = 5, kind: str = "groupkfold",
               seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Returns [(train_idx, val_idx), ...] indexing the array `clust` describes.

    groupkfold -> GroupKFold over shuffled cluster ids. What
                  Scripts.Splitter.split_by_similarity does, so folds match the
                  notebooks for the same clustering and seed.
    balanced   -> whole clusters assigned greedily largest-first, so fold SIZES
                  match as closely as the cluster sizes allow.
    random     -> molecules assigned independently, ignoring clusters entirely.
                  Near-duplicates then span train and test; only for reproducing
                  older runs.
    """
    clust = np.asarray(clust)
    n = len(clust)
    rng = np.random.default_rng(seed)

    if kind == "random":
        fold_of = rng.permutation(n) % n_folds
    elif kind == "groupkfold":
        from sklearn.model_selection import GroupKFold
        uniq = np.unique(clust)
        if len(uniq) < n_folds:
            raise ValueError(
                f"{len(uniq)} cluster(s) but {n_folds} folds requested. Lower "
                "split.n_folds or split.threshold, or use split.folds: random.")
        remap = {old: new for new, old in enumerate(rng.permutation(uniq))}
        groups = np.array([remap[c] for c in clust])
        fold_of = np.empty(n, dtype=int)
        for f, (_, va) in enumerate(GroupKFold(n_splits=n_folds)
                                    .split(np.zeros(n), groups=groups)):
            fold_of[va] = f
    elif kind == "balanced":
        groups, counts = np.unique(clust, return_counts=True)
        if len(groups) < n_folds:
            raise ValueError(
                f"{len(groups)} cluster(s) but {n_folds} folds requested. Lower "
                "split.n_folds or split.threshold, or use split.folds: random.")
        order = groups[np.argsort(-counts)]
        load = np.zeros(n_folds, dtype=int)
        assign = {}
        for g in order:
            f = int(np.argmin(load))
            assign[g] = f
            load[f] += int((clust == g).sum())
        fold_of = np.array([assign[c] for c in clust])
    else:
        raise ValueError(f"unknown split.folds {kind!r}; "
                         "allowed: groupkfold, balanced, random")

    rounds = []
    sizes = [int((fold_of == i).sum()) for i in range(n_folds)]
    for f in range(n_folds):
        va = np.where(fold_of == f)[0]
        tr = np.where(fold_of != f)[0]
        if len(va) == 0:
            raise ValueError(
                f"fold {f} is empty (sizes {sizes}). Too few clusters for this "
                "many folds -- lower split.threshold or split.n_folds.")
        rounds.append((tr, va))
    return rounds


# --------------------------------------------------------------------------- #
# dataset / loaders
# --------------------------------------------------------------------------- #
class FvDataset:
    """One item = (x, mask, x_ab, mask_ab, x_global, y, y_mask).

    Streams that the model does not use are still present but empty, so the batch
    signature never changes and the training loop has no branches.

    Deliberately not a torch.utils.data.Dataset subclass: DataLoader accepts any
    object with __len__ and __getitem__, and avoiding the base class keeps this
    module importable without torch.
    """

    def __init__(self, idx, X=None, mask=None, X_ab=None, mask_ab=None,
                 Xg=None, Z=None):
        self.idx = np.asarray(idx)
        self.X, self.mask = X, mask
        self.X_ab, self.mask_ab = X_ab, mask_ab
        self.Xg, self.Z = Xg, Z

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        import torch
        r = int(self.idx[i])
        z = torch.zeros(1)
        x = torch.from_numpy(self.X[r]).float() if self.X is not None else z
        m = torch.from_numpy(self.mask[r]).float() if self.mask is not None else z
        xa = torch.from_numpy(self.X_ab[r]).float() if self.X_ab is not None else z
        ma = torch.from_numpy(self.mask_ab[r]).float() if self.mask_ab is not None else z
        g = torch.from_numpy(self.Xg[r]).float() if self.Xg is not None else z
        if self.Z is None:
            y = torch.zeros(1)
            ym = torch.zeros(1)
        else:
            row = self.Z[r]
            ym = torch.from_numpy(np.isfinite(row).astype(np.float32))
            y = torch.from_numpy(np.nan_to_num(row, nan=0.0)).float()
        return x, m, xa, ma, g, y, ym


def make_loader(idx, batch_size=32, shuffle=False, **arrays):
    from torch.utils.data import DataLoader
    return DataLoader(FvDataset(idx, **arrays), batch_size=batch_size,
                      shuffle=shuffle, drop_last=False)
