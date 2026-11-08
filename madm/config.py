"""Configuration. The single source of truth -- no module-level globals anywhere else.

Every run is identified by `cfg.hash()`, which covers everything that affects the
result. That hash goes into the run directory name and into every output row, so a
number can always be traced back to the settings that produced it.

Feature-generation parameters live here too (`features:`), are written into the
feature cache, and are checked against the cache before training. Taking them from
a function's default arguments is how a model ends up trained on 10-neighbour
features and served 6-neighbour ones.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class _NoDuplicates(yaml.SafeLoader):
    """SafeLoader that refuses a repeated key instead of keeping the last one.

    YAML resolves duplicates before any validation sees the file, so the
    "unknown keys are an error" guard cannot catch them. A second `model:`
    block silently REPLACES the first -- every setting in the earlier one is
    discarded, with no error and nothing in the log to say so. That is the worst
    possible failure mode for a file whose whole purpose is to make a run
    reproducible, and it is invisible in a diff unless you already suspect it.
    """

    def construct_mapping(self, node, deep=False):
        seen = {}
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(
                    f"duplicate key {key!r} in the config, at line "
                    f"{key_node.start_mark.line + 1} and line "
                    f"{seen[key] + 1}. YAML keeps only the LAST one, so "
                    "everything under the first is silently discarded -- merge "
                    "them into a single block.")
            seen[key] = key_node.start_mark.line
        return super().construct_mapping(node, deep)



@dataclass
class Paths:
    table: str = "data/train.csv"          # VH/VL + assay columns
    structures: str = "results/pdb"        # generated .pdb, one per unique Fv
    # Your own structures: crystal/cryo-EM models, or predictions from something
    # other than ABodyBuilder2. Searched BEFORE anything is predicted, so a
    # directory of experimental structures means only the rest are built. Files are
    # matched by `<Ab_ID>.pdb` or `<sha1(VH|VL)[:16]>.pdb`; a `pdb` column in the
    # table overrides both. null -> predict everything.
    structures_in: Optional[str] = None
    features: str = "results/features"     # cached .npz
    runs: str = "runs"                     # trained bundles
    # Directory holding Hidrophobisicy_score.xlsx and Residue_max_SASA.xlsx, read
    # by the feature extractor. null -> the copies shipped inside the package, so
    # a plain `pip install madm` can featurise without any files alongside it.
    # Set it only to override them with your own scales.
    reference_tables: Optional[str] = None


@dataclass
class FeaturesCfg:
    """Arguments to Scripts.Features3D_v1.extract_features_batch.

    Defaults are the values used in Generate_Embedding_clean.ipynb cell 12, NOT
    the function's own defaults -- `k_neighbors` differs (6 here, 10 there) and
    changes the width of the `neighbors` channel block, hence n_feat.
    """
    max_len: int = 290
    score: str = "WiWh"          # hydrophobicity scale passed as `Score`
    k_neighbors: int = 6
    contact_large: List[float] = field(default_factory=lambda: [6.0, 9.0, 12.0, 16.0])
    contact_small: List[float] = field(default_factory=lambda: [6.0, 9.0, 12.0])
    contact_selected: List[int] = field(default_factory=lambda: [0, 2])
    # Seconds allowed for hydride's hydrogen relaxation, per molecule. Placing
    # hydrogens is quick; optimising every rotatable OH / NH3+ / SH orientation
    # is not, and on a structure with a bad contact it can fail to terminate,
    # stalling a whole panel on one molecule. Overrun keeps the ideal-geometry
    # hydrogens and warns. 0 skips relaxation entirely -- faster and bounded, at
    # the cost of less accurate H-bond counts.
    #
    # Part of the cache signature: relaxed and unrelaxed hydrogens are different
    # quantities in the same column, so changing this builds a new cache rather
    # than silently mixing them.
    relax_timeout_s: float = 30.0
    relax_iterations: Optional[int] = None

    def thresholds(self) -> Dict[str, list]:
        return {"large": list(self.contact_large),
                "small": list(self.contact_small),
                "selected": list(self.contact_selected)}

    def signature(self) -> str:
        """Short hash of everything that changes the feature tensor. Goes into the
        cache filename, so features built with different settings cannot collide."""
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True)
                            .encode()).hexdigest()[:8]


@dataclass
class ModelCfg:
    hidden: int = 64          # feature-stream trunk width  -> z_feat is 2*hidden
    ab_hidden: int = 64       # AbLang-stream trunk width   -> z_ab   is 2*ab_hidden
    head: int = 16            # feature head width  (ignored when simple_head)
    ab_head: int = 32         # AbLang head width   (ignored when ab_simple_head)
    dropout: float = 0.25
    ab_mode: str = "small"    # small | large | conv
    # Which frozen language model feeds the second stream. The stream is named
    # `ablang` for what it was first; the encoder reads its width off the cache,
    # so swapping the embedder needs no architecture change. ESM has seen
    # camelid VHHs in UniRef, where AbLang2-paired has not -- which is the case
    # for using it. Each embedder writes its own cache file.
    embedder: str = "ablang2"  # ablang2 | esm2-35m | esm2-150m | esm2-650m
    # Which transformer block to read, for the ESM options. -1 is the last and
    # is usually NOT the best: the final layer is specialised for masked-token
    # prediction, and property prediction transfers better from mid-stack.
    # Roughly half the depth is the usual starting point (17 of 33 for 650M,
    # 15 of 30 for 150M, 6 of 12 for 35M). Ignored by ablang2.
    embedder_layer: int = -1
    pool: str = "attn"        # FEATURE-stream readout: attn | meanmax
    # AbLang-stream readout. None -> follow `pool`, which is how every run before
    # this field behaved. The two streams are pooled separately because the
    # notebooks pool them separately: attention on the feature trunk, the
    # per-assay `Abl_pool` on the AbLang trunk. One shared setting cannot express
    # that, and silently applying the AbLang choice to the feature trunk changes
    # what the feature stream is.
    ab_pool: Optional[str] = None            # None -> follow pool
    use_global: bool = True
  # global features, feature head only
    # True  -> a single Dropout->Linear head.
    # False -> Linear->LayerNorm->GELU->Dropout->Linear, whose width is `head`.
    # `head` therefore has no effect while this is True. Set per assay; see the
    # per-assay configurations in MultiTask.ipynb for the values used.
    simple_head: bool = True
    ab_simple_head: Optional[bool] = None    # None -> follow simple_head
    # Late fusion: p = w*p_feat + (1-w)*p_ab, with w = sigmoid(w_logit) per assay.
    # `init_w` is where that blend starts; 0.5 is the plain average, which is also
    # what a warm-started two-stream model begins at.
    init_w: float = 0.5
    # Pin the blend at `init_w` instead of learning it. Two uses: an ablation that
    # separates "fusion helps" from "a LEARNED blend helps" -- with freeze_w and
    # init_w = 0.5 the model is a fixed average of the two streams, so a fused
    # score that matches the learned one says the weight contributed nothing; and
    # fitting a stream to a fixed share when one stream is known to be unreliable.
    # Inert for single-stream runs, which have no w_logit at all.
    freeze_w: bool = False
    # imgt -> scatter per-residue features onto a fixed IMGT frame built from the
    #         training molecules, so column j is the same position in every
    #         antibody. This is what Model_TwoStream.ipynb trains on and what
    #         infer_twostream reproduces at inference.
    # raw  -> feed the N-terminally padded arrays as they come out of the feature
    #         extractor. Only for reproducing older runs.
    frame: str = "imgt"       # imgt | raw


@dataclass
class TrainCfg:
    epochs: int = 260
    lr: float = 1e-3
    weight_decay: float = 5e-3
    batch_size: int = 32
    patience: int = 20
    # One model is trained per seed, in CV and again on all data. The saved
    # ensemble averages them, which is worth ~0.02 Spearman over a single fit and
    # makes the number reproducible rather than a draw from initialisation noise.
    seeds: List[int] = field(default_factory=lambda: [42, 7, 134, 2024, 88])
    # Seeds for the FINAL models only. None -> use `seeds` for both.
    #
    # The two serve different purposes and are worth separating. CV's seeds only
    # have to make the median best epoch stable, and five do that. The final fit's
    # seeds ARE the released ensemble, and more of them both improve it and shrink
    # the initialisation noise in any comparison between two arms -- which is the
    # noise that decides whether a small difference on a hold-out is real.
    #
    # Raising `seeds` to 20 would also quadruple the CV, for no gain.
    final_seeds: Optional[List[int]] = None
    warm_start: bool = True   # fit each stream alone first, copy weights in
    warm_epochs: int = 200
    # true  -> keep the epoch that scored best on the validation fold (what CV
    #          has always done, and what `epochs`/`patience` are calibrated for)
    # false -> run the full budget and keep the LAST epoch. For experiments on a
    #          small fold, where choosing the epoch on the same twenty molecules
    #          that are then scored contributes more variance than whatever is
    #          being measured.
    select_best: bool = True
    grad_clip: float = 1.0
    # Pointwise loss on uncensored points. The notebooks call train_simple with
    # nonzero_weight=1, which makes Scripts.Train_loader.weighted_mse_loss plain
    # MSE -- so `mse` is the value every published number was trained with.
    loss: str = "mse"                 # mse | smooth_l1
    # Which validation number picks the epoch. The notebooks pass select_on="mse"
    # everywhere. Spearman is scale-invariant, so a model that shrinks its
    # predictions toward the mean pays NO penalty for it: the SMP feature stream
    # reached sd(pred) = 0.19 against sd(true) = 1.0 and still scored rho = 0.57.
    # `mse` penalises shrinkage and bias; `pearson` penalises shrinkage only.
    select_on: str = "mse"            # mse | spearman | pearson
    optimizer: str = "adam"           # adam | adamw
    # HOW THE AUXILIARIES ARE WEIGHTED in the joint loss.
    #   uncertainty  each AUXILIARY gets one learned log-variance (Kendall, Gal &
    #                Cipolla 2018) while the TARGET stays pinned at the derived
    #                max(len(aux), 1), so the model can switch off donors that
    #                transfer negatively without ever putting the target's weight
    #                at risk. This is what MultiTask.ipynb does and what the
    #                published multi-task numbers used, so it is the default.
    #                NOTE the target's SHARE of the loss then drifts upward as
    #                auxiliaries are down-weighted: "the target holds half the
    #                loss" is true only at initialisation. The learned weights are
    #                logged at the end of each fit for exactly that reason.
    #   fixed        every auxiliary at 1.0 for the whole run. The weighting is
    #                then reproducible from the config alone. For ablation.
    # Single-task runs are unaffected either way: with no auxiliaries there is
    # nothing to weight, and no weighter is constructed.
    aux_weighting: str = "uncertainty"   # uncertainty | fixed
    # Effective weight of each auxiliary at initialisation. 1.0 starts a run at
    # exactly the `fixed` baseline, so training can only move away from it
    # deliberately; 0.5 is the plain Kendall form at s_t = 0.
    aux_init_weight: float = 1.0


@dataclass
class TargetCfg:
    log: bool = True               # log1p before z-scoring
    threshold: float = 0.0         # for AUC; in ORIGINAL units
    higher_is_bad: bool = True
    # Value below which the assay cannot distinguish measurements. On its own this
    # is only a NUMBER -- the three switches below decide what is done with it, and
    # all three are the caller's choice rather than implied by setting the floor.
    noise_floor: Optional[float] = None
    # Which SIDE of `noise_floor` is censored. "floor": values at or below it are
    # below detection (the default, and what every assay here used until now).
    # "ceiling": values at or above it are beyond the instrument's range -- a HIC
    # molecule so hydrophobic it never eluted within the gradient is recorded at
    # the maximum, and its true retention time is unknown but larger.
    #
    # The distinction is not cosmetic: a censored point carries "the true value is
    # past this limit", and fitting it as if it were an exact measurement drags
    # the model toward the limit. Censored molecules are also the extreme ones,
    # which is precisely where a developability model is asked to discriminate.
    censor_side: str = "floor"       # floor | ceiling
    # Whether the TARGET LOSS is left-censored (Tobit) below the floor.
    #
    #   True   censored points contribute the log-probability of being below the
    #          floor, not a squared error against a value the assay cannot resolve
    #   False  ordinary `train.loss` on every point
    #
    # False reproduces the notebooks, whose CENSOR_AT_FLOOR defaults to False --
    # there `noise_floor` selects Train_loader.tobit_loss, which censors on
    # `target == 0` and therefore never fires after z-scoring, collapsing to MSE.
    # So a floor set in a notebook with CENSOR_AT_FLOOR=False changes nothing about
    # the loss. Set True here for the behaviour the floor implies.
    censor_loss: bool = False
    # Measurement-noise SD at the floor, as a FRACTION of the floor's magnitude in
    # the space the labels were standardised in. Converted to z-space per fold;
    # see madm.data.floor_sigma_z. A fixed sigma is wrong by 0.2x-17x across the
    # real assays, in both directions.
    floor_sigma_frac: float = 0.5
    # Whether REPORTED rank metrics are clipped at `noise_floor` before scoring.
    #
    #   False (default)  report in original units, uncensored. The number is the
    #                    raw measurement, and includes ordering among values the
    #                    assay cannot actually distinguish.
    #   True             clip truth and prediction at the floor first. Ranking
    #                    below the detection limit is ranking on noise, so this is
    #                    the more conservative reading.
    #
    # This is a REPORTING choice only. Model selection always uses the censored
    # rank metric (see madm.train.evaluate), mirroring Scripts.Train.train_simple:
    # stopping on sub-floor ordering would select on noise. Both `rho` and `rho_c`
    # are written to cv_metrics.csv whatever this is set to, so a run can be
    # re-read under either convention without retraining.
    report_censored: bool = False
    # Rescale predictions onto the scale of the original training values. The map
    # is monotone, so Spearman and AUC are unchanged exactly; it fixes the range,
    # not the ranking. Fitted on out-of-fold predictions.
    calibrate: str = "quantile"           # none | linear | quantile


@dataclass
class SplitCfg:
    """Two independent choices, mirroring the notebook's GRAPH_CLUST / RANDOM_FOLD
    switches but as named values rather than booleans.

    cluster : how molecules are grouped
        greedy   CD-HIT-like greedy representative clustering on VH and VL
                 separately, cluster label = (H_id, L_id). The notebook default
                 (GRAPH_CLUST = False).
        graph    connected components of the k-mer Jaccard graph over whole
                 antibodies, "VH >= t OR VL >= t" (GRAPH_CLUST = True).
        none     every molecule its own cluster; only meaningful with folds=random.

    `linkage` was removed: single linkage at a threshold IS the connected
    components of that threshold graph, so it produced the same partition as
    `graph` -- two spellings of one method, which only made the choice look
    larger than it was.

    folds : how groups become folds
        groupkfold  GroupKFold over shuffled cluster ids, as Splitter.split_by_similarity.
        balanced    whole clusters assigned greedily largest-first, so fold sizes match.
        random      molecules assigned independently, ignoring clusters. Near-duplicates
                    then span train and test and every score rises 15-25%. Only for
                    reproducing older runs (RANDOM_FOLD = True).
    """
    cluster: str = "greedy"      # greedy | graph | none
    folds: str = "groupkfold"    # groupkfold | balanced | random
    threshold: float = 0.8       # k-mer Jaccard for cluster membership
    k: int = 4
    min_len_vh: int = 90         # shorter VH -> forced singleton
    min_len_vl: int = 80
    n_folds: int = 5
    seed: int = 0


@dataclass
class Config:
    target: str = "SMP"
    aux: List[str] = field(default_factory=list)      # empty -> single-task
    streams: List[str] = field(default_factory=lambda: ["feat", "ablang"])
    paths: Paths = field(default_factory=Paths)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    target_opts: TargetCfg = field(default_factory=TargetCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    # {assay -> column name in the table}, for tables whose value columns are not
    # simply the assay name (`Insulin_rescaled`, `AC_SINS_rescaled`, ...). Only
    # needed where the obvious spellings do not match; `load_train` prints the
    # column it used either way, and naming it here records the choice in the run
    # config so a number can be traced to the column it came from.
    columns: Dict[str, str] = field(default_factory=dict)
    # Check that a SUPPLIED .pdb is the molecule the table says it is. Every
    # feature is read off the structure, so a mismatched file attaches one
    # antibody's geometry to another's label -- an error no metric would reveal.
    # Only affects structures you provide; predicted ones are built from VH/VL and
    # cannot disagree. Turn off only if your structures are deliberately mutated
    # relative to the table, and know that you are then training on that.
    check_pdb_sequence: bool = True
    device: str = "cuda"

    # ---- derived; never configured, so it cannot disagree with `aux` --------
    @property
    def tasks(self) -> List[str]:
        return [self.target] + list(self.aux)

    @property
    def target_weight(self) -> float:
        """Target weight in the joint loss = max(len(aux), 1), so the target keeps
        half the total loss whatever the auxiliary count, and a single-task model
        is unaffected. Derived from `aux`, not configured -- configuring it
        separately is how it ends up counting assays that were never used."""
        return float(max(len(self.aux), 1))

    @property
    def ab_simple_head(self) -> bool:
        m = self.model
        return m.simple_head if m.ab_simple_head is None else bool(m.ab_simple_head)

    @property
    def ab_pool(self) -> str:
        """Resolved AbLang readout. Same None-means-follow rule as the head, so a
        config that never mentions it behaves exactly as it did before."""
        m = self.model
        return m.pool if m.ab_pool is None else str(m.ab_pool)

    @property
    def reference_tables(self) -> Path:
        """Directory holding the two hydrophobicity/SASA lookup tables.

        `paths.reference_tables: null` resolves to the copies inside the package,
        which is what makes a plain `pip install madm` able to featurise. Setting
        it points at your own -- an override, not a requirement.
        """
        p = self.paths.reference_tables
        if p:
            return Path(p)
        return Path(__file__).resolve().parent / "reference"

    # ---- io ---------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            # A typo'd filename is the most common first-run error, and the bare
            # FileNotFoundError from read_text() gives the user nothing to act on.
            near = sorted(q.name for q in (p.parent if p.parent.as_posix() != "" else
                                           Path(".")).glob("*.y*ml"))
            raise FileNotFoundError(
                f"{p} not found (looked in {p.parent.resolve()}).\n"
                + (f"YAML files there: {', '.join(near)}" if near else
                   "No .yaml/.yml files in that directory -- is this the right "
                   "working directory?"))
        raw = yaml.load(p.read_text(), Loader=_NoDuplicates) or {}
        sub = {"paths": Paths, "features": FeaturesCfg, "model": ModelCfg,
               "train": TrainCfg, "target_opts": TargetCfg, "split": SplitCfg}
        unknown = [k for k in raw if k not in sub and k not in
                   ("target", "aux", "streams", "device", "columns",
                    "check_pdb_sequence")]
        if unknown:
            raise ValueError(f"{path}: unknown top-level key(s) {unknown}")
        kw = {k: v for k, v in raw.items() if k not in sub}
        for name, klass in sub.items():
            block = raw.get(name) or {}
            bad = [k for k in block if k not in klass.__dataclass_fields__]
            if bad:
                raise ValueError(f"{path}: unknown key(s) {bad} under '{name}'")
            kw[name] = klass(**block)
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))

    def validate(self) -> None:
        # A quoted list in YAML is a STRING, and a string is iterable -- without
        # this check `streams: "[feat, ablang]"` is walked character by character
        # and the error names every letter instead of the mistake.
        examples = {"streams": "[feat, ablang]", "aux": "[Ovalbumin, Insulin]"}
        for field, example in examples.items():
            v = getattr(self, field)
            if isinstance(v, str):
                raise ValueError(
                    f"{field} is a string ({v!r}), not a list -- a quoted YAML "
                    f"list is just text. Remove the quotes:\n  {field}: {example}")
        bad = [s for s in self.streams if s not in ("feat", "ablang")]
        if bad:
            # "both" is the common one: the two-stream model is written as the two
            # names, not as a third value, and saying so here saves a round trip.
            extra = ("\nFor late fusion write both names: streams: [feat, ablang]"
                     if any(str(s).lower() in ("both", "2s", "fusion", "all")
                            for s in bad) else "")
            raise ValueError(f"unknown stream(s) {bad}; allowed: feat, ablang"
                             + extra)
        if not self.streams:
            raise ValueError("streams must contain at least one of: feat, ablang")
        if self.target in self.aux:
            raise ValueError(f"target {self.target!r} also listed in aux")
        if len(set(self.aux)) != len(self.aux):
            raise ValueError(f"duplicate assays in aux: {self.aux}")
        if self.model.hidden % 8 or self.model.ab_hidden % 8:
            raise ValueError("hidden and ab_hidden must be divisible by 8 (GroupNorm)")
        if self.model.ab_mode not in ("small", "large", "conv"):
            raise ValueError(f"model.ab_mode must be small/large/conv, got {self.model.ab_mode!r}")
        from .prepare import EMBEDDERS
        if self.model.embedder not in EMBEDDERS:
            raise ValueError(f"model.embedder must be one of {sorted(EMBEDDERS)}, "
                             f"got {self.model.embedder!r}")
        if self.model.pool not in ("attn", "meanmax"):
            raise ValueError(f"model.pool must be attn or meanmax, got {self.model.pool!r}")
        if self.model.ab_pool not in (None, "attn", "meanmax"):
            raise ValueError("model.ab_pool must be attn, meanmax or null (follow "
                             f"model.pool), got {self.model.ab_pool!r}")
        if self.model.frame not in ("imgt", "raw"):
            raise ValueError(f"model.frame must be imgt or raw, got {self.model.frame!r}")
        if self.split.cluster == "linkage":
            # Not silently aliased to `graph`: the run directory is named after
            # the config, so accepting a retired spelling would put two names on
            # one set of results.
            raise ValueError(
                "split.cluster: linkage was removed -- it computed the same "
                "partition as `graph` (single linkage at a threshold is the "
                "connected components of that graph). Use `graph`; results are "
                "unchanged.")
        if self.split.cluster not in ("greedy", "graph", "none"):
            raise ValueError("split.cluster must be greedy, graph or none, "
                             f"got {self.split.cluster!r}")
        if self.split.folds not in ("groupkfold", "balanced", "random"):
            raise ValueError("split.folds must be groupkfold, balanced or random, "
                             f"got {self.split.folds!r}")
        if self.split.cluster == "none" and self.split.folds != "random":
            raise ValueError("split.cluster: none only makes sense with folds: random -- "
                             "every molecule is its own cluster, so a grouped fold rule "
                             "gives you a random split while claiming otherwise")
        if self.split.n_folds < 2:
            raise ValueError(f"split.n_folds must be >= 2, got {self.split.n_folds}")
        if self.target_opts.censor_side not in ("floor", "ceiling"):
            raise ValueError("target_opts.censor_side must be 'floor' or "
                             f"'ceiling', got {self.target_opts.censor_side!r}")
        if not 0.0 < self.split.threshold <= 1.0:
            raise ValueError(f"split.threshold must be in (0, 1], got {self.split.threshold}")
        if self.train.loss not in ("mse", "smooth_l1"):
            raise ValueError(f"train.loss must be mse or smooth_l1, got {self.train.loss!r}")
        if self.train.select_on not in ("mse", "spearman", "pearson"):
            raise ValueError("train.select_on must be mse, spearman or pearson, "
                             f"got {self.train.select_on!r}")
        if self.train.optimizer not in ("adam", "adamw"):
            raise ValueError(f"train.optimizer must be adam or adamw, got {self.train.optimizer!r}")
        if self.split.n_folds < 2:
            raise ValueError(f"split.n_folds must be >= 2, got {self.split.n_folds}")
        if self.target_opts.censor_side not in ("floor", "ceiling"):
            raise ValueError("target_opts.censor_side must be 'floor' or "
                             f"'ceiling', got {self.target_opts.censor_side!r}")
        if not 0.0 < self.split.threshold <= 1.0:
            raise ValueError(f"split.threshold must be in (0, 1], got {self.split.threshold}")
        if self.train.loss not in ("mse", "smooth_l1"):
            raise ValueError(f"train.loss must be mse or smooth_l1, got {self.train.loss!r}")
        if self.train.select_on not in ("mse", "spearman", "pearson"):
            raise ValueError("train.select_on must be mse, spearman or pearson, "
                             f"got {self.train.select_on!r}")
        if self.train.optimizer not in ("adam", "adamw"):
            raise ValueError(f"train.optimizer must be adam or adamw, got {self.train.optimizer!r}")
        if self.train.final_seeds is not None:
            if not self.train.final_seeds:
                raise ValueError("train.final_seeds is an empty list; omit it (or "
                                 "set null) to use train.seeds")
            if len(set(self.train.final_seeds)) != len(self.train.final_seeds):
                raise ValueError("train.final_seeds contains duplicates: the same "
                                 "seed twice is the same model twice, which makes "
                                 "the ensemble look larger than it is")
        if len(set(self.train.seeds)) != len(self.train.seeds):
            raise ValueError("train.seeds contains duplicates")
        if self.train.aux_weighting not in ("uncertainty", "fixed"):
            raise ValueError("train.aux_weighting must be uncertainty or fixed, "
                             f"got {self.train.aux_weighting!r}")
        if self.train.aux_init_weight <= 0:
            raise ValueError("train.aux_init_weight must be positive, got "
                             f"{self.train.aux_init_weight}")
        if self.target_opts.calibrate not in ("none", "linear", "quantile"):
            raise ValueError("target_opts.calibrate must be none, linear or quantile, "
                             f"got {self.target_opts.calibrate!r}")
        if not 0.0 < self.model.init_w < 1.0:
            raise ValueError(f"model.init_w must be strictly between 0 and 1, "
                             f"got {self.model.init_w}")
        if self.model.freeze_w and len(self.streams) < 2:
            print("[config] NOTE model.freeze_w has no effect: a single-stream "
                  "run has no blend weight to freeze.")
        if self.features.k_neighbors < 1:
            raise ValueError("features.k_neighbors must be >= 1")
        sel, small = self.features.contact_selected, self.features.contact_small
        if any(i < 0 or i >= len(small) for i in sel):
            raise ValueError(f"features.contact_selected {sel} indexes outside "
                             f"contact_small (len {len(small)})")

    def hash(self) -> str:
        return hashlib.sha1(
            json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    def run_dir(self) -> Path:
        d = Path(self.paths.runs) / f"{self.target}_{self.tag()}_{self.hash()}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def fit_seeds(self) -> List[int]:
        """Seeds for the final models: `train.final_seeds`, else `train.seeds`."""
        return list(self.train.final_seeds or self.train.seeds)

    def stream_label(self, stream: str) -> str:
        """Readable name for one stream in a run directory.

        The embedding stream is called `ablang` internally for what it was first,
        but a run on ESM2 named `feat+ablang` is actively misleading: the two
        differ in the representation, not just a setting, and the only thing
        telling them apart would be the config hash. So the stream is named for
        the embedder it actually used, with the layer appended when it is not the
        default -- a mid-stack ESM tensor and the last layer are different inputs.
        """
        if stream != "ablang":
            return stream
        emb = self.model.embedder
        if emb == "ablang2":
            return "ablang"
        lay = self.model.embedder_layer
        return emb if lay == -1 else f"{emb}L{lay}"

    def tag(self) -> str:
        s = "+".join(self.stream_label(x) for x in self.streams)
        return f"{s}_{'MT' if self.aux else 'ST'}_{self.split.cluster}-{self.split.folds}"
