# How a run works

Splitting, seeds, what is trained on, and how the reported numbers are
produced. Start from the [README](../README.md) if you have not run anything
yet.

## Splits — cross-validation during training

**Everything in this section applies to `madm train` only.** It controls how the
training molecules are divided into cross-validation folds, which decides two
things: the score reported in `cv_metrics.csv`, and the epoch budget the final
models inherit. It has no effect on `madm predict` — a trained bundle contains
weights, and predicting for a new antibody does not involve folds.

So this is the setting that determines **what the reported number means**. The
same data and the same architecture will give a Spearman 15–25 % higher under
`folds: random` than under a cluster-based split, because near-duplicate
antibodies then sit in both training and validation. Neither number is wrong;
they answer different questions:

| you want to know | use |
|---|---|
| how well this will work on an antibody unlike anything measured | `cluster: graph`, `folds: groupkfold` |
| how well it works on new variants of families already measured | `cluster: greedy`, `folds: groupkfold` (the notebook default) |
| to reproduce an older run that ignored clusters | `folds: random` — not a generalisation estimate |

Two independent knobs: `cluster` groups molecules, `folds` turns groups into
folds. They mirror the notebook's `GRAPH_CLUST` / `RANDOM_FOLD` switches, and the
pair is recorded in the run directory name and in `summary.json`, so a score can
always be traced back to the split that produced it.

`split.cluster` — how molecules are grouped:

| value | what |
|---|---|
| `greedy` | CD-HIT-like greedy representatives on VH and VL separately; the label is the pair `(H_id, L_id)`. `Scripts.Splitter.greedy_cdhit_like_cluster`. |
| `graph` | connected components of "k-mer Jaccard ≥ t on VH **or** VL". `Scripts.similarity_cluster`, `chain_rule="either"`. |
| `none` | every molecule its own cluster. Only valid with `folds: random`. |


`split.folds` — how groups become folds:

| value | what |
|---|---|
| `groupkfold` | GroupKFold over shuffled cluster ids, as `Scripts.Splitter.split_by_similarity`. |
| `balanced` | whole clusters assigned greedily largest-first, so fold sizes match. |
| `random` | molecules assigned independently, ignoring clusters. |

`graph` guarantees one thing — molecules in *different* clusters are below the
threshold on **both** chains — and nothing about within-cluster similarity. That
is the guarantee CV needs, and it is why `graph` is the right choice for a
leakage-safe estimate.

**`greedy` does not give that guarantee.** Its label is the pair `(H_id, L_id)`,
so two antibodies share a cluster only when *both* chains match. Requiring both
makes the partition **finer, not safer**: an antibody pair with an identical heavy
chain and a different light chain gets two different labels and can be split
across folds. Use `greedy` to reproduce the notebook's `GRAPH_CLUST = False` runs,
not to claim a generalisation estimate.

In all three, a chain shorter than `min_len_vh` / `min_len_vl` has an empty k-mer
set and Jaccard 0 against everything, so it becomes a singleton — including
against an identical truncated chain.

`random` guarantees nothing. Near-duplicates span train and test and every model
scores 15–25 % higher. It exists to reproduce older runs; it is not a
generalisation estimate. Chains shorter than `min_len_vh` / `min_len_vl` produce no
k-mers and become forced singletons rather than matching each other by accident.

The remaining split settings:

```yaml
split:
  threshold: 0.8   # k-mer Jaccard above which two chains are the same cluster
  k: 4             # k-mer length
  min_len_vh: 90   # a shorter VH becomes a singleton rather than a spurious match
  min_len_vl: 80
  n_folds: 5
  seed: 0          # shuffles the cluster ids; the folds are otherwise deterministic
```

`threshold` is the one to think about. It is the definition of "too similar to be
a fair test", and it is not a property of the method but of the claim you want to
make: at 0.8 two antibodies differing in a few CDR positions are the same cluster,
at 0.6 whole families collapse together and the held-out folds get harder. The
run prints the cluster structure it produced, which is worth reading before the
scores:

```
[split] cluster=graph folds=groupkfold: 330 clusters over 382 molecules
        (largest 25) at threshold 0.8
```

A largest cluster holding a large share of the data means one fold is mostly that
family — with `folds: groupkfold` the fold sizes then differ substantially, and
`folds: balanced` evens them out without weakening the leakage guarantee. A
cluster count close to the molecule count (here 330 of 382) means the panel is
mostly singletons, so grouping changes little; a count far below it means the
opposite and the split is doing real work.

## Seeds and ensembling

Two seed lists, because they do different jobs:

```yaml
train:
  seeds: [42, 7, 134, 2024, 88]   # cross-validation
  final_seeds: null               # the released models; null -> use `seeds`
```

CV's seeds only have to make the **median best epoch** stable, and five do that.
The final fit's seeds **are** the released ensemble: more of them improve it, and
they shrink the initialisation noise in any comparison between two arms — which
is the noise that decides whether a small difference on a hold-out is real.

Raising `seeds` to 20 would quadruple the cross-validation for no gain. Set
`final_seeds: [0, 1, ..., 19]` instead: CV stays at five, twenty final models are
fitted, and `predict` scores all twenty. `summary.json` records both lists
(`seeds`, `cv_seeds`).

`train.seeds` (5 by default) controls how many models are fitted. Every seed is
trained in cross-validation *and* again on all labelled molecules, and all of them
are saved.

The final models have no validation set — so there is nothing to early-stop on,
and carving one out of the training data would both shrink it and stop on data the
model is fitting. They run for a **fixed number of epochs: the median best epoch
observed in CV**, where a genuine held-out fold existed.

**The warm-start models obey the same rule.** Passing the training rows as a
validation set makes `train_model` select the epoch that best fits the data it is
fitting — the most overfit one — and then copies those weights into the deployed
model. Inside `fit_final` the warm fits run with no validation loader for a fixed
number of epochs inherited from CV.

**That warm budget is per stream.** The two streams converge on quite different
timescales — frozen AbLang embeddings against 3D features — so each warm model
gets the median of *its own* stopping epoch across folds × seeds:

```
fitting 5 final models on 812 labelled molecules (640 with a SMP value,
172 auxiliary-only) for 143 epochs (median best epoch in CV, warm start
feat=41, ablang=178, no early stopping)...
[train] warm-start budgets differ by stream {'feat': 41, 'ablang': 178}
```

A single shared number would be the median *across* streams — with two streams,
the mean of two values that are neither — training one past where it stopped
generalising and stopping the other while it was still improving. Those weights
are the initialisation the fused model builds on, and the final fit has no early
stopping to absorb it. `cv_metrics.csv` carries one `warm_epoch_<stream>` column
each, so the gap is visible rather than inferred.

`n_ep`, by contrast, is **not** per stream: it is the best epoch of the *fused*
model, which has a single stopping curve.

Nothing is carried between runs: the CV that supplies the epoch count runs on the
same data, in the same `madm train` call, immediately before the final fit. So a
new dataset gets its own folds, its own early stopping and its own budget. Two
cases where that budget stops being evidence are now reported rather than assumed:

```
[train] WARNING no valid best epoch came out of CV; falling back to train.epochs=260,
        which is a configured cap rather than a measurement
[train] WARNING 40% of CV runs stopped at the epochs=260 ceiling -- the median is a
        property of the cap, not of convergence. Raise train.epochs.
```

`summary.json` records which of the two applied, as `epochs_from: cv_median` or
`config_fallback`. One approximation remains by construction: the CV models saw
~80 % of the molecules and the final ones see all of them, so at a fixed batch
size the same epoch count is ~25 % more gradient steps. The honest description is
"trained for the number of epochs that was best on held-out folds", not "trained
to convergence".

## What the models train on

A molecule with an auxiliary measurement but **no target value** cannot be
scored — no metric can be computed for it — so it never appears in a validation
fold. It can still be trained on, and it is: the auxiliary heads and the shared
encoder learn from it at no cost in target labels, which is the reason to train
multi-task at all.

```
[table] 578 molecules   SMP: 382 labelled
[table]   aux Ovalbumin: 323 labelled, 240 co-measured with SMP
[split] + 196 molecule(s) with auxiliary labels but no SMP value added to every
        training fold (382 scored, 578 trained on)
```

| index | who | used for |
|---|---|---|
| `scored` | has a target value | folds, all metrics, `n_scored` |
| `aux_only` | auxiliary value(s), no target | training only — in every CV training fold and in the final fit |

This is **not** leakage of the scored molecule: the molecules being scored are
still absent from training entirely, which `add_aux_only` asserts per fold and
refuses to proceed without. What it does mean is that `n_trained` exceeds
`n_scored` in `summary.json`, and every reported metric refers to the smaller
number.

It only has an effect with `--model mt`. With no auxiliaries, an aux-only row
carries nothing the loss can use and `masked_loss` skips it, so the training set
is unchanged — the run prints `no molecule carries an auxiliary label without a
<target> value` when there is nothing to add.

Imputation-style evaluation — keeping a *scored* molecule in the training pool
with only its target column masked, as `MultiTask.ipynb` cell 22 does — is
deliberately **not** implemented here. That measures assay imputation, and mixing
it into the same `spearman` field would make the headline number mean two
different things depending on the run. See `models/README.md` on reading a
released model's score.

### Regime names: the code and the paper

The exposure keys written by `Splitter.py` into the result files and used by
`madm train` as `T/A/S` predate the terminology used in the manuscript. They are
the same three regimes under older names, kept as keys so that released model
cards and existing result files stay readable:

| key in code and YAML | name in the paper | the molecule being scored |
|---|---|---|
| `transductive` (`T`) | **assay imputation** | already measured in ≥1 other assay of the panel; the target cell is the missing entry |
| `analogue` (`A`) | **analogue** | not measured itself, but a close sequence relative in the panel was |
| `strict` (`S`) | **prospective** | no measurements of its own and no measured relative |

Two things the names do *not* mean. `transductive` is used loosely: transduction
proper requires only that the test input be available during training, whereas
here the molecule is present *with labels*, for other assays — the task is
completion of a partly observed label matrix, which is why the paper calls it
imputation. And `strict` does not mean prediction is impossible: the sequence
encoder generalises to molecules unrelated to anything in training, and
prospective scores are the ones in Table 3. What reverts in that regime is the
auxiliary route specifically — multi-task collapses towards single-task,
because the panel holds no information about the molecule beyond its sequence.

No missing value is ever filled in during training. The three names describe what
an evaluation measures, not a step in the pipeline.

Outputs in `runs/<name>/`:

| file | contents |
|---|---|
| `cv_metrics.csv` | one row per fold × seed, incl. `epoch` and one `warm_epoch_<stream>` per stream |
| `oof_predictions.csv` | one row per molecule: `true`, `pred_seed*`, `pred`, `pred_sd`, `pred_calibrated` |
| `bundle.pt` | every seed's weights + transforms + feature provenance, used by `madm predict` |
| `summary.json` | single-model vs seed-averaged headline numbers |

`madm train` prints both a per-seed breakdown and the seed-averaged out-of-fold
score. Report the second one — it is what the saved ensemble actually does. The
per-run numbers are single models and are systematically lower.

The prediction table gives `<target>_pred` (ensemble), `_sd`, `_min`, `_max`,
`_rank`, one `_seed<N>` column per model, and — when `threshold` is set — `_flag`
plus `_agree`, the fraction of seeds voting the same way as the ensemble. Rows come
back **in the input order**, one per input row; `_rank` carries the ordering (1 =
worst by the assay's own convention) so nothing is lost by not sorting.

`_sd` is disagreement *between seeds*, not a confidence interval. It shows how much
of a prediction is initialisation noise, which is useful for triage, but it does
not capture systematic error: a unanimous ensemble can still be unanimously wrong.

## Validating predictions

If the inference table carries the target column, `madm predict` treats it as
ground truth — never as input — and adds scores, a per-source breakdown, and two
figures written next to `--out`:

```
predictions_validation.png    actual vs predicted + ROC, one panel row per source
predictions_calibration.png   actual/predicted spread, and residual vs actual
```

```
=== validation on 121 measured molecules ===
  Spearman 0.784   Pearson 0.812   AUC 0.926   (rank metrics censored at the noise floor)

per source:
  Chen_2024                    n=74    Spearman 0.766   AUC 0.943
  Ginkgo                       n=47    Spearman 0.741   AUC 0.910
  (pooled numbers across sources partly measure the OFFSET between source scales,
   not ranking within either -- read the per-source rows)
```

Three conventions, all stated on the figure rather than left to the reader:

- every number comes from `madm.metrics.score`, the same function the training
  loop uses, so a figure and a printed table cannot disagree about what
  "Spearman" means
- **censoring is a reporting choice, not implied by having a noise floor** — see
  below. The panel says which convention it used, and the sub-detection region is
  shaded on the scatter either way
- per-seed predictions, if present, are drawn as a faint cloud behind each point,
  so initialisation noise is visible rather than hidden in the ensemble mean

### The noise floor: three independent choices

Setting `noise_floor` does not, by itself, do anything. It is a number; three
separate settings decide what is done with it, and **all three are yours to
choose**:

| setting | affects | default | note |
|---|---|---|---|
| `noise_floor` | — | `null` | the value, in original units |
| `censor_loss` | training | `false` | Tobit loss on the target below the floor |
| `report_censored` | reporting | `false` | clip both arrays before rank metrics |
| *(not settable)* | model selection | always on | early stopping always censors |

Model selection is deliberately not exposed: stopping on ordering below the
detection limit is stopping on noise, whatever you do with the loss or the report.

`censor_loss: false` is the default because it reproduces the notebooks. There
`CENSOR_AT_FLOOR` defaults to `False`, and a floor then selects
`Train_loader.tobit_loss`, which censors on `target == 0` — a condition that never
holds after z-scoring, so it silently collapses to MSE. In other words, a floor set
in a notebook with `CENSOR_AT_FLOOR=False` changes nothing about training. Set
`censor_loss: true` for the behaviour the floor actually implies.

### Censored or uncensored reporting

`target_opts.report_censored` decides whether reported rank metrics clip truth and
prediction at `noise_floor` first:

| | reports | argument for it |
|---|---|---|
| `false` *(default)* | original units, uncensored | it is the raw measurement, unmodified |
| `true` | both arrays clipped at the floor | ordering below the detection limit is ordering on noise |

**It affects reporting only.** Model selection always uses the censored metric
(`madm.train.evaluate`), because early-stopping on sub-floor ordering would stop
on noise — the same convention as `Scripts.Train.train_simple._evaluate`.

`cv_metrics.csv` carries **both** `rho` and `rho_c` (and `pearson`/`pearson_c`)
whatever the flag is set to, so a finished run can be re-read under either
convention without retraining. The flag only decides which one is printed, plotted
and put in `summary.json`.

Worth knowing which way your data leans before choosing. If a large share of the
target values sit below the floor — for one of our assays the median measurement
was 0.01 against a floor of 0.05 — then an uncensored Spearman is substantially a
ranking of values the assay cannot distinguish. That is not a reason to avoid
uncensored reporting, but it is a reason to say so alongside the number.

The flag travels in the bundle, so `madm predict` scores a model the way its
training run was reported.

The pooled-versus-per-source caveat matters more than it looks. Two sources
measured on differently calibrated instruments can produce a high pooled
correlation purely from the offset between their value ranges, while the ranking
*within* each is poor. Read the per-source rows first.

## Calibration — predictions on the training scale

A regularised model shrinks: its predictions have less spread than the measured
values. Inverting the z-score restores the right *units* but not the right
*range*, and because `expm1` is convex a mild linear shrinkage in z-space comes
back as a compressed, bent relationship in assay units — a true 22 predicted as 8.

`target_opts.calibrate` fits a monotone map from out-of-fold predictions onto the
training values:

```
              sd     min      max     rho  pearson    auc
truth      10.03   -0.89   105.28
raw pred    7.05   -0.10    81.10   0.907    0.910  0.949
linear     10.03   -1.66   113.87   0.907    0.910  0.949
quantile    9.77   -0.89   105.28   0.907    0.914  0.949
```

Spearman and AUC are **unchanged exactly** — the map is monotone, so no molecule
changes rank. What changes is the numbers: the predicted range now matches the
measured one, so a prediction can be compared against a threshold in assay units,
plotted against measurements, or handed to someone who expects real values.

- `quantile` (default) matches the whole training distribution
- `linear` matches only the mean and sd — simpler, extrapolates further
- `none` leaves predictions as the model emits them

It is fitted on out-of-fold predictions, never on the training fit, otherwise it
learns the model's optimism instead of its shrinkage. Beyond the fitted range it
extrapolates linearly rather than clipping, so a molecule more extreme than
anything in training does not silently collapse onto the maximum.

The prediction table keeps `<target>_raw` alongside `<target>_pred` so you can see
what the model emitted before rescaling. Calibration does not make the ranking
better — it makes the values usable.

## Loss, stopping rule and optimiser

Four settings under `train:` that decide what "best" means. The defaults are what
the notebooks use — changing any of them changes the model, not just the log.

```yaml
train:
  loss: mse                    # mse | smooth_l1
  select_on: mse               # mse | spearman | pearson
  optimizer: adam              # adam | adamw
  aux_weighting: uncertainty   # uncertainty | fixed
```

**`loss`** — the notebooks call `train_simple` with `nonzero_weight=1`, which
collapses `Scripts.Train_loader.weighted_mse_loss` to plain MSE. Smooth L1 is
available but no published number used it, and it changes how much the largest
residuals matter.

**`select_on`** — which validation number picks the epoch, mirroring
`Scripts.Train._selection_score`. Spearman is a poor stopping rule *on its own*:
it is scale-invariant, so a model that shrinks its predictions toward the mean
pays nothing for doing so — which is how the SMP feature stream reached
`sd(pred) = 0.19` against `sd(true) = 1.0` while still scoring `rho = 0.57`. `mse`
penalises shrinkage and bias; `pearson` penalises shrinkage only.

The validation loss is computed on the **target task only**. A multi-task total
would move with how many auxiliaries happened to be measured in the batch, which
is not a property of the model. Rank metrics are computed after clipping both
arrays at `noise_floor`, as `train_simple._evaluate` does: differences below the
assay's detection limit are not measurable, so ranking on them is ranking on noise.

**`optimizer`** — Adam folds weight decay into the gradient, AdamW decouples it.
At `weight_decay: 5e-3` they are not interchangeable.

**`aux_weighting`** — how the auxiliaries are weighted against the target.

`uncertainty` (default) gives each **auxiliary** one learned log-variance
(Kendall, Gal & Cipolla 2018, `madm.model.AuxUncertaintyWeighter`) while the
target stays pinned at `max(len(aux), 1)`. Plain uncertainty weighting *balances*
tasks — the learned weight is inversely proportional to a task's fitted noise, so
a noisy target gets down-weighted and capacity goes to the cleaner auxiliaries,
which is backwards for a target-focused run. Pinning the target means the model
can switch off donors that transfer negatively without the target's weight ever
being at risk. This is what `MultiTask.ipynb` does, and what the published
multi-task numbers used.

`aux_init_weight: 1.0` starts every auxiliary at exactly the `fixed` weighting,
so training can only move away from that baseline deliberately — the same
argument as warm-starting a fused model at the plain average of its two streams.
`0.5` is the plain Kendall form at `s = 0`.

The log-variances go in their own optimiser group with **weight decay off**:
decaying `s` pulls every weight back toward `init_weight`, a hidden prior
fighting the quantity being learned.

Because the target's share of the loss drifts upward as auxiliaries fall, the
learned weights are printed per seed during the final fit and averaged into
`summary.json` as `aux_weights_final`. An auxiliary driven toward zero is a
result, not a diagnostic: it says that donor transfers nothing.

```
  seed 42 fitted   w_feat=0.612
           learned aux weights: {'Ovalbumin': 0.94, 'Insulin': 0.31}
```

`fixed` holds every auxiliary at 1.0 for the whole run, so the weighting is
reproducible from the config alone. Use it for ablation. Both settings are inert
for single-task runs — with no auxiliaries no weighter is constructed, and the
RNG stream is untouched, so a single-task baseline is bit-identical either way.

## Censored targets

Set `target_opts.noise_floor` **and** `target_opts.censor_loss: true` when the
assay cannot distinguish values below some level, and the target loss becomes
censored (Tobit): uncensored points keep the configured `train.loss` (MSE by
default), censored ones get the log-probability of falling past the limit.
`target_opts.censor_side` chooses which end is censored — `floor` for a detection
limit, `ceiling` for an assay that saturates.

`censor_loss` defaults to **false**: a noise floor on its own censors the
reported rank metrics and the per-epoch selection metric, not the loss.

The censored term needs a **scale**, `sigma`, and it is not the floor's position.
`floor_z` is the floor's *distance from the label mean*, so a floor sitting near
the mean drives it toward zero and a sigma derived from it turns the loss into a
near-vertical hinge. Measured on the real assays, that mistake is wrong by 0.2×–17×
in both directions. `target_opts.floor_sigma_frac` (0.5) is the noise SD as a
fraction of the floor's *magnitude* in log space, converted per fold — mirroring
`Scripts.Splitter.floor_sigma_z`.

The result is exactly `Scripts.Train_loader.tobit_loss_floor` at
`nonzero_weight=1`, the value every notebook passes:

```
mean over all elements of [ censored ? -log CDF((floor − pred)/sigma) : loss(pred, y) ]
```

That function's `nonzero_weight=None` branch instead auto-balances the censored and
observed classes from each batch. No notebook uses it, so it is not reproduced.

**`noise_floor` is not inert when you report uncensored.** It still selects the
censored loss for the target and drives the early-stopping metric. Removing it to
"tidy up" a run that reports uncensored numbers therefore changes training, not
just presentation — in the notebooks it switches the target's loss from MSE to
smooth L1 while leaving the single-stream arms on MSE, which quietly makes the
arms non-comparable. Set `report_censored: false` instead and leave the floor
where it is.
