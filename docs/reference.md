# Reference

Every configuration setting, the metrics, the library API and the package
layout.

## Configuration

Every setting that affects a result lives in `config.yaml`. There are no globals
anywhere else, and `cfg.hash()` over the file names the run directory — so a
number can always be traced back to the settings that produced it.

**Nothing here has to be set.** Every key has a default, so a working config is
`target:` plus `paths.table:`. Two worked examples sized for a small and a large
assay are in
[Quick start, step 2](../README.md#i-train-your-own) — start from one of those rather than
from this reference.

Two values are **derived, not configured**:

- `tasks = [target] + aux`
- `target_weight = max(len(aux), 1)` — the target's weight in the joint loss. It
  puts the target at half the total **at initialisation**, whatever the auxiliary
  count, and leaves a single-task model unaffected. Configuring it separately is
  how it ends up counting assays that were filtered out and never trained on.
  Under the default `train.aux_weighting: uncertainty` the auxiliary weights are
  then learned, so the target's *share* drifts upward during training as weak
  donors are down-weighted; `fixed` holds the split constant for the whole run.

`model.simple_head` picks the head: `true` (default) is `Dropout → Linear`, which
is what the per-assay configurations in `MultiTask.ipynb` use for SMP, Ovalbumin,
DNA, Insulin and BVP; `false` is `Linear → LayerNorm → GELU → Dropout → Linear`
(HIC, AC-SINS). Both match `Scripts.models_shared._head` exactly, so a warm start
is a weight copy rather than an approximation.

**Four model fields are per stream**, each with a `null` default meaning "follow
the feature-stream value", so a config that omits them behaves as it always did:

| feature stream | AbLang stream | |
|---|---|---|
| `hidden` | `ab_hidden` | trunk width; readout is `2 × width` |
| `head` | `ab_head` | head width (ignored where the head is simple) |
| `simple_head` | `ab_simple_head` | `null` → follow `simple_head` |
| `pool` | `ab_pool` | `null` → follow `pool` |

**The blend weight can be frozen.** `model.init_w` sets where the late-fusion
weight starts (0.5, the plain average) and `model.freeze_w: true` pins it there
rather than learning it. The fused score alone cannot distinguish a blend that
found a useful ratio from one that is merely averaging two correlated
predictions, so a frozen-`w` run is the control for that: if it matches the
learned run, the weight contributed nothing and the gain is ensembling. Both are
hashed into the run directory, and `blend_weights()` reports the learned values.

The notebooks set them independently — attention on the feature trunk with
`Abl_pool` on the AbLang trunk (`meanmax` for SMP, Ovalbumin, DNA, Insulin, BVP;
`attn` for HIC, AC-SINS), and `ab_simple_head=False` throughout — so a single
shared value could not reproduce them in either direction.

**Where structures come from** is two settings, one per direction:

| key | CLI | meaning |
|---|---|---|
| `paths.structures_in` | `--pdb-dir` | a directory of structures **you** supply; `null` (default) predicts everything |
| `paths.structures` | — | where MADM **writes** the ones it predicts |
| `check_pdb_sequence` | `--no-check-pdb-sequence` | verify a supplied `.pdb` is the molecule its row names; `true` by default |

`check_pdb_sequence` is top-level rather than under `features:`, because the
`features:` block is hashed into the feature-cache filename and a validation
switch does not change the tensor. What each molecule's structure actually *was*
is recorded inside the cache instead — see
[Re-running after you add structures](data-and-inputs.md#re-running-after-you-add-structures).

## Metrics

`madm.metrics.score` returns, in original units:

- `rho` — Spearman. The primary number. Invariant to any monotone transform.
- `pearson` — linear agreement. **Not** transform-invariant: on a log-scaled
  target it is dominated by the largest few values and can sit 0.1 below Spearman
  purely from back-transform curvature.
- `auc` — above/below `target_opts.threshold`. `NaN` when only one class is
  present, rather than a misleading 0.5.
- `rho_c`, `pearson_c` — the same after clipping truth and prediction at
  `noise_floor`.

`score` always returns **both** conventions — `rho`/`pearson`/`auc` uncensored and
`rho_c`/`pearson_c`/`auc_c` clipped at `noise_floor` — so nothing downstream has to
recompute to change its mind. `target_opts.report_censored` picks which pair is
printed and plotted; `cv_metrics.csv` keeps both.

Everything is computed in float64. In float32, `np.maximum(pred, floor)` rounds
the clip point differently for predictions and truths and silently perturbs the
ranking of values sitting exactly at the floor.

## Using it as a library

```python
from madm import Config, Ensemble

ens = Ensemble("runs/SMP_feat+ablang_ST_greedy-groupkfold_2abaef6a15/bundle.pt")
ens.streams          # ['feat', 'ablang'] — what inputs it needs
ens.imgt_positions   # the frame it was trained on
ens.feature_params   # the features: block it was built with
ens.predict(arrays)  # original-unit predictions
```

Scoring and figures are usable on their own, against predictions from anywhere:

```python
from madm.metrics import score
from madm.plots import validation_figure, calibration_figure

score(truth, pred, threshold=0.3, higher_is_bad=True, noise_floor=0.1)
validation_figure(truth, pred, "v.png", threshold=0.3, noise_floor=0.1,
                  sources=src, seed_preds=per_seed)     # one row per source + pooled
calibration_figure(truth, pred, "c.png", noise_floor=0.1)
```

`Config`, `madm.table`, `madm.metrics` and `madm.features` import without torch, so
validating a table, scoring predictions or aligning features does not require a
deep-learning install. `madm.plots` needs matplotlib but not torch.

Checking a directory of structures against a table needs neither torch nor
ImmuneBuilder, so it is worth doing before a long run:

```python
from madm.prepare import check_pdb_matches, pdb_chain_sequences

pdb_chain_sequences("1bey.pdb")           # {'H': 'QVQ...', 'L': 'DIQ...'}
check_pdb_matches("1bey.pdb", vh, vl)     # None if it is this molecule, else why not
```

## Layout

```
madm/config.py     dataclasses <- YAML, validation, run hash
madm/table.py      read + report; keeps every supplied sequence
madm/features.py   channel assembly + IMGT frame (pure numpy)
madm/prepare.py    structures (yours, then predicted), 3D features, AbLang
                   embeddings; cached by content hash
madm/features3d.py the 3D feature extractor -- vendored, so the package is
                   self-contained and needs nothing beside it
madm/reference/    the two hydrophobicity/SASA lookup tables (packaged data)
environment.yml    conda environment: everything, right channels, right order
environment-ablang.yml  embeddings only -- pure pip, no compiled packages
pyproject.toml     pip metadata; `[structure]` omits the conda-only packages
madm/data.py       target transform, clustering, folds, training indices, loaders
data/sources.xlsx  every (assay, source): reference, DOI, transform, counts
madm/model.py      encoders + the one model class
madm/train.py      the one training loop + CV
madm/metrics.py    scoring, paired deltas
madm/plots.py      validation + calibration figures (matplotlib, no torch)
madm/predict.py    bundle save/load, provenance checks, inference
madm/cli.py        doctor | models | prepare | train | predict
models/            released models, one .tar per model + README.md
```
