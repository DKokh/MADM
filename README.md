# MADM: Multi-assay Antibody Developability Model



Model for predicting antibody developability from VH/VL sequences across multiple assays 
Trained and validated on eight assays covering hydrophobicity (HIC), heparin binding (HAC), self-association (AC-SINS) and polyreactivity against five reagents (BVP, dsDNA, insulin, ovalbumin and soluble membrane proteins, SMP).

## Published in 
Kokh, DB "Multi-Assay Antibody Developability Prediction with Shared Structure-Language Representations" 

## Licence

**Apache-2.0** — see [LICENSE](LICENSE). This covers the code and the trained
model weights alike: use, modification and redistribution are permitted,
commercial use included.

No training data is redistributed. `data/sources.xlsx` lists every (assay,
source) pair with its reference and DOI; see [NOTICE](NOTICE) for attribution,
including the Ginkgo Datapoints GDPa1 dataset (CC BY 4.0), which must be
carried forward with any redistribution.

### Disclaimer
- This project was developed independently by the author, on their own time and using their own infrastructure. It is not a product of, nor is it affiliated with, any employer.

- Claude (Claude Opus 5, Anthropic) was used to assist with code generation,
debugging and documentation. All scientific decisions, implementation review
and validation are the author's.


# Overview

- The model has two streams, one from the predicted 3D structure and one from a
  sequence language model. Either can be run alone, or both can be combined by
  **late fusion**  (each stream predicts, and one learned weight per assay mixes
  the two predictions)
- Auxiliary assays, where available, add information for training the target
  assay: Ovalbumin ELISA measurements can improve an Insulin model even when the
  two were measured on different (but related) antibodies.
- Structures are predicted with ABodyBuilder2, but you can **supply your own** and
  MADM builds only what is missing.

```
VH/VL table ──► structures ──► 3D features ──► IMGT frame ─┐
            │                                              ├─► train/
            │                                              │  predict                          
            └─►   PLM   embeddings ────────────────────────┘
```
## Contents

**Getting started**

1. [Install](#install) - environment, the three conda-only dependencies, `madm doctor`
2. [Model variants](#model-variants) - what `feat`, `ablang`, `2s` and `mt` are
3. [Quick start](#quick-start)
    - [I. Train your own](#i-train-your-own)
    - [II. Predict or validate](#ii-predict-or-validate-if-you-have-measurements)
    - [III. Predict with a released model](#iii-predict-with-a-released-model--no-training-needed) - no training needed

**Inputs**

4. [Measurement scale](#measurement-scale--madmrescale) - `madm.rescale`, for data pooled from published sources
5. [Input handling](#input-handling--what-is-reported-what-raises-an-error) - what is reported, what raises
6. [Using your own structures](#using-your-own-structures) - [a `pdb` column](#option-a--a-pdb-column-in-the-table) or [a directory](#option-b--a-directory); [descriptors without a model](#descriptors-without-a-model)
7. [When a structure cannot be built](#when-a-structure-cannot-be-built)
8. [Features](#features) - the `features:` block and the [two reference tables](#two-reference-tables-are-required)
9. [The IMGT frame](#the-imgt-frame)

**How a run is put together**

10. [Splits](#splits--cross-validation-during-training) - cross-validation during training
11. [Seeds and ensembling](#seeds-and-ensembling)
12. [What the models train on](#what-the-models-train-on) - and [the regime names](#regime-names-the-code-and-the-paper) used in the code vs the paper

**Reading the output**

13. [Validating predictions](#validating-predictions) - the [noise floor](#the-noise-floor-three-independent-choices) and [censored reporting](#censored-or-uncensored-reporting)
14. [Calibration](#calibration--predictions-on-the-training-scale)

**Training details**

15. [Loss, stopping rule and optimiser](#loss-stopping-rule-and-optimiser)
16. [Censored targets](#censored-targets)

**Reference**

17. [Configuration](#configuration) - reference for every setting (worked examples are in Quick start)
18. [Metrics](#metrics)
19. [Using it as a library](#using-it-as-a-library)
20. [Layout](#layout)



## Install

### The short way

```bash
conda env create -f environment.yml     # from the directory holding pyproject.toml
conda activate madm
madm doctor --config config.yaml
```

`environment.yml`  installs everything below in the right
order and from the right channels, including the four packages pip cannot supply.

 narrowing installation means either a different file or deleting marked lines:

| what you need | how |
|---|---|
| everything | `environment.yml` |
| embeddings only (`--model ablang`) | **`environment-ablang.yml`** — pure pip, no conda channels, no compiled packages |
| structures you *supply* (`pdb` column or `--pdb-dir`) | `environment.yml`, minus the three lines marked `[PREDICT ONLY]` |
| no embeddings (`--model feat`) | `environment.yml`, minus the one line marked `[ABLANG]` |

Note the third row: if you supply your own structures  you don't need `ImmuneBuilder`,
`openmm` and `pdbfixer`, but **still need** `biotite`, `hydride`, `mdtraj`, `propka` and
`anarci`.

The rest of this section explains what those files do and how to do it by hand.

### Step by step

**1. Install into an environment of your own, on Python 3.10–3.12.** Not the system
Python, and not the newest release:

- PyTorch and `ablang2` 

```bash
# conda
conda create -n madm python=3.11 -y && conda activate madm

# or venv
python3.11 -m venv .venv && source .venv/bin/activate && pip install -U pip
```

**2. Then, from the directory holding `pyproject.toml`**
either of the following options (depends on the task):

```bash
pip install -e .              # core: training and inference
pip install -e ".[structure]" # + the 3D feature stack (see below)
pip install -e ".[ablang]"    # + ablang2                 (needed for the ablang stream)
pip install -e ".[plots]"     # + matplotlib              (validation figures)
pip install -e ".[all]"       # everything
```

**3. Four dependencies pip cannot supply**, all for structure generation. Skip this
whole step for `--model ablang`.

```bash
conda install -c conda-forge openmm pdbfixer -y   # ImmuneBuilder's refinement step
conda install -c conda-forge mdtraj -y            # compiled; conda is far more reliable than pip
conda install -c bioconda anarci hmmer -y         # numbering; anarci needs HMMER
pip install -e ".[all]"
```

The full structure stack, and which part of the run each piece serves:

| package | needed by | from | still needed if you supply `.pdb` files? |
|---|---|---|---|
| `ImmuneBuilder` | predicting structures | pip | **no** |
| `openmm`, `pdbfixer` | ImmuneBuilder's refinement step | **conda-forge** — pdbfixer is not on PyPI | **no** |
| `biotite` | PDB parsing, SASA, contacts | pip | yes |
| `hydride` | protonation | pip | yes |
| `mdtraj` | geometry | conda-forge (pip needs a compiler) | yes |
| `propka` | pKa and residue charge | pip | yes |
| `anarci` | IMGT numbering | **bioconda** — needs HMMER | yes |

Read the last column before assuming that `--pdb-dir` or a `pdb` column lets you
skip this step. It removes the *prediction* side only; the bottom five are
imported by the feature extractor at module load, whoever produced the files.
Check them all in one go rather than one failed run per module:

```bash
python -c "import biotite, hydride, mdtraj, propka, anarci; print('all ok')"
```


### Two downloads happen on first use, not at install

ImmuneBuilder fetches its weights from Zenodo the first time `ABodyBuilder2()` is
constructed, and AbLang2 fetches its checkpoint from Hugging Face. So a
successfully installed package can still fail at the first structure:

```
requests.exceptions.HTTPError: 504 Server Error: Gateway Time-out for url:
https://zenodo.org/record/7258553/files/antibody_model_1?download=1
```

Zenodo returns 502/504 under load fairly often. 
The one-liner to fetch them up front:

```bash
python -c "from ImmuneBuilder import ABodyBuilder2; ABodyBuilder2()"
```

Do that once on a new instance. Afterwards the structure step is offline, and
`--pdb-dir` avoids the download entirely.
Note
`--model ablang` - an embedding-only run that does not need structure generation 


## Model variants

There are two switches between models, and the four models inthe paper are their combinations.
parameter `--model` sets both:

| `--model` | `streams` | `aux` | needs |
|---|---|---|---|
| `feat` | `[feat]` | cleared | structures |
| `ablang` | `[ablang]` | cleared | — |
| `2s` | `[feat, ablang]` | cleared | structures |
| `mt` | `[feat, ablang]` | kept | structures |

`streams` chooses the inputs (features or embedding or both) ; `aux` chooses single- or multi-task. Any combination
is valid - including a multi-task model on AbLang alone.

Anything but `mt` **clears `aux`**. Without that, `--model 2s` on a multi-task
config would quietly stay multi-task, and since `cfg.hash()` names the run
directory, the two runs would collide in one folder.

With both streams (features & embedding) the prediction is a **late fusion**:

```
p_a = w_a · feat_head_a(z_feat, z_glob) + (1 − w_a) · ablang_head_a(z_ab)
w_a = sigmoid(w_logit_a)          one learned scalar per assay
```

The streams never mix before the output. A shared head over concatenated features
lowers the training loss by leaning on whichever stream fits more easily.

## Quick start

### I. Train your own

**1. Prepare a table.** CSV or XLSX with `Ab_ID, VH, VL` and one column per assay.
A missing value means *not measured*.

```csv
Ab_ID,VH,VL,SMP,Ovalbumin
Ab001,QVQLVESGGGLVQ...,DIQMTQSPSSLSA...,0.42,0.31
Ab002,EVQLVESGGGLVK...,EIVLTQSPATLSL...,1.87,
```

**Every usable sequence is kept and predicted.** Duplicates are reported, not
removed; a repeated `(VH,VL)` carrying two *different* measurements is printed as
a warning, because the model cannot separate those molecules and the spread is a
floor on achievable error. Repeated `Ab_ID`s are suffixed (`A`, `A#2`) so the
output still joins back row for row. A sequence that failed on structure generation step or embeddeing ( implausibly short, non-standard residues)  is named in a `WARNING` and
skipped. Error raised: a missing column, too few labelled molecules, or *every*
row unusable.

**Optional columns.**     
- `source`, if present, drives the per-source breakdown at
validation time.     
- `pdb` names a structure you already have for that molecule, so
MADM predicts only the rest — see
[Using your own structures](#using-your-own-structures):

```csv
Ab_ID,VH,VL,SMP,pdb
Ab001,QVQLVESGGGLVQ...,DIQMTQSPSSLSA...,0.42,/home/me/pdbs/Chen_2024/Ab001.pdb
Ab002,EVQLVESGGGLVK...,EIVLTQSPATLSL...,1.87,
```

Give a full path  when file names repeat across datasets. An
empty cell means the structure to be generated.

**One table, or one file per assay.** MADM reads a single table, because
multi-task training needs to know which measurements belong to the same molecule
— and that is a property of the whole panel, not of any one assay. If your data
is one file per assay, join them first with the two helpers in `Scripts/`:

```bash
# 1. one row per (VH,VL) per assay, keeping the preferred source's value
python Scripts/unique_tables.py Data --out Data/unique

# 2. outer-join them into one table; the FIRST assay is the target
python Scripts/make_training_table.py Data/unique SMP Ovalbumin Insulin BVP \
       --out data/train_SMP.xlsx
```

**Column names need not be the assay names.** Value columns in this project are
inconsistent — `SMP` and `Ovalbumin` plain, but `Insulin_rescaled`,
`BVP_rescaled`, `AC_SINS_rescaled` — so `load_train` resolves each assay by
trying `<assay>`, `<assay with _ for ->`, and both with `_rescaled`, then prints
what it used:

```
[table] assay -> column: Insulin -> 'Insulin_rescaled', BVP -> 'BVP_rescaled'
```

For a run you intend to publish, name them explicitly instead; a name absent from the table raises ; name is
stored in the run's config, so a reported number can be traced to the column it
came from:

```yaml
columns: {Insulin: Insulin_rescaled, BVP: BVP_rescaled, AC-SINS: AC_SINS_rescaled}
```

Note what those suffixes imply: with the registry defaults the target is a raw
value while some auxiliaries are rescaled. Training is unaffected (every task is
z-scored separately), but it matters when you compare auxiliary correlations
across assays.

**2. Parameters for run are described in `config.yaml`.**  Everything has a default; a working config is three
lines:

```yaml
target: SMP
paths:
  table: data/train.csv    # RELATIVE TO THE WORKING DIRECTORY, not to this file
```

Add only what you actually want to change. The four settings that are likely to be changed
next:

```yaml
aux: [Ovalbumin, Insulin]  # train on other assays too -- makes it multi-task
streams: [feat, ablang]    # 3D features, AbLang2 embeddings, or both
columns: {}                # only where a column is not named after its assay
paths:
  structures_in: null      # a directory of YOUR .pdb files (see "Using your own
                           # structures"); null predicts them all
```

Unknown keys are an **error**

<details>
<summary><b>Tuning for your dataset size — two worked examples</b></summary>



**Small — Ovalbumin, ~200–500 labels**

```yaml
target: Ovalbumin
aux: [SMP, Insulin, BVP]     # borrow strength; this is where `mt` earns its keep
streams: [feat, ablang]
paths: {table: data/train.csv}

model:
  hidden: 16                 # feature trunk, narrow: few labels to support it
  ab_hidden: 64              # unchanged; the embedding's width does not shrink
  head: 16
  ab_head: 32
  dropout: 0.3
  ab_mode: small
  ab_pool: meanmax
  simple_head: true

train:
  epochs: 200
  lr: 1.0e-3
  weight_decay: 5.0e-3
  batch_size: 16             # 300 labels at batch 32 is ~9 updates an epoch
  patience: 30               # small validation folds are noisy; stop later
  seeds: [42, 7, 134, 2024, 88, 11, 77]

split: {cluster: graph, folds: groupkfold, n_folds: 5}
target_opts:
  calibrate: linear          # quantile needs more out-of-fold points
```

**Large — HIC, ~500–1000 labels**

```yaml
target: HIC
aux: []                      # enough labels of its own; try `mt`, but compare
streams: [feat, ablang]
paths: {table: data/train.csv}

model:
  hidden: 48                 # wider trunk: 770 labels support it
  ab_hidden: 64              # same as above — this one does not scale with data
  head: 16
  ab_head: 32
  dropout: 0.25
  ab_mode: conv              # the only assay using the convolutional AbLang block
  ab_pool: attn
  simple_head: false         # Linear -> LayerNorm -> GELU -> Dropout -> Linear
  ab_simple_head: false

train:
  epochs: 300
  lr: 1.0e-3
  weight_decay: 1.0e-3       # lower than the 5e-3 used elsewhere
  batch_size: 32
  patience: 20
  seeds: [42, 7, 134, 2024, 88]

split: {cluster: graph, folds: groupkfold, n_folds: 5}
target_opts:
  calibrate: quantile
```

| setting | small | large | reason |
|---|---|---|---|
| `hidden` | 16 | 48 | feature-trunk width; the capacity the labels can support |
| `ab_hidden` | 64 | 64 | **unchanged.** It projects a frozen embedding of fixed width (480 for AbLang2, 1280 for ESM2), which does not depend on how many labels the assay has |
| `dropout` | 0.3 | 0.25 | fewer labels, more regularisation |
| `weight_decay` | 5e-3 | 1e-3 | HIC is the one assay fitted with the lower value |
| `ab_mode`, `ab_pool` | `small`, `meanmax` | `conv`, `attn` | the sequence stream earns a heavier encoder only where there are labels for it |
| `batch_size` | 16 | 32 | 300 labels at batch 32 is ~9 updates an epoch |
| `patience` | 30 | 20 | small validation folds give a noisy stopping signal |
| `seeds` | 7 | 5 | seed-to-seed spread is largest when data is scarce |
| `calibrate` | `linear` | `quantile` | quantile matching needs enough points to estimate a distribution |
| `aux` | use them | optional | auxiliaries help most when the target has fewest labels |
| `init_w`, `freeze_w` | 0.5, `false` | 0.5, `false` | the late-fusion blend; defaults in both, see below |

**The blend weight, and when to fix it.** A two-stream model predicts
`p = w·p_feat + (1 − w)·p_ab`, with `w` learned per assay. `model.init_w` sets
where it starts (0.5 is the plain average) and `model.freeze_w: true` pins it
there instead of learning it:

```yaml
model:
  init_w: 0.5
  freeze_w: true     # fixed 50/50 average rather than a learned blend
```

`w` is fitted on the training loss, so it reports which stream fits the
*training* data better, not which generalises. Freezing it helps where the
training set contains large groups of near-identical sequences — patent variant
series, affinity-maturation panels: the embedding stream fits those cheaply, `w`
follows it, and the advantage disappears on a held-out cluster. Where the
training molecules are mostly unrelated the learned weight is the better one.
Measured on this corpus, freezing improved AC-SINS (37 % singletons, a
109-molecule largest cluster) and degraded HIC (63 % singletons).

Both settings are hashed into the run directory, so a frozen and a learned run
land in separate directories and cannot be confused. `w_feat` is written per
fold and seed to `cv_metrics.csv` — consistently far from 0.5 is the warning
sign. Inert for single-stream runs, which have no blend weight.

**One thing to change by hand for `--model mt`.** The shared feature trunk serves
the target *and* every auxiliary, so it needs more capacity than the same assay
does on its own. The published multi-task runs widen it to at least 32
(`HID_MTL = max(32, hidden)` in `MultiTask.ipynb`), which affects the three assays
configured narrower than that — Ovalbumin, dsDNA and Hep, all at `hidden: 16`.
MADM uses `model.hidden` exactly as given and applies no such rule, so **set
`hidden: 32` in an `mt` config for those assays**; for HIC (48), AC-SINS, SMP,
BVP and Insulin the single-task value already satisfies it and nothing changes.
`ab_hidden`, `head` and `ab_head` are the same in both modes.

**Treat both as starting points.** Only the cross-validated number settles it,
and the run gives three ways to read it: `cv_metrics.csv` (per fold × seed), the
seed-averaged score printed at the end, and `cv_validation.png`. Changing any of
the above reuses the feature cache, so a sweep costs training time only — see
[Re-using features for a parameter sweep](#re-using-features-for-a-parameter-sweep).


</details>

**3. Run.** One command does the whole workflow — structures, features,
embeddings, cross-validation, final fit, bundle:

```bash
madm train --config config.yaml --model 2s
```

`--model` selects one of the four models without editing the config:

| flag | streams | aux | what it is |
|---|---|---|---|
| `--model feat` | `[feat]` | cleared | structure features only |
| `--model ablang` | `[ablang]` | cleared | frozen AbLang2 embeddings only |
| `--model 2s` | both | cleared | two-stream late fusion, single task |
| `--model mt` | both | kept | ...plus the auxiliary assays |

Anything but `mt` also **clears `aux`** — otherwise `--model 2s` on a multi-task
config would quietly stay multi-task and both runs would hash to the same
directory. `--aux Ovalbumin,Insulin`, `--target`, `--table` override the config
too.


If you already have structures, there are two ways to hand them over, and only
the missing ones are predicted.

**1. All in one folder, named `<Ab_ID>.pdb`** — point at the folder:

```bash
madm train --config config.yaml --model 2s --pdb-dir data/my_pdbs
```

**2. Names that repeat across datasets** (the same `Ab_1` in several) — add a
`pdb` column to the table holding the full path, and pass nothing extra:

```
Ab_ID   VH     VL     SMP    pdb
Ab_1    QVQ…   DIQ…   0.42   /home/me/pdbs/Chen_2024/Ab_1.pdb
Ab_1    EVQ…   EIV…   1.87   /home/me/pdbs/Jain_2017/Ab_1.pdb
```

```bash
madm train --config config.yaml --model 2s
```

 See also
[Using your own structures](#using-your-own-structures) for the full resolution
order and why every supplied file is checked against its row's sequences first.

**4. What a finished run contains.** `madm train` prints the directory as its
last line: `run -> runs/<target>_<streams>_<ST|MT>_<cluster>-<folds>_<hash>`.

The embedding stream is named for the embedder that produced it, not for the
`ablang` stream key — so an ESM2 run lands in `..._feat+esm2-650m_...`, and
`..._feat+ablang_...` always means AbLang2. A non-default `embedder_layer` is
appended (`esm2-650mL17`), since a mid-stack tensor and the last layer are
different inputs rather than a tuning choice.

```
runs/SMP_feat+ablang_MT_greedy-groupkfold_2abaef6a15/
  bundle.pt             <- the model. This is what `madm predict --run` takes
  config.yaml           what was asked for
  run.log               what happened
  cv_metrics.csv        one row per fold x seed
  oof_predictions.csv   one row per molecule: true, pred_seed*, pred, pred_sd
  cv_validation.png     out-of-fold truth vs prediction, per source + ROC
  cv_calibration.png    predicted vs actual spread, and the residual
  summary.json          single-model vs seed-averaged headline numbers
```

The two figures are the same ones `madm predict` draws, on the **out-of-fold**
predictions: a single number hides a source sitting on its own offset, a handful of points carrying the
correlation, and predictions collapsed toward the mean. The last is invisible to
any rank metric and is what the calibration panel exists to show.

They need matplotlib (`pip install -e ".[plots]"`). Without it the run completes
and says so — everything in the figures is in `oof_predictions.csv`.

**One model per seed is trained (five by default) and all of them saved in
`bundle.pt`.** `predict` loads all five and averages them and contains everything needed to actually apply the models :

| in `bundle.pt` | why it must travel with the weights |
|---|---|
| `arch`, `streams`, `tasks` | rebuilds the network the weights fit |
| `target_transform` | predictions come back in original units, not z-scores |
| `feat_mean`, `feat_sd` | features must be standardised exactly as in training |
| `imgt_positions` | column *j* must mean the residue it meant during training |
| `calibrator` | restores the spread a regularised model shrinks |
| `feature_params` | refuses a feature cache built with different settings |


### II. Predict or validate (if you have measurements)


```bash
madm predict --run runs/SMP_feat+ablang_ST_greedy-groupkfold_2abaef6a15 \
             --table new_antibodies.csv --out predictions.csv
```

`--run` is a **directory** containing `bundle.pt` — either a training run from
step I, or a released model once its archive has been extracted (see step III).
`--table` is the input:

| column | required | meaning |
|---|---|---|
| `VH`, `VL` | **yes** | the only model input |
| `Ab_ID` | no | generated as `Ab00000…` if absent; used to join the output back |
| *the target*, e.g. `SMP` | no | **ground truth**, never input — triggers scoring and figures |
| `source` | no | per-source breakdown in the validation report |
| `pdb` | no | a structure you already have for that row |

the target column is matched by name :

```bash
madm predict --run runs/SMP_... --table val.csv --truth-column PSR --out p.csv
```

this is the column the model is to be judged against. 
When no target column is found at all, `predict`  lists
the other columns it saw, so a measured table silently scored as a blind one .

A table with `VH, VL` gets predictions — **one output row per input row, in the
order supplied**, with a `status` column that is `predicted` or, where no
structure or embedding could be built, `no_structure_or_embedding` with an empty
prediction. If the table *also* carries the target column, that column is treated
as ground truth  and you additionally get scores (on the rows
that have both a measurement and a prediction), a per-source breakdown, and two
figures next to `--out`:

```
predictions_validation.png    truth-vs-prediction and ROC, one row per source
predictions_calibration.png   predicted vs actual spread, and the residual
```

The scatter carries three things : large black-edged circles for the
**ensemble** prediction, a faint grey cloud behind each for the **per-seed**
predictions, and a fitted line with a **95% bootstrap confidence band** alongside
the dashed `y = x`. Keeping the fit and `y = x` both visible is the point — the
gap between them *is* the shrinkage and the offset, and a slope drawn without a
band invites reading a trend off eleven points as if it were established.

A `source` column, if present, drives the per-source rows. 

`--pdb-dir` works here too, and a `pdb` column in the prediction table is read the
same way as in a training table; the output then carries a `pdb_source` column
saying whether each number rests on a structure you supplied or a predicted one.
The bundle's own `structures_in` (training set) is ignored at prediction time.

The bundle carries the architecture, the feature standardisation, the target
transform, the IMGT frame and the feature-generation settings, and refuses a
feature cache that disagrees with any of them.

### III. Predict with a released model — no training needed

`models/` holds released models as **archived run directories**, one `.tar` per
model. Extract one and it is an ordinary run directory, used exactly like the run
you just trained and needing no labels of your own. `--run` takes the extracted
directory, not the archive:

```bash
cd models && tar -xf SMP_2s.tar && cd ..       # -> models/runs/SMP_feat+ablang_ST_.../
ls models/runs                                 # the directory name carries a config hash
madm predict --run models/runs/SMP_feat+ablang_ST_<split>_<hash> \
             --table new_antibodies.csv --out predictions.csv
```

Eight are released: two-stream single-task models for HIC, HAC, AC-SINS,
Ovalbumin and SMP (`_2s`), and multi-task models for Insulin, Ovalbumin and SMP
(`_mt`). [`models/README.md`](models/README.md) lists them with the score and
split each was measured under — read that before quoting a number, since the
splits differ between rows and none of the values is the one in the paper.

| suffix | streams | auxiliaries | needs a structure? |
|---|---|---|---|
| `_feat` | 3D structural features | — | **yes** |
| `_ablang` | frozen AbLang2 embeddings | — | no |
| `_2s` | both, late fusion | — | **yes** |
| `_mt` | both, late fusion | yes | **yes** |


### Report file

A run reports everythind that was printed, so every run keeps a copy, without being asked:

| command | log |
|---|---|
| `madm train` | `<run>/run.log`, beside `config.yaml` |
| `madm predict` | `<out>.log`, beside the predictions |
| any command | `--log PATH` overrides both |

`stderr` goes to the same file and interleaves in order. Logs are **appended**, with a header
naming the time and the exact command, so re-running does not erase the run you
are comparing against.

```
=== 2026-09-13 14:22:05  madm train --config config_SMP.yaml --model mt ===
[table] assay -> column: Insulin -> 'Insulin_rescaled'
[table] WARNING VH: 1 sequence(s) shorter than 90: Ab_0861 (len 74)
[structures] provenance (molecules): {'supplied': 812}
```

The run directory then holds both halves of the record: `config.yaml` is what you
asked for, `run.log` is what happened.

## Measurement scale — `madm.rescale`

**Skip this if your measurements come from one laboratory and you are only
predicting.** It matters when you are assembling a training table from published
sources (e.g. listed in the paper), or validating a released model against numbers from a paper.

### The problem

Developability measurements from independent publications are not on one scale. 
Pool them without correction and the offset between two laboratories looks exactly
like a difference between two antibodies. 
The released models were trained on a harmonised scale. 

### What the module does

For each (assay, source) in the training corpus there is a linear transform,
recovered from molecules recorded in both scales:

```
harmonised = a * value + b
```

```python
from madm.rescale import rescale
df["HIC_h"] = rescale(df.HIC, df.source_HIC, "HIC")
```

```bash
python -m madm.rescale --list-sources --assay HIC
python -m madm.rescale --table mine.xlsx --assay HIC --out mine_h.xlsx
```

Covers `AC-SINS`, `BVP`, `DNA`, `HIC`, `Hep`, `Insulin`, `Ovalbumin`, `SMP`.

**The full table is shipped as `data/sources.xlsx`** — one row per (assay,
source) with the reference, DOI, the transform applied, how many molecules it was
fitted from and how many distinct sequences that source contributed to training.
Three sheets: `sources`, `columns` (what each column means) and `status` (what
each status value means). It is the same content `--list-sources` prints, in a
form you can sort and cite; it is also Table S2 of the paper.

### Preparing your table

Two columns are needed: the measurement, and a source column whose entries match
the published names **exactly**. `--list-sources` prints them with the transform
and the citation:

```
HIC   (10 sources)
   Ginkgo                             n=244 train=246   harmonised = 3.78126 * x -0.544382
                                        Arsiwala et al. 2025 (GDPa1)  10.1080/19420862.2025.2593055
   Jain_2017                          n=135 train=135   harmonised = 1 * x
                                        Jain et al. 2017  10.1073/pnas.1616408114
   absci                              n=13  train=13    harmonised = 1 * x +8.647
                                        Absci 2023 (preprint)  10.1101/2023.01.08.523187
```

**The two counts mean different things.** `n` is how many molecules the transform
was *fitted* from — those recorded on both scales, which is often a small subset.
`train` is how many **distinct sequences** that source contributed to the training
set for this assay, after deduplication. A source can have a large `train` and a
tiny `n` (plenty of data, few anchor molecules), and the reverse happens where a
source was used to fit a rule but later excluded. Quote `train` when describing
dataset composition; `n` only describes the confidence in the conversion.

`train` is absent for a source that never reached training, and absent entirely
from tables generated before October 2026.

Column names are auto-detected (`HIC` + `source_HIC`, `AC_SINS` +
`source_AC_SINS`, and so on); `--value-column` and `--source-column` override.

### SMP and Ovalbumin need no conversion

Both are listed as `identity`: their published values are already on the training
scale. This is a finding, not an omission. On the 118 molecules measured for SMP
by both Ginkgo and another source the raw values agree (median ratio 1.02, mean
difference 0.05), whereas applying a conversion would move Ginkgo's mean from 0.18
to 0.56 against the others' 0.23 — introducing an offset rather than removing one.
The same holds for Ovalbumin.

### If your source is not listed

The values come back as `NaN`, never passed through unchanged, and the run names
the source and suggests a close match if there is one:

```
[rescale] HIC: NO TRANSFORM for 2 source name(s); those values are NaN
            'Jain2017'   did you mean 'Jain_2017' or 'Jain_2024'?
            'MyLab_2026'
```

A raw number silently occupying a harmonised column cannot be detected
downstream, which is the whole reason the module exists. `strict=True` (or
`--strict`) raises instead, for a pipeline that should stop rather than produce a
table with holes in it.

If the name is merely spelled differently, correct it and re-run. **If the data is
genuinely from a new laboratory, no transform exists and none can be invented.**
Deriving one needs molecules measured on *both* scales: pick a handful of
antibodies already in the corpus — clinical-stage antibodies such as those in Jain
et al. 2017 are the usual choice, since most panels include some — measure them
alongside your own, and fit the line through those anchor points. Without anchors
the offset and the biology cannot be separated, by this module or any other
method.

### Reading the status column

Every rescaled value carries one, so you can tell a measured transform from an
assumed one:

| status | meaning |
|---|---|
| `exact` | recovered to machine precision from *n* co-recorded molecules |
| `approx` | affine, but the published values are rounded; expect ~1e-3 error |
| `identity` | the published value is already on the training scale |
| `single_row` | **one molecule only** — the slope was assumed (`a = 1`), not measured |
| `no_transform` | source not recognised; the value is `NaN` |

`single_row` transforms are applied by default but flagged. Pass
`allow=("exact", "approx", "identity")` — or `--no-assumed` — to refuse them.

### Regenerating the coefficients (maintainers)

`madm/_rescale_tables.py` is generated, not hand-edited:

```bash
python Scripts/fit_rescale_transforms.py --data-dir Data \
       --emit MADM/madm/_rescale_tables.py --check-co-measured
```

`--check-co-measured` runs the test that decides which column is the harmonised
one, by comparing sources on molecules they both measured. It also reports
inter-laboratory reproducibility on your own data — the same molecules on two
platforms — which bounds what any model can reach:

| assay | n | rho between sources |
|---|---|---|
| HIC | 131 | 0.98 |
| Hep | 49 | 0.92 |
| AC-SINS | 133 | 0.64 |
| SMP | 118 | 0.60 |
| Ovalbumin | 17 | 0.44 |

The emitter refuses to write if any group comes back `NOT_LINEAR`: a bad
coefficient reaching the package is worse than a failed build.

## Input handling — what is reported, what raises an error

The loader keeps **every sequence you supply** and predicts all of them. Problems
are reported rather than repaired, because which fix is right is a decision about
your data:

| condition | behaviour |
|---|---|
| duplicate `(VH,VL)`, same measured value | `NOTE`, both rows kept and predicted |
| duplicate `(VH,VL)`, **different** values | `WARNING` naming the ids and the spread |
| duplicate `Ab_ID` | `NOTE`, ids suffixed `A`, `A#2`, sequences untouched |
| empty VH or VL, VH < 90, VL < 80, or non-standard residues — **training** | `WARNING` naming the `Ab_ID` and the length, row dropped |
| the same, **prediction** | `WARNING`, row kept, returned with `status = no_structure_or_embedding` |
| *every* row unusable | raises — that is a wrong file, not bad data |
| `pdb` cell names a file that is missing, empty, or the wrong molecule | `WARNING` naming it; the row is **skipped**, not predicted — the cell asserted which structure belongs to it |
| a file found by name in `--pdb-dir` fails the same checks | `WARNING`; the molecule falls through to prediction, since nothing was asserted about it |
| missing required column | raises |
| fewer than 20 labelled target values | raises |

A molecule whose sequence is empty, too short, or not made of the twenty standard
residues cannot be folded or embedded. One such row should not stop a run over
eight hundred others — the same argument as a structure that fails to build — so
it is named and skipped rather than refused. What it must never do is disappear
silently, so every one is listed with its length:

```
[table] WARNING VH: 1 sequence(s) shorter than 90: Ab_0861 (len 74)
[table] dropped 1 of 872 row(s) from the training table: they cannot be folded or
embedded. Check them if that is a surprise -- a length of 0 is usually a blank
spreadsheet row, and a short VH is often a VHH or a truncated entry.
```

Read those warnings. A length of 0 is almost always a trailing row the spreadsheet
kept and can be ignored; a VH of 74 is a real molecule you are now not training
on, and whether to fix the sequence or accept the loss is your decision.

`predict` keeps the row instead of dropping it, because it returns one row per
input row — the molecule comes back unpredicted with a status, so a table you
send in comes back the same length.

The limits live in `table.screen_sequences` (`min_vh=90`, `min_vl=80`) and are not
config-exposed; a VHH panel has no VL at all and needs a different pipeline
rather than a lower threshold.

The conflicting-duplicates warning is the one to read. Two measurements of the
same sequence are a disagreement the model cannot resolve — it sees one input and
two answers — so their spread is a **floor on achievable error**, and worth knowing
before you interpret a Spearman.

```
[table] WARNING 1 duplicate group(s) carry DIFFERENT measured values for the same sequence:
[table]   SMP: ['Ab001', 'Ab117'] span 0.4 .. 0.9
```

Duplicates cost almost nothing to carry: structures, features and embeddings are
all keyed on `sha1(VH|VL)`, so a repeated sequence is built once and reused.

## Using your own structures

By default MADM predicts every structure with ABodyBuilder2. If you already have
some — crystal or cryo-EM models, or predictions from something else — point it at
them and **only the missing ones are built**. There are two ways to do that. Pick
by whether your file names alone identify a molecule.

### Option A — a `pdb` column in the table

One path per row. Use this when file names are **not unique**: the same `Ab_1`
exists in several datasets, filed in a directory per dataset, and only a full path
says which one a row means.

```
Ab_ID   VH     VL     SMP    pdb
Ab_1    QVQ…   DIQ…   0.42   /home/me/pdbs/Chen_2024/Ab_1.pdb
Ab_1    EVQ…   EIV…   1.87   /home/me/pdbs/Jain_2017/Ab_1.pdb
Ab_7    QVH…   DIV…   0.85
```

```bash
madm train --config config.yaml --model mt
```

Nothing else is needed — no `--pdb-dir`, no config change. An empty cell means
*no structure supplied*, so `Ab_7` is predicted as usual. Absolute paths are used
as given; a relative path is resolved from the working directory.

### Option B — a directory

One directory, files named after the molecule. Use this when names **are** unique.

```yaml
paths:
  structures_in: data/my_pdbs      # yours; searched first
  structures: results/pdb          # where MADM writes the ones it predicts
```

```bash
madm train --config config.yaml --model mt --pdb-dir data/my_pdbs
```

`--pdb-dir` overrides `structures_in`; either alone is enough. Two naming forms
are accepted:

| file name | matches |
|---|---|
| `<Ab_ID>.pdb` | `data/my_pdbs/Ab_042.pdb` → the row with `Ab_ID = Ab_042` |
| `<sha1(VH\|VL)[:16]>.pdb` | MADM's own naming, so a `results/pdb` from another machine can be dropped in whole |

The directory is **not** searched recursively, so subdirectories are ignored. That
is deliberate: with names that repeat across datasets, a recursive search would
settle the collision by filesystem order and pair one dataset's geometry with
another's label. If your files are laid out that way, use option A.

### Combining them, and what both share

They are not exclusive. The `pdb` column is checked first, per row, and the
directory is the fallback for rows whose cell is empty — so you can path the
awkward ones explicitly and let the rest resolve by name.

Resolution order per molecule, first usable file wins:

1. the `pdb` column
2. `<pdb-dir>/<Ab_ID>.pdb`
3. `<pdb-dir>/<sha1(VH|VL)[:16]>.pdb`
4. a structure built by an earlier run, in `paths.structures`
5. ABodyBuilder2

`.pdb`, `.ent` and `.pdb1` are all accepted. Chains must be named **H and L** (or
A and B, which the extractor renames); MADM models an Fv, so split a whole IgG
first. Supply structures for none, some or all of the table — the run reports the
mix, and `predictions.csv` carries a `pdb_source` column:

```
[structures] 812/812 available (790 unique Fv)
[structures] provenance (molecules): {'predicted': 640, 'supplied': 172}
```

If **every** structure is supplied, ImmuneBuilder is never imported — so you do
not need it installed at all.

### The sequence check

Every feature is computed from the **structure**. `VH`/`VL` are carried along for
bookkeeping and never enter the tensor. A file under the wrong name therefore
attaches one antibody's geometry to another antibody's measurement, and nothing
downstream would show it: the run just learns something slightly wrong and
reports a slightly worse number. So each supplied file is checked against the
row's VH/VL before it is used:

```
[structures] WARNING 2 supplied structure(s) were REJECTED, e.g.
             [('Ab_042', '1bey.pdb: VH: 61% identity to the structure's H chain
               (119 residues modelled vs 121 in the table)')]
```

The run continues either way — a rejected file from a `pdb` cell is skipped, one
found by name in `--pdb-dir` is predicted instead. Missing density is fine — a disordered CDR loop is still the right molecule; the
threshold is 90% identity, which passes gaps and fails a different antibody. Turn
it off with `check_pdb_sequence: false` or `--no-check-pdb-sequence` only if your
structures are *deliberately* mutated relative to the table, and note that you are
then training on that difference.

**A `pdb` cell that cannot be used is skipped, not predicted around.** The cell is
an assertion that *this* file is the structure for *this* row, so if the path is
wrong, the file is empty, or the sequence check rejects it, MADM warns and moves
on:

```
[structures] WARNING Ab_042: file does not exist -- skipped (/home/me/pdbs/Ab_042.pdb)
```

The molecule leaves with `pdb = ""` and is handled exactly like one whose
structure could not be built. Substituting a predicted structure would hide the
mistake behind a plausible result — and pull in a structure predictor you were not
asking for. Fix the paths, or **blank the cells** for the ones you do want
predicted; an empty cell claims nothing and is predicted as usual.

A file found by *name* in `--pdb-dir` is different: you asserted nothing about it,
so if it fails, that molecule does fall through to prediction.

### Re-running after you add structures

The feature cache is named for the **molecules** and the `features:` settings,
neither of which changes when you add a directory of crystal structures. So the
cache also records which structure each molecule came from, and refuses to serve
itself if that has changed:

```
features_9f3c....npz: this cache was built from different structures.
  172 of 790 molecule(s) differ, e.g. now='supplied:1bey.pdb' cached='predicted'
  Rebuild with `madm prepare --overwrite` (or drop the file).
```

Editing a `.pdb` in place, without renaming it, is the one case this cannot see —
use `--overwrite` then.

### Descriptors without a model

The per-residue descriptors are usable on their own — as input to a different
model, or to look at what a structure contributes before any training happens.
Nothing about generating them requires a target, a label or a trained model.

The direct route is the extractor itself, which takes paths and returns the raw
per-descriptor dictionary:

```python
from pathlib import Path
from madm.features3d import extract_features_batch, combine_features
import madm

ref = Path(madm.__file__).parent / "reference"      # the two lookup tables

feats = extract_features_batch(
    pdb_paths=["my_pdbs/Ab_1.pdb", "my_pdbs/Ab_2.pdb"],
    Ab=["Ab_1", "Ab_2"],                            # identifiers, for bookkeeping
    vh=[vh1, vh2], vl=[vl1, vl2],                   # checked against the files
    source_list=["mine", "mine"],
    DIR_DATA_PSR=str(ref),
    max_len=290, k_neighbors=6)        # 6, not the function's own default of 10

feats["sasa"]            # [N, L, 1]   one entry per descriptor group
feats["hydro_patch_sizes"]
combine_features(feats)  # [N, L, F]   everything concatenated, model input order
```

`feats` is keyed by descriptor group — angles, SASA, neighbours, hydrophobic and
charge sums, hydrogen bonds, contacts, aromatics, quasi-dipole, patch sizes, CDR
flags — which is what you want if you intend to use some and not others.
`combine_features` concatenates them in the order the model expects; the channel
layout is listed under [Features](#features).

**The library route**, if you want the assembled tensor with caching and the
provenance checks a training run gets. This is what `madm train` itself calls, so
the arrays are identical to the ones a model is fitted on — including structure
prediction for any molecule you have no `.pdb` for:

```python
import numpy as np, pandas as pd
from madm.config import Config
from madm.prepare import build_structures, build_features

cfg = Config.load("config.yaml")          # only `features:` and `paths:` are read here
df  = pd.DataFrame({"Ab_ID": ["Ab_1", "Ab_2"],
                    "VH":    [vh1, vh2],
                    "VL":    [vl1, vl2]})

# 1. structures: yours are used where available, the rest are predicted
df = build_structures(df, cfg.paths.structures,
                      pdb_dir=cfg.paths.structures_in,   # None -> predict everything
                      check_sequence=cfg.check_pdb_sequence)

# 2. features: cached on the molecules + the `features:` settings
npz = build_features(df, cfg.paths.features, cfg.reference_tables, cfg.features)

d = np.load(npz, allow_pickle=True)
X, Xg, mask = d["X"], d["Xg"], d["mask"]   # [N,L,66], [N,7], [N,L]
names = [str(s) for s in d["names"]]       # one per channel of X
```

No target column is needed on this path. `df` comes back with a `pdb` column
naming the structure used for each molecule, and an empty string where none could
be built — those rows carry no features, which is why `X`
can be shorter than `df`.

The same thing from the command line, if you have a table already:

```bash
madm prepare --config config.yaml --pdb-dir my_pdbs
```

`madm prepare` loads a *training* table, so it expects the target column to exist
even though nothing is trained; add a column of `NaN` named after `target:` if you
only have sequences and structures. The library route above has no such
requirement.

Both paths need the two reference tables (`Hidrophobisicy_score.xlsx`,
`Residue_max_SASA.xlsx`), which ship inside the package, and neither needs torch.

## When a structure cannot be built

Structure prediction fails on real panels — an unusual residue, a chain ANARCI
will not number, a job killed mid-write. One such antibody must not cost the
batch, and it must never quietly change *which* molecules a result describes:

```
[structures] 97/100 available (100 unique Fv)
[structures] 3 failed, e.g. [('Ab_042', 'numbering failed ...')]
[structures] these molecules carry pdb == '' and are dropped from the feature
             stream; `madm predict` still returns a row for each of them.
[arrays]     dropped 3 molecule(s) with no cached features
```

| situation | behaviour |
|---|---|
| one antibody fails | recorded with its error, batch continues, `pdb = ""` |
| **every** antibody fails | raises — there is nothing to featurise |
| a cached `.pdb` is empty or truncated | rebuilt, not trusted (`exists()` alone would hand a half-written file to the extractor, which then fails with a parsing error that points at the wrong cause) |
| `ImmuneBuilder` not installed | raises with the install command, `--pdb-dir`, and the `--model ablang` alternative — an environment problem, not a data one. Only raised if something actually has to be predicted |
| a supplied `.pdb` is rejected or absent | reported, and the molecule falls through to prediction (see above) |
| `--model ablang` | structures are never built; a structure failure cannot affect the run |

In **training**, unbuildable molecules are dropped with a printed count, so the
model is fitted on a smaller set than the table implies — the count is the thing
to read, because it also changes the fold composition.

In **prediction**, every input row comes back, in the order supplied:

```csv
Ab_ID,VH,VL,SMP_pred,status
Ab001,QVQ...,DIQ...,0.42,predicted
Ab042,EVQ...,EIV...,,no_structure_or_embedding
```

A missing line would silently break a positional join and is indistinguishable
from a lost row once the console has scrolled; an empty prediction with a reason
is not. Validation scores and figures use only rows with both a measurement and a
prediction, and say how many measured molecules were excluded for lack of one.

## Features

`features:` holds the arguments to `madm.features3d.extract_features_batch`.
They belong in the config because they change the **tensor**, not just the runtime:

```yaml
features:
  max_len: 290
  score: WiWh
  k_neighbors: 6                        # -> `neighbors` is [L, 6]
  contact_large: [6.0, 9.0, 12.0, 16.0]
  contact_small: [6.0, 9.0, 12.0]
  contact_selected: [0, 2]
```

The defaults are the values `Generate_Embedding_clean.ipynb` uses. Note
`k_neighbors: 6` — the extractor's own default is 10, which adds four channels and
gives a feature matrix no published number was computed on.

Every setting is hashed into the cache filename **and** stored inside the cache.
Changing one produces a new file rather than silently reusing a tensor of a
different width, `load_arrays` refuses a cache whose settings disagree with the
config, and `predict` refuses a cache whose settings disagree with the bundle.

### Re-using features for a parameter sweep

Feature extraction is the expensive step, and the cache name depends on exactly
two things: **which molecules**, and the **`features:` block**. Nothing about the
model, the training schedule, the splits or the target is in it. So every one of
these reuses the features you already built, with no flag:

```bash
madm train --config c.yaml --model mt                  # different streams
madm train --config c.yaml --model mt --aux Ovalbumin  # different auxiliaries
madm train --config c.yaml --target DNA                # different target
# and any edit to model:, train:, split:, target_opts:
```

Each run is written to its own directory, named after the config hash, so a sweep
never overwrites a previous result.

Two changes **do** force re-extraction, both on purpose:

| change | why |
|---|---|
| anything under `features:` | it changes the tensor, so reusing the old one would be wrong |
| adding or removing table rows | the cache is one `.npz` for the whole set, keyed on the molecules in it |

The second is the one that surprises people. Structures are one file per molecule
and survive an appended table; features are a single array and do not. Add rows
in batches rather than one at a time, and run `madm prepare` once afterwards if
you would rather pay that cost deliberately than at the start of a training run.

The cache holds model input, not the extractor's raw output: `X` [N, L, F] (every
per-residue block concatenated, NaN mapped to 0), `Xg` [N, G] (`pI_and_charge`),
`mask`, the IMGT numbering, and `feature_names` — so the column order a
feature-importance table is indexed by is on disk rather than implied by dict
iteration order.

### Two reference tables are required

Feature generation reads two small lookup tables, **shipped inside the package**
(`madm/reference/`):

| file | holds | used for |
|---|---|---|
| `Hidrophobisicy_score.xlsx` | per-residue hydrophobicity scales, one column per scale | `hydro_sum`, hydrophobic patches, the flatness channels |
| `Residue_max_SASA.xlsx` | maximum SASA of each free residue | turns absolute SASA into *relative* burial: `sasa`, `sasa_sch`, `charge_exp` |

Two scales are read from the first file: the one named by `features.score`
(`WiWh` by default) and `Ja` normalised `TAP`. A `score` value that is not a
column in that file fails inside the extractor, not at config validation, so
check the spelling against the sheet if feature generation dies on the first
molecule.

`paths.reference_tables: null` — the default — uses the packaged copies, so
nothing has to sit beside the install. Set it only to **override** them with your
own scales; it is a separate setting rather than being derived from wherever the
input table sits, because `predict` reads a table from the user's directory and
would otherwise look for the lookup files there.

If either is missing, the install is incomplete rather than the config wrong, and
the error says so:

```
FileNotFoundError: .../madm/reference/Hidrophobisicy_score.xlsx not found. These
ship inside the package, so this usually means an incomplete install (reinstall
with `pip install -e .`) -- or set paths.reference_tables to the directory
holding Hidrophobisicy_score.xlsx and Residue_max_SASA.xlsx.
```

## The IMGT frame

The extractor pads from the N-terminus, so raw column *j* is a different residue in
different antibodies as soon as one has an indel. `model.frame: imgt` (the default)
scatters each molecule onto a fixed frame of IMGT positions instead:

```
raw    [N, 290, F]  column 12 = "the 13th residue of whatever this molecule is"
imgt   [N, P,   F]  column 12 = "H_14", in every molecule and at inference
```

The frame is built from the **training** molecules, stored in the bundle, and
reapplied at inference. Positions absent from a molecule are zero and masked out;
positions absent from the frame are dropped. A molecule whose numbering length
disagrees with its residue mask cannot be placed and is reported and dropped
rather than trained on as an empty tensor.

`frame: raw` reproduces older runs. It is not a neutral choice — the conv stream
sees a different input.

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

## Configuration

Every setting that affects a result lives in `config.yaml`. There are no globals
anywhere else, and `cfg.hash()` over the file names the run directory — so a
number can always be traced back to the settings that produced it.

**Nothing here has to be set.** Every key has a default, so a working config is
`target:` plus `paths.table:`. Two worked examples sized for a small and a large
assay are in
[Quick start, step 2](#i-train-your-own) — start from one of those rather than
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
[Re-running after you add structures](#re-running-after-you-add-structures).

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
