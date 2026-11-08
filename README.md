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

**This file**

1. [Install](#install) - environment, the three conda-only dependencies, `madm doctor`
2. [Model variants](#model-variants) - what `feat`, `ablang`, `2s` and `mt` are
3. [Quick start](#quick-start) - [train your own](#i-train-your-own), [predict or validate](#ii-predict-or-validate-if-you-have-measurements), [use a released model](#iii-predict-with-a-released-model--no-training-needed)
4. [How close are my molecules to the training set?](#how-close-are-my-molecules-to-the-training-set)

**Detail, in `docs/`**

- [Data and inputs](docs/data-and-inputs.md) - the measurement scale and
  `madm.rescale`, what the table reader reports and what it refuses, supplying
  your own structures, generating descriptors without a model, the `features:`
  block and the IMGT frame
- [How a run works](docs/how-a-run-works.md) - splits and clustering, seeds and
  ensembling, which molecules are trained on and which are scored, validation,
  calibration, the loss and stopping rule, censored targets
- [Reference](docs/reference.md) - every configuration setting, the metrics, the
  library API, the package layout
- [Released models](models/README.md) - what is published, what each score means


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
[Using your own structures](docs/data-and-inputs.md#using-your-own-structures):

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
[Re-using features for a parameter sweep](docs/data-and-inputs.md#re-using-features-for-a-parameter-sweep).


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
[Using your own structures](docs/data-and-inputs.md#using-your-own-structures) for the full resolution
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

## How close are my molecules to the training set?

A score is a property of the *relationship* between an evaluation set and a
training set, not of the model alone — which is why cluster CV, a random split
and an external panel disagree, and why none of them is "the" number. Before
trusting a prediction it is worth knowing where your molecules sit on that
scale.

`madm.sketch` answers it with MinHash sketches, so it needs no training
sequences — only a small file of hashes that can be shipped with a model:

```python
from madm.sketch import sketch_many, max_similarity
import pandas as pd

train = pd.read_excel("training_table.xlsx")
mine  = pd.read_csv("new_antibodies.csv")

A = sketch_many(train.VH, train.VL)        # [n, 128] uint64 minima
B = sketch_many(mine.VH,  mine.VL)

nn = max_similarity(B, A)                  # nearest-neighbour Jaccard, per molecule
print(f"median {nn.mean():.2f}   below 0.4: {(nn < 0.4).mean():.0%}")
```

How to read it. A randomly chosen pair of antibodies sits near **0.16**. A
median around **0.65** means close relatives were in training and the model is
interpolating. **0.2–0.5**, which is where the GDPa3 and GDPa5 panels fall, means
it is extrapolating, and the external numbers in the paper are the ones that
apply. Above **0.8** your molecules are inside the training clusters, and any
score you compute against your own labels will be optimistic.

`max_similarity(A, A, exclude_self=True)` gives the same measure for the
training set against itself, which is how the blue distributions in Figs. 3 and
8 of the paper were made — the density of training space, and the baseline your
own molecules should be compared against.

`save()` and `load()` store sketches with the `k` and `n_hashes` they were built
with, and `sketch_like()` rebuilds a query sketch with matching settings;
sketches made with different settings cannot be compared and raise rather than
returning a wrong number. The estimate is a MinHash approximation with a
standard error of about 0.04 at the default 128 hashes.
