# Released models

Trained models, shipped as archived training runs — so a released model and one
you trained yourself are used identically, because they are the same thing:

```
models/
├── README.md                   this file: what is here and what each score means
├── HIC_2s.tar                  one archived run
├── SMP_2s.tar                  another
└── ...                         one .tar per released model
```

Each archive expands to `runs/<target>_<streams>_<ST|MT>_<split>_<hash>/`
containing everything the training run wrote:

```
bundle.pt           weights (all seeds), architecture, transforms, IMGT frame,
                    feature settings, calibrator  <- this is the model
config.yaml         the exact settings that produced it
summary.json        headline numbers, single model vs seed ensemble
cv_metrics.csv      per fold and seed
oof_predictions.csv per molecule, out of fold
run.log             what happened
```

## Getting one ready to use

```bash
cd models
tar -xf SMP_2s.tar                 # -> runs/SMP_feat+ablang_ST_<split>_<hash>/
ls runs                            # the exact name carries the config hash
madm predict --run runs/SMP_feat+ablang_ST_<split>_<hash> \
             --table new_antibodies.csv --out predictions.csv
```


## What is released so far

Eight models: five single-task two-stream (`_2s`) and three multi-task (`_mt`).
The `split` column says how each score was measured, not how the model was
fitted -- in every case the released weights come from a fit on **all** labelled
molecules, and the split determines only the cross-validated number and the
epoch budget.

| archive | target | auxiliaries | split used for CV | ensemble CV rho | AUC | n scored |
|---|---|---|---|---|---|---|
| `HIC_2s.tar` | HIC | - | cluster (greedy, groupkfold) | 0.762 | 0.916 | 770 |
| `HAC_2s.tar` | Hep / HAC | - | cluster (graph, balanced) | 0.811 | 0.904 | 171 |
| `Ovalbumin_2s.tar` | Ovalbumin | - | cluster (greedy, balanced) | 0.701 | 0.848 | 322 |
| `SMP_2s.tar` | SMP | - | cluster (greedy, groupkfold) | 0.623 | 0.879 | 380 |
| `AC-SINS_2s.tar` | AC-SINS | - | **random** | 0.683 | 0.868 | 620 |
| `Insulin_mt.tar` | Insulin | Ovalbumin, SMP, HIC | **random** | 0.870 | 0.939 | 295 |
| `Ovalbumin_mt.tar` | Ovalbumin | SMP, Insulin | **random** | 0.715 | 0.855 | 322 |
| `SMP_mt.tar` | SMP | Ovalbumin, Insulin, BVP | **random** | 0.650 | 0.880 | 380 |


**Read the split column before comparing rows.** 
Where sequence clusters do not coincide with data source the two
schemes differ by roughly 0.04; where they do, by much more -- the gap then
reflects differences in experimental setup between sources, not only in
sequences.

**These splits may differ from the one reported in the paper**, which uses
`graph` clustering throughout. Three rows here use `greedy`, which labels a
molecule by the pair `(H_id, L_id)` and therefore splits antibodies sharing a
heavy chain but not a light one; the main README explains why that is a finer
partition rather than a safer one.

**MADM does not implement auxiliary-imputation CV.** A validation molecule is
removed from the training pool entirely, together with any auxiliary labels it
carries, so every score above is auxiliary-naive -- unlike the multi-task column
of Fig. 5. See "What the models train on" in the main README.

**`n scored` is not the training set size.** A multi-task run also trains on
molecules that carry an auxiliary label but no target value: `Ovalbumin_mt` is
scored on 322 molecules and trained on 543, `SMP_mt` on 380 and 710, and
`Insulin_mt` on 295 and 1053. `summary.json` in each archive records both.

## Naming

`<Assay>_<model>`, where `<model>` is the `--model` switch:

| suffix | streams | auxiliaries | needs a structure? |
|---|---|---|---|
| `_feat` | 3D structural features | — | **yes** |
| `_ablang` | frozen AbLang2 embeddings | — | no |
| `_2s` | both, late fusion | — | **yes** |
| `_mt` | both, late fusion | yes | **yes** |

`_ablang` is the one to reach for when you cannot run a structure predictor, or
when you want an answer in seconds rather than minutes per antibody.

## Predicting

A CSV or XLSX with `Ab_ID, VH, VL`:

```bash
madm predict --run runs/SMP_feat+ablang_ST_<split>_<hash> \
             --table new_antibodies.csv --out predictions.csv
```

Structures, features and embeddings are built on demand and cached by sequence
hash, so the second call on overlapping molecules is nearly free. The output has
one row per input row, in the order supplied:

```csv
Ab_ID,VH,VL,SMP_pred,SMP_seed42,...,status
Ab001,QVQ...,DIQ...,0.42,0.39,...,predicted
Ab042,EVQ...,EIV...,,,,no_structure_or_embedding
```

`status` is how you spot molecules whose structure could not be built — they come
back with an empty prediction rather than disappearing.

## Validating against your own measurements

Add a column named after the target and the same command scores itself:

```bash
madm predict --run runs/<extracted-model-dir> --table my_panel.csv --out check.csv
```

```
=== validation on 74 measured molecules ===
  Spearman 0.612   Pearson 0.704   AUC 0.871   (original units, uncensored)

per source:
  Chen_2024                    n=52    Spearman 0.633   AUC 0.884
  jain_2017                    n=22    Spearman 0.498   AUC 0.812
```

 `check_validation.png` and `check_calibration.png`: The target column is
treated as ground truth and never as model input. If the file has a `source`
column you also get the per-source rows: a pooled
correlation across sources partly measures the offset between their value scales
rather than the ranking within either.

Only rows with **both** a measurement and a prediction are scored; the count of
measured molecules excluded for lack of a prediction is printed.

## Before trusting a prediction

**The value scale.** Predictions come back on the scale of the training values
for that assay, after the calibrator maps them out of z-space. If your assay
reports on a different scale -- a different protocol or reference standard --
the *ranking* transfers and the numbers do not. Each archive's `config.yaml`
names the table the model was fitted on.

**Similarity to the training set.** A model interpolates where your molecules
resemble what it saw and extrapolates where they do not, and the prediction
alone does not say which. A random antibody pair sits near 0.16 on 4-mer
Jaccard, so a median nearest-neighbour similarity around 0.65 means close
relatives were in training.

**The noise floor.** Where one is set, ordering among values below it is not
measurable, and a Spearman computed across them is partly ranking noise.



