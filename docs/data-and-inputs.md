# Data and inputs

How MADM reads a table, where structures come from, and how the feature
tensor is built. Start from the [README](../README.md) if you have not run
anything yet.

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
