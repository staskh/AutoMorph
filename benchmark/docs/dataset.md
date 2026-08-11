# Dataset

## FIVES

[FIVES](https://doi.org/10.6084/m9.figshare.19688169.v1) is 800 colour fundus photographs at
2048x2048, each with a pixel-level vessel annotation produced by trained graders. Every image
carries two labels the benchmark uses:

* a **disease**: `amd`, `dr` (diabetic retinopathy), `glaucoma`, or `normal` — 200 each;
* a **quality score** of 0-3, where 3 is the cleanest.

It is published as a RAR5 archive under CC BY 4.0.

## The store

The benchmark reads a prepared store rather than the archive:

```
.benchmark_data/fives/
  manifest.csv        one row per image: key, split, disease, quality labels, crop geometry
  original/images/    the native-resolution photographs (2048x2048)
  original/vessel/    the native-resolution expert annotations
  images/             912x912 renditions, unused by this benchmark
  vessel/  fov/       912x912 annotations and field-of-view masks, unused by this benchmark
```

Only `manifest.csv` and `original/` matter here. The benchmark deliberately feeds AutoMorph the
**native-resolution originals**: M0 derives both the retina crop and the microns-per-pixel scale
from the photograph itself, so handing it an already-cropped 912 square would produce a
meaningless scale.

### Building it

```bash
uv run python -m benchmark.fetch_fives --keep-originals   # download 1.6 GB, build the store
uv run python -m benchmark.fetch_fives --from-originals   # reprocess 912 from stored originals
```

The first downloads the figshare archive, extracts it (needs `bsdtar` — macOS ships it; on
Debian/Ubuntu `apt install libarchive-tools`), crops and resizes every image, and deletes the raw
data. `--keep-originals` retains the native-resolution copies, costing ~1.6 GB; **the benchmark
needs them**, because it feeds AutoMorph the originals. An existing store is not rebuilt without
`--force`.

`--from-originals` reprocesses the 912 renditions from the stored originals with no download — the
path to use when the crop or resampling rule changes.

`$AUTOMORPH_BENCHMARK_DATA` overrides where stores live (default `.benchmark_data/`). It is
deliberately separate from `$AUTOMORPH_DATA`, which is where a pipeline *run* writes; reference
datasets outlive any single run.

> **Provenance and verification.** `fetch_fives.py`, `geometry.py`, `paths.py` and
> `datasets/fives.py` came from a separate project (`pheno_automorph`) and originally depended on
> that project's `preprocess` module — a hand port of AutoMorph's own `M0_Preprocess/fundus_prep.py`.
> They now run on AutoMorph's `fundus_prep` directly, through the thin adapter in
> `benchmark/preprocess.py`, so there is only one copy of the crop rule in this repository.
>
> That substitution is verified two ways rather than assumed:
> `benchmark/tests/test_preprocess.py` asserts the adapter's output is **byte-identical** to
> `fundus_prep.process_without_gb`, and rebuilding three images with `--from-originals` reproduces
> the pheno_automorph-built store **byte-for-byte** across `images/`, `vessel/` and `fov/`, with
> every manifest number unchanged.

## The 32 images

`benchmark/selection.py` picks them, subject to three constraints:

| Constraint | Value | Why |
| --- | --- | --- |
| Quality score | exactly 3 | A failure should indict the pipeline, not an unreadable photograph |
| Per disease | 8 each, 32 total | Per-disease numbers stay comparable |
| Split | FIVES `test` only | The held-out split, and every disease can supply 8 at quality 3 |

Selection is **not** sampling. Candidates are sorted by key and the first 8 of each disease are
taken, so a rerun against the same store picks the same 32 images with no random seed to carry
around. The chosen rows are written to `benchmark/results/End2End_original/selection.csv` every run.

Quality-3 images available per disease in the `test` split — every group clears 8 comfortably:

| Disease | Available | Used |
| --- | --- | --- |
| amd | 42 | 8 |
| dr | 36 | 8 |
| glaucoma | 22 | 8 |
| normal | 33 | 8 |

Glaucoma is the tightest group; it is also the disease with by far the most low-quality images in
FIVES overall (31 of the 33 images scored 0 are glaucoma).

## Three images no benchmark should score

FIVES ships three annotations that cannot support a verdict. `selection.UNSCORABLE_KEYS` excludes
them, so no combination of `--split`, `--per-disease` and `--quality-score` can select one:

| Key | Problem |
| --- | --- |
| `train_447_G` | Annotation PNG contains no vessel pixels at all — Dice would be 0 against any non-empty prediction |
| `train_448_G` | Same |
| `train_174_D` | Carries `test_54_D`'s annotation. The two are the same eye photographed twice, so the mask is offset from this image's vessels and scores ~0.15 against a prediction sitting correctly on them |

All three are `train` split, so the recorded run — which draws only from `test` — was never affected.
The guard matters for any reconfigured run.
