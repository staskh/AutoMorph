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

> **Reproducibility gap.** `benchmark/fetch_fives.py` builds this store, but it imports
> `benchmark.datasets`, `benchmark.geometry`, `benchmark.paths` and `pheno_automorph.preprocess`,
> none of which exist in this repository — it was brought over from another project. The store on
> disk was built elsewhere and cannot currently be rebuilt from here. The benchmark itself does not
> depend on that script; it only reads the store, and fails with a clear message if it is absent.
>
> That the store is nonetheless correct is checkable, and was checked: running the annotation
> through AutoMorph's own M0 crop reproduces the store's recorded crop radius and annotated pixel
> count exactly (see [protocol.md](protocol.md)).

## The 32 images

`benchmark/selection.py` picks them, subject to three constraints:

| Constraint | Value | Why |
| --- | --- | --- |
| Quality score | exactly 3 | A failure should indict the pipeline, not an unreadable photograph |
| Per disease | 8 each, 32 total | Per-disease numbers stay comparable |
| Split | FIVES `test` only | The held-out split, and every disease can supply 8 at quality 3 |

Selection is **not** sampling. Candidates are sorted by key and the first 8 of each disease are
taken, so a rerun against the same store picks the same 32 images with no random seed to carry
around. The chosen rows are written to `benchmark/results/selection.csv` every run.

Quality-3 images available per disease in the `test` split — every group clears 8 comfortably:

| Disease | Available | Used |
| --- | --- | --- |
| amd | 42 | 8 |
| dr | 36 | 8 |
| glaucoma | 22 | 8 |
| normal | 33 | 8 |

Glaucoma is the tightest group; it is also the disease with by far the most low-quality images in
FIVES overall (31 of the 33 images scored 0 are glaucoma).
