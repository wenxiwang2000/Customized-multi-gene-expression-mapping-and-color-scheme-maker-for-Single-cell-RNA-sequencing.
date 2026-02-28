# Multiple-Gene LOOM Visualizer

A command-line tool for generating publication-ready UMAP/TSNE gene expression maps from `.loom` single-cell datasets.

It produces:

- Individual gene expression maps (3 pages)
- A combined RGB overlay map (1 page)
- Clean, axis-free figures
- Consistent dot sizes across all panels
- Exported multi-page PDF

---

## Features

- Safe `.loom` loading with automatic key detection
- Automatic embedding detection (UMAP / TSNE)
- On-the-fly embedding computation if missing
- Custom per-gene min/max normalization
- Grey background for non-expressing cells
- RGB blended overlay for co-expression
- Scanpy-compatible colorbars
- Publication-ready PDF output

---

## Output Structure (PDF)

Page 1 — Gene 1 (CapaR default)  
Page 2 — Gene 2 (Dh31-R default)  
Page 3 — Gene 3 (Lkr default)  
Page 4 — Combined RGB overlay  

---
## Installation

Requires:

- Python ≥ 3.9
- scanpy
- anndata
- loompy
- matplotlib
- numpy
- scipy

Install dependencies:

```bash
pip install scanpy anndata loompy matplotlib numpy scipy
```

<img width="443" height="215" alt="image" src="https://github.com/user-attachments/assets/358d0345-5b0e-410c-be40-2bbd9551261f" />
