# three_gene_maps_same_size_combine.py
# (Only the combined plot logic is changed to match dot size.)

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import anndata as ad
import scanpy as sc
import loompy
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.colors as mcolors

# === Custom colormaps ===
custom_cmap_capar = mcolors.LinearSegmentedColormap.from_list("capar", ["#E6F4F4", "#77BABA"])  # light → strong
custom_cmap_dh31  = mcolors.LinearSegmentedColormap.from_list("dh31",  ["#E6F4F4", "#618E8E"])
custom_cmap_lkr   = mcolors.LinearSegmentedColormap.from_list("lkr",   ["#E6F0FF", "#5A75B6"])


plt.rcParams["axes.grid"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

LIGHT_GREY = "#d3d3d3"

def guess_key(candidates, available):
    for k in candidates:
        if k in available:
            return k
    return None

def get_embedding_from_ca(ca: dict):
    if "Embedding" in ca:
        arr = ca["Embedding"]
        if getattr(arr, "dtype", None) is not None and arr.dtype.names and len(arr.dtype.names) >= 2:
            xname, yname = ("_X","_Y") if "_X" in arr.dtype.names and "_Y" in arr.dtype.names else (arr.dtype.names[0], arr.dtype.names[1])
            return ("umap", np.column_stack([arr[xname], arr[yname]]))
    if "Embeddings_X" in ca and "Embeddings_Y" in ca:
        return ("umap", np.column_stack([ca["Embeddings_X"], ca["Embeddings_Y"]]))
    if all(k in ca for k in ("UMAP_1","UMAP_2")):
        return ("umap", np.column_stack([ca["UMAP_1"], ca["UMAP_2"]]))
    if all(k in ca for k in ("TSNE_1","TSNE_2")):
        return ("tsne", np.column_stack([ca["TSNE_1"], ca["TSNE_2"]]))
    if all(k in ca for k in ("X","Y")):
        return ("umap", np.column_stack([ca["X"], ca["Y"]]))
    if all(k in ca for k in ("_X","_Y")):
        return ("umap", np.column_stack([ca["_X"], ca["_Y"]]))
    return (None, None)

def read_loom_safely(loom_path: Path):
    with loompy.connect(str(loom_path), mode="r", validate=False) as lc:
        n_rows, n_cols = lc.shape
        ra_keys = set(lc.ra.keys())
        var_key = guess_key(
            ["Gene","gene","GeneName","GeneID","GeneNameExtended","genes","feature_name","feature_names","var_names"],
            ra_keys
        )
        if var_key is None:
            var_key = next((k for k in ra_keys if lc.ra[k].dtype.kind in "OUS" and lc.ra[k].shape[0]==n_rows), None)
        var_names = np.array(lc.ra[var_key]).astype(object) if var_key else np.array([f"g{i}" for i in range(n_rows)], dtype=object)

        ca_keys = set(lc.ca.keys())
        obs_key = guess_key(["CellID","cell_id","CellName","Barcode","Cell","obs_names","cell_names"], ca_keys)
        if obs_key is None:
            obs_key = next((k for k in ca_keys if lc.ca[k].dtype.kind in "OUS" and lc.ca[k].shape[0]==n_cols), None)
        obs_names = np.array(lc.ca[obs_key]).astype(object) if obs_key else np.array([f"c{i}" for i in range(n_cols)], dtype=object)

        X = csr_matrix(lc[:, :].T)

        adata = ad.AnnData(X=X)
        adata.var_names = var_names
        adata.obs_names = obs_names

        basis, coords = get_embedding_from_ca(lc.ca)
        if coords is not None:
            if basis == "tsne":
                adata.obsm["X_tsne"] = coords
            else:
                adata.obsm["X_umap"] = coords
    return adata

def ensure_embedding(adata, preferred="umap"):
    has_umap = "X_umap" in adata.obsm
    has_tsne = "X_tsne" in adata.obsm
    if preferred=="umap" and has_umap: return "umap"
    if preferred=="tsne" and has_tsne: return "tsne"
    if has_umap: return "umap"
    if has_tsne: return "tsne"
    sc.pp.normalize_total(adata); sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata._inplace_subset_var(adata.var["highly_variable"])
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    sc.tl.umap(adata)
    return "umap"

def fuzzy_pick(gene_query: str, var_names) -> str:
    if gene_query in var_names: return gene_query
    q = gene_query.lower().replace("-", "").replace("_", "")
    lower_map = {g.lower().replace("-", "").replace("_", ""): g for g in var_names}
    if q in lower_map: return lower_map[q]
    for k, g in lower_map.items():
        if q in k or k in q:
            return g
    raise KeyError(f"Gene '{gene_query}' not found in var_names.")

def vector_for_gene(adata, gene):
    x = adata[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    v = np.asarray(x).ravel().astype(float)
    v[np.isnan(v)] = 0.0
    return v

def clip_and_scale(v, vmin, vmax):
    vv = v.copy()
    vv[vv < vmin] = vmin
    vv[vv > vmax] = vmax
    if vmax > vmin:
        vv = (vv - vmin) / (vmax - vmin)
    else:
        vv = np.zeros_like(vv)
    return vv

def _strip_axes(fig):
    for ax in fig.axes:
        ax.set_axis_off()
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

def single_gene_page(adata, gene, vmin, vmax, cmap, title_text, basis,
                     na_color=LIGHT_GREY, size=10, zero_is_na=False):
    expr = vector_for_gene(adata, gene)
    mask = (expr == 0) if zero_is_na else (expr <= vmin)
    expr_for_plot = expr.copy()
    expr_for_plot[mask] = np.nan
    adata.obs["_expr_plot"] = expr_for_plot

    try:
        if basis == "umap":
            fig = sc.pl.umap(
                adata, color="_expr_plot", cmap=cmap, na_color=na_color,
                frameon=False, size=size, return_fig=True, show=False,
                vmin=vmin, vmax=vmax
            )
        else:
            fig = sc.pl.tsne(
                adata, color="_expr_plot", cmap=cmap, na_color=na_color,
                frameon=False, size=size, return_fig=True, show=False,
                vmin=vmin, vmax=vmax
            )
    except TypeError:
        if basis == "umap":
            sc.pl.umap(adata, color="_expr_plot", cmap=cmap, na_color=na_color,
                       frameon=False, size=size, show=False, vmin=vmin, vmax=vmax)
        else:
            sc.pl.tsne(adata, color="_expr_plot", cmap=cmap, na_color=na_color,
                       frameon=False, size=size, show=False, vmin=vmin, vmax=vmax)
        fig = plt.gcf()

    _strip_axes(fig)
    for ax in fig.axes:
        ax.set_title(title_text, fontsize=12)
    fig.tight_layout()
    ax0 = fig.axes[0]
    xlim, ylim = ax0.get_xlim(), ax0.get_ylim()
    figsize = fig.get_size_inches()
    return fig, (xlim, ylim, figsize)

def dominant_color_page(
    coords, r, g, b, title_text, xlim, ylim, figsize,
    size=None, sizes=None, alpha=1.0, na_color=(0.9, 0.9, 0.9)
):
    """
    Draw a single scatter where each cell is colored by the gene with the
    largest normalized expression (r,g,b already in [0,1]).
    Cells with r=g=b=0 are light grey.
    If `sizes` (array of per-cell sizes) is provided, it is used to exactly
    match the single-gene page. Otherwise `size` is used uniformly.
    """
    import numpy as np
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.axis("off")

    N = coords.shape[0]
    if sizes is None:
        sizes_full = np.full(N, size if size is not None else 1.0, dtype=float)
    else:
        sizes_full = np.array(sizes).astype(float)

    # Base layer: all cells light grey (no expression)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=sizes_full, c=[na_color], edgecolors="none", linewidths=0, alpha=1.0
    )

    # Winner-take-all color (one color per cell)
    arr = np.vstack([r, g, b])             # 3 x N
    max_idx = np.argmax(arr, axis=0)       # which channel wins
    max_val = np.max(arr, axis=0)          # how strong the winner is
    has_any = max_val > 0

    # Plot each group on top, reusing identical dot sizes to page 1
    for idx, color in zip([0, 1, 2], ["red", "green", "blue"]):
        m = has_any & (max_idx == idx)
        if np.any(m):
            ax.scatter(
                coords[m, 0], coords[m, 1],
                s=sizes_full[m],
                c=color, edgecolors="none", linewidths=0, alpha=alpha
            )

    # Keep just the gene names (no extra wording)
    ax.set_title(title_text, fontsize=12, pad=6)
    return fig

# --------- ONLY THIS FUNCTION IS CHANGED to accept "sizes" ----------
def overlay_rgb_page(coords, r, g, b, title_text, size=10, xlim=None, ylim=None,
                     figsize=None, sizes=None):
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
    rgb = np.stack([r, g, b], axis=1)
    is_off = (rgb.sum(axis=1) == 0)
    colors = rgb.copy()
    colors[is_off] = np.array([0.85, 0.85, 0.85])  # light grey for all-zeros

    s = sizes if sizes is not None else size
    ax.scatter(coords[:, 0], coords[:, 1], s=s, c=colors, linewidths=0, edgecolors="none")

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    ax.set_title(title_text, fontsize=12)
    ax.set_axis_off()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    return fig
# -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Create CapaR/Dh31-R/Lkr UMAP/TSNE maps + RGB overlay to a PDF, with grey non-expressors and no axes.")
    p.add_argument("loom", help="Path to .loom")
    p.add_argument("--capar", default="CapaR")
    p.add_argument("--dh31", default="Dh31-R")
    p.add_argument("--lkr",  default="Lkr")
    p.add_argument("--capar_minmax", nargs=2, type=float, default=[0.0, 11.3])
    p.add_argument("--dh31_minmax",  nargs=2, type=float, default=[0.0, 5.9])
    p.add_argument("--lkr_minmax",   nargs=2, type=float, default=[0.0, 6.067])
    p.add_argument("--basis", choices=["auto","umap","tsne"], default="auto")
    p.add_argument("--pointsize", type=float, default=10.0)
    p.add_argument("--out", default="three_gene_maps.pdf")
    p.add_argument("--combined_pointsize", type=float, default=None,
                   help="Override dot size for Page 4 only; if omitted, reuse Page 1 size.")
    p.add_argument("--combined_alpha", type=float, default=1.0,
                   help="Opacity (0–1) for Page 4 colored dots.")

    args = p.parse_args()

    loom_path = Path(args.loom)
    if not loom_path.exists():
        raise SystemExit(f"LOOM not found: {loom_path}")

    adata = read_loom_safely(loom_path)
    adata.var_names_make_unique()

    try:
        g_r = fuzzy_pick(args.capar, adata.var_names)
    except KeyError:
        raise SystemExit(f"Could not find CapaR like '{args.capar}'.")
    try:
        g_g = fuzzy_pick(args.dh31, adata.var_names)
    except KeyError:
        try:
            g_g = fuzzy_pick(args.dh31.replace("-", ""), adata.var_names)
        except KeyError:
            raise SystemExit(f"Could not find Dh31-R like '{args.dh31}'.")
    try:
        g_b = fuzzy_pick(args.lkr, adata.var_names)
    except KeyError:
        raise SystemExit(f"Could not find Lkr like '{args.lkr}'.")

    pref = "umap" if args.basis in ("auto", "umap") else "tsne"
    basis = ensure_embedding(adata, preferred=pref)
    coords = adata.obsm["X_umap"] if basis=="umap" else adata.obsm["X_tsne"]

    v_r = vector_for_gene(adata, g_r)
    v_g = vector_for_gene(adata, g_g)
    v_b = vector_for_gene(adata, g_b)

    rmin, rmax = args.capar_minmax
    gmin, gmax = args.dh31_minmax
    bmin, bmax = args.lkr_minmax

    pdf_path = Path(args.out)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(pdf_path)) as pdf:
        # Page 1 (reference for size/limits)
        fig1, ref = single_gene_page(
            adata, g_r, vmin=rmin, vmax=rmax, cmap=custom_cmap_capar,
            title_text=g_r, basis=basis,
            na_color=LIGHT_GREY, size=args.pointsize, zero_is_na=False
        )

        # >>> NEW: pull the exact sizes Scanpy used on page 1
        ax0 = fig1.axes[0]
        ref_sizes = None
        for coll in ax0.collections:
            if hasattr(coll, "get_sizes"):
                s = coll.get_sizes()
                if s is not None and len(s) > 0:
                    ref_sizes = s
                    break

        pdf.savefig(fig1); plt.close(fig1)
        ref_xlim, ref_ylim, ref_figsize = ref

        # Page 2
        fig2, _ = single_gene_page(
            adata, g_g, vmin=gmin, vmax=gmax, cmap=custom_cmap_dh31,
            title_text=g_g, basis=basis,
            na_color=LIGHT_GREY, size=args.pointsize, zero_is_na=True
        )
        pdf.savefig(fig2); plt.close(fig2)

        # Page 3
        fig3, _ = single_gene_page(
            adata, g_b, vmin=bmin, vmax=bmax, cmap=custom_cmap_lkr,
            title_text=g_b, basis=basis,
            na_color=LIGHT_GREY, size=args.pointsize, zero_is_na=True
        )
        pdf.savefig(fig3); plt.close(fig3)

        # ---------- Page 4 (combined, exact single-gene colors + mixed overlaps) ----------
        # Normalize with your chosen min/max per gene (same as single pages)
        LIGHT_GREY_RGB = np.array([0.85, 0.85, 0.85])

        # ---------- Page 4 (combined, exact single-gene colors + mixed overlaps + RIGHT colorbars) ----------
        # ---------- Page 4 (combined with right-side single-gene-style colorbars) ----------
        # ---------- Page 4 (combined; light-grey bg; Dh31-R colorbar from Scanpy) ----------
        rn = clip_and_scale(v_r, rmin, rmax)  # CapaR
        gn = clip_and_scale(v_g, gmin, gmax)  # Dh31-R
        bn = clip_and_scale(v_b, bmin, bmax)  # Lkr

        import matplotlib as mpl
        import matplotlib.colors as mcolors

        # SAME cmaps as pages 1–3
        cmap_r = mpl.cm.get_cmap("Reds")
        cmap_g = mpl.cm.get_cmap("Greens")  # Dh31-R
        cmap_b = mpl.cm.get_cmap("Blues")

        # Map to the exact RGBs used on single-gene pages
        cr = cmap_r(rn)[:, :3];
        cr[rn <= 0] = 0.0
        cg = cmap_g(gn)[:, :3];
        cg[gn <= 0] = 0.0
        cb = cmap_b(bn)[:, :3];
        cb[bn <= 0] = 0.0

        # Screen blend; cells with no expression -> light grey
        combined_rgb = 1.0 - (1.0 - cr) * (1.0 - cg) * (1.0 - cb)
        mask_none = (rn <= 0) & (gn <= 0) & (bn <= 0)
        combined_rgb[mask_none] = LIGHT_GREY_RGB

        # Page-4 dot sizes (reuse Page 1 unless overridden)
        n = coords.shape[0]
        if getattr(args, "combined_pointsize", None) is None:
            if ref_sizes is None:
                sizes_vec = np.full(n, float(args.pointsize))
            else:
                rs = np.asarray(ref_sizes).ravel()
                sizes_vec = np.full(n, float(args.pointsize)) if rs.size != n else rs
        else:
            sizes_vec = np.full(n, float(args.combined_pointsize))
        alpha = getattr(args, "combined_alpha", 1.0)

        # ==== 1) Ask Scanpy to create a figure WITH Dh31-R colorbar (same as your UMAP) ====
        if basis.lower() == "umap":
            fig4 = sc.pl.umap(
                adata, color=g_g, cmap="Greens", na_color=LIGHT_GREY,
                frameon=False, size=float(args.pointsize),
                vmin=gmin, vmax=gmax, return_fig=True, show=False
            )
        elif basis.lower() == "tsne":
            fig4 = sc.pl.tsne(
                adata, color=g_g, cmap="Greens", na_color=LIGHT_GREY,
                frameon=False, size=float(args.pointsize),
                vmin=gmin, vmax=gmax, return_fig=True, show=False
            )
        else:
            raise ValueError(f"Unsupported basis: {basis}")

        # Main axes should be axes[0]; a colorbar axis is added on the right by Scanpy.
        ax = fig4.axes[0]

        # Match your reference limits/figsize and remove any default title
        fig4.set_size_inches(*ref_figsize)
        ax.set_xlim(ref_xlim);
        ax.set_ylim(ref_ylim)
        ax.set_title("")  # we'll set our own below

        # ==== 2) Clear the scatter drawn by Scanpy, keep the colorbar ====
        ax.cla()
        ax.set_axis_off()
        ax.set_facecolor("white");
        fig4.patch.set_facecolor("white")

        # Light-grey organ background (same grey as zero-expression cells)
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=sizes_vec, c=[LIGHT_GREY_RGB],
                   edgecolors="none", linewidths=0, alpha=1.0)

        # Overlay blended colors only for expressing cells
        mask_expr = ~mask_none
        if np.any(mask_expr):
            ax.scatter(coords[mask_expr, 0], coords[mask_expr, 1],
                       s=sizes_vec[mask_expr], c=combined_rgb[mask_expr],
                       edgecolors="none", linewidths=0, alpha=alpha)

        # Minimal title with gene names only
        ax.set_title(f"{g_r} + {g_g} + {g_b}", fontsize=12, pad=6)

        fig4.tight_layout()
        pdf.savefig(fig4);
        plt.close(fig4)
        # -----------------------------------------------------------------------------------

        # Page 4 (combined) — SAME dot size as page 1
        #r = clip_and_scale(v_r, rmin, rmax)
        #g = clip_and_scale(v_g, gmin, gmax)
        #b = clip_and_scale(v_b, bmin, bmax)
        #fig4 = overlay_rgb_page(
            #coords, r=r, g=g, b=b,
            #title_text=f"{g_r} + {g_g} + {g_b}",
            #size=args.pointsize, xlim=ref_xlim, ylim=ref_ylim,
            #figsize=ref_figsize, sizes=ref_sizes  # <-- key line
        #)
        #pdf.savefig(fig4); plt.close(fig4)

    print(f"[OK] wrote PDF → {pdf_path.resolve()}")

if __name__ == "__main__":
    main()
