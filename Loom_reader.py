# gene_expression_viewer_streamlit_soft_wide.py
# Streamlit app: load a .loom file, choose genes, change colors, and export UMAP gene maps.
#
# Run:
#   streamlit run gene_expression_viewer_streamlit_soft_wide.py
#
# Input:
#   - A .loom file path, or upload a .loom file from the browser.
#
# Output:
#   - Interactive 4-panel gene expression map
#   - Wider map shape controls: panel width/height + X stretch
#   - Download PNG
#   - Download PDF
#
# Notes:
#   - If the loom already contains UMAP/t-SNE coordinates, the app uses them.
#   - If not, the app can calculate UMAP from the expression matrix.

from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path
from typing import Iterable

import anndata as ad
import loompy
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from scipy.sparse import csr_matrix


# =========================
# Page style
# =========================
st.set_page_config(page_title="Single-cell Gene Expression Viewer", layout="wide")

st.title("Single-cell Gene Expression Viewer")
st.caption("Load a .loom file, choose genes, change colors, and export pretty UMAP expression maps.")

LIGHT_GREY = "#d3d3d3"


# =========================
# Target presets
# =========================
TARGET_PRESETS = {
    "Octalpha2R": ["Octalpha2R", "Octα2R", "Oct alpha2 R", "Oct-alpha2R", "FBgn0038653", "CG18208", "TC011641"],
    "Tdc1": ["Tdc1", "Tdc-1", "tdc1", "FBgn0259977", "CG30445"],
    "Tdc2": ["Tdc2", "Tdc-2", "tdc2", "FBgn0050446", "CG30446"],
    "TyrR": ["TyrR", "FBgn0038542", "CG7431"],
    "TyrRII": ["TyrRII", "TyrR2", "TyrR-II", "FBgn0038541", "CG16766"],
    "CapaR": ["CapaR", "capaR", "FBgn0037100", "CG14575"],
    "Lkr": ["Lkr", "LkR", "LKR", "LK-R", "FBgn0035610", "CG10626"],
    "Tbh": ["Tbh", "tbh", "FBgn0010329", "CG1543"],
}

DEFAULT_GENES = ["Octalpha2R", "Tdc1", "Tdc2", "TyrR"]
DEFAULT_COLORS = ["#6E8FD6", "#D84ACB", "#2F7F73", "#E46C5A"]


# =========================
# General helpers
# =========================
def clean_path(path_text: str) -> str:
    """Remove quotes/spaces that often appear when copying Windows/Mac paths."""
    return str(path_text).strip().strip('"').strip("'")


def norm_name(x: object) -> str:
    """Normalize gene names for forgiving matching."""
    s = str(x)
    s = s.replace("β", "beta").replace("Β", "beta")
    s = s.replace("α", "alpha").replace("Α", "alpha")
    s = s.lower()
    for ch in ["-", "_", ".", " ", "/", "\\", ":", ";", "|", "(", ")"]:
        s = s.replace(ch, "")
    return s


def _to_dense_1d(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    v = np.asarray(x).ravel().astype(float)
    v[np.isnan(v)] = 0.0
    return v


def vector_for_gene(adata: ad.AnnData, gene: str) -> np.ndarray:
    return _to_dense_1d(adata[:, gene].X)


def guess_key(candidates: Iterable[str], available: set[str]) -> str | None:
    for k in candidates:
        if k in available:
            return k
    return None


def safe_n_comps(adata: ad.AnnData, requested: int = 50) -> int:
    """PCA components must be smaller than cells/genes."""
    max_allowed = max(2, min(adata.n_obs, adata.n_vars) - 1)
    return int(min(requested, max_allowed))


def safe_n_pcs(adata: ad.AnnData, requested: int) -> int:
    if "X_pca" not in adata.obsm:
        return 2
    max_pcs = adata.obsm["X_pca"].shape[1]
    return int(max(2, min(requested, max_pcs)))


def get_embedding_from_ca(ca):
    """Find UMAP/t-SNE coordinates from loom column attributes."""
    if "Embedding" in ca:
        arr = ca["Embedding"]
        if getattr(arr, "dtype", None) is not None and arr.dtype.names and len(arr.dtype.names) >= 2:
            if "_X" in arr.dtype.names and "_Y" in arr.dtype.names:
                xname, yname = "_X", "_Y"
            else:
                xname, yname = arr.dtype.names[0], arr.dtype.names[1]
            return "umap", np.column_stack([arr[xname], arr[yname]])

    pairs = [
        ("Embeddings_X", "Embeddings_Y", "umap"),
        ("UMAP_1", "UMAP_2", "umap"),
        ("umap_1", "umap_2", "umap"),
        ("X_umap_1", "X_umap_2", "umap"),
        ("TSNE_1", "TSNE_2", "tsne"),
        ("tsne_1", "tsne_2", "tsne"),
        ("X", "Y", "umap"),
        ("_X", "_Y", "umap"),
    ]
    for xkey, ykey, basis in pairs:
        if xkey in ca and ykey in ca:
            return basis, np.column_stack([ca[xkey], ca[ykey]])

    return None, None


def read_loom_safely(loom_path: str | Path) -> ad.AnnData:
    """Read loom into AnnData and keep all row attributes in adata.var for gene matching."""
    loom_path = Path(loom_path)

    with loompy.connect(str(loom_path), mode="r", validate=False) as lc:
        n_rows, n_cols = lc.shape

        ra_keys = list(lc.ra.keys())
        ca_keys = list(lc.ca.keys())

        var_key = guess_key(
            [
                "Gene", "gene", "GeneName", "GeneID", "GeneNameExtended",
                "genes", "feature_name", "feature_names", "var_names",
                "Accession", "Symbol", "symbol",
            ],
            set(ra_keys),
        )

        if var_key is None:
            var_key = next((k for k in ra_keys if getattr(lc.ra[k], "shape", [None])[0] == n_rows), None)

        if var_key:
            var_names = np.array(lc.ra[var_key]).astype(str)
        else:
            var_names = np.array([f"g{i}" for i in range(n_rows)], dtype=str)

        obs_key = guess_key(
            ["CellID", "cell_id", "CellName", "Barcode", "Cell", "obs_names", "cell_names"],
            set(ca_keys),
        )

        if obs_key is None:
            obs_key = next((k for k in ca_keys if getattr(lc.ca[k], "shape", [None])[0] == n_cols), None)

        if obs_key:
            obs_names = np.array(lc.ca[obs_key]).astype(str)
        else:
            obs_names = np.array([f"c{i}" for i in range(n_cols)], dtype=str)

        # loom shape = genes x cells; AnnData expects cells x genes.
        X = csr_matrix(lc[:, :].T)

        adata = ad.AnnData(X=X)
        adata.var_names = var_names
        adata.obs_names = obs_names
        adata.var_names_make_unique()
        adata.obs_names_make_unique()

        # Store row attributes for matching symbols / FlyBase / CG / TC IDs.
        for k in ra_keys:
            try:
                arr = np.array(lc.ra[k])
                if arr.shape[0] == n_rows:
                    adata.var[k] = arr.astype(str)
            except Exception:
                pass

        # Store column attributes if they look one-dimensional.
        for k in ca_keys:
            try:
                arr = np.array(lc.ca[k])
                if arr.shape[0] == n_cols and arr.ndim == 1:
                    adata.obs[k] = arr.astype(str)
            except Exception:
                pass

        basis, coords = get_embedding_from_ca(lc.ca)
        if coords is not None:
            if basis == "tsne":
                adata.obsm["X_tsne"] = coords
            else:
                adata.obsm["X_umap"] = coords

    adata.uns["source_type"] = ".loom file"
    return adata


def find_gene_in_adata(adata: ad.AnnData, aliases: Iterable[str]) -> tuple[str | None, str, str]:
    """
    Find a gene by searching:
    1. adata.var_names
    2. every column in adata.var
    3. forgiving partial matches
    """
    wanted = [norm_name(a) for a in aliases if str(a).strip()]

    # 1) exact match in var_names
    for var_name in adata.var_names:
        if norm_name(var_name) in wanted:
            return str(var_name), "var_names", str(var_name)

    # 2) exact match in all var columns
    for col in adata.var.columns:
        values = adata.var[col].astype(str)
        for idx, value in values.items():
            if norm_name(value) in wanted:
                return str(idx), col, str(value)

    # 3) forgiving partial match in var_names
    for var_name in adata.var_names:
        nv = norm_name(var_name)
        for w in wanted:
            if w and (w in nv or nv in w):
                return str(var_name), "var_names_partial", str(var_name)

    # 4) forgiving partial match in all var columns
    for col in adata.var.columns:
        values = adata.var[col].astype(str)
        for idx, value in values.items():
            nv = norm_name(value)
            for w in wanted:
                if w and (w in nv or nv in w):
                    return str(idx), f"{col}_partial", str(value)

    return None, "", ""


def find_default_gene(adata: ad.AnnData, preset_name: str) -> str:
    aliases = TARGET_PRESETS.get(preset_name, [preset_name])
    found, _, _ = find_gene_in_adata(adata, aliases)
    if found is not None and found in adata.var_names:
        return found
    return str(adata.var_names[0])


def gene_colors_rgba(expr: np.ndarray, vmin: float, vmax: float, color_hex: str, boost: float) -> np.ndarray:
    """
    App-style expression colors:
    - zero/low cells = light grey
    - positive cells = white -> selected color
    - boost changes visual intensity only, not the data
    """
    boosted = expr * float(boost)

    mask = boosted <= vmin
    expr_clipped = np.clip(boosted, vmin, vmax)

    if vmax > vmin:
        scaled = (expr_clipped - vmin) / (vmax - vmin)
    else:
        scaled = np.zeros_like(expr_clipped, dtype=float)

    cmap = mcolors.LinearSegmentedColormap.from_list("gene_cmap", ["#ffffff", color_hex])
    rgba = cmap(scaled)
    rgba[mask] = mcolors.to_rgba(LIGHT_GREY)
    return rgba


def make_gene_fig(
    *,
    coords: np.ndarray,
    adata: ad.AnnData,
    gene_names: list[str],
    colors: list[str],
    vmins: list[float],
    vmaxs: list[float],
    boost: float,
    point_size: float,
    show_colorbar: bool,
    panel_width: float,
    panel_height: float,
    x_stretch: float,
    y_stretch: float,
    keep_equal_aspect: bool,
    show_positive_count: bool,
    title_prefix: str = "",
) -> plt.Figure:
    n = len(gene_names)

    # Visual-only shape control. This does not change expression values or UMAP calculation.
    coords_plot = np.asarray(coords).copy()
    coords_plot[:, 0] = coords_plot[:, 0] * float(x_stretch)
    coords_plot[:, 1] = coords_plot[:, 1] * float(y_stretch)

    width = max(float(panel_width) * n, 8)
    fig, axes = plt.subplots(1, n, figsize=(width, float(panel_height)))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        gene = gene_names[i]
        expr = vector_for_gene(adata, gene)
        rgba = gene_colors_rgba(expr, vmins[i], vmaxs[i], colors[i], boost=boost)

        ax.scatter(coords_plot[:, 0], coords_plot[:, 1], c=rgba, s=point_size, edgecolors="none")

        # Picture-2 style: use a wide plotting box. If equal aspect is OFF,
        # Matplotlib fills the landscape panel instead of making a tall/square map.
        if keep_equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("auto")

        positive = int(np.sum(expr > 0))
        if show_positive_count:
            ax.set_title(f"{title_prefix}{gene}\npositive cells: {positive:,}", fontsize=11)
        else:
            ax.set_title(f"{title_prefix}{gene}", fontsize=12, pad=8)

        ax.margins(0.02)
        ax.axis("off")

        if show_colorbar:
            cmap = mcolors.LinearSegmentedColormap.from_list("gene_cmap", ["#ffffff", colors[i]])
            norm = mcolors.Normalize(vmin=vmins[i], vmax=vmaxs[i])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            cb.set_label(f"boosted expression ×{boost:g}", fontsize=7)

    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig: plt.Figure, dpi: int) -> bytes:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return buf.getvalue()


def build_pdf_bytes(
    *,
    overview_fig: plt.Figure,
    coords: np.ndarray,
    adata: ad.AnnData,
    gene_names: list[str],
    colors: list[str],
    vmins: list[float],
    vmaxs: list[float],
    boost: float,
    point_size: float,
    show_colorbar: bool,
    panel_width: float,
    panel_height: float,
    x_stretch: float,
    y_stretch: float,
    keep_equal_aspect: bool,
    show_positive_count: bool,
) -> bytes:
    buf = BytesIO()
    with PdfPages(buf) as pdf:
        # one gene per page
        for i, gene in enumerate(gene_names):
            fig_i = make_gene_fig(
                coords=coords,
                adata=adata,
                gene_names=[gene],
                colors=[colors[i]],
                vmins=[vmins[i]],
                vmaxs=[vmaxs[i]],
                boost=boost,
                point_size=point_size,
                show_colorbar=show_colorbar,
                panel_width=panel_width,
                panel_height=panel_height,
                x_stretch=x_stretch,
                y_stretch=y_stretch,
                keep_equal_aspect=keep_equal_aspect,
                show_positive_count=show_positive_count,
            )
            pdf.savefig(fig_i, bbox_inches="tight")
            plt.close(fig_i)

        # final overview page
        pdf.savefig(overview_fig, bbox_inches="tight")

    buf.seek(0)
    return buf.getvalue()


@st.cache_resource(show_spinner=True)
def load_loom_cached(path_text: str, do_normalize_log: bool) -> ad.AnnData:
    data_path = clean_path(path_text)
    p = Path(data_path)

    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {data_path}")
    if p.suffix.lower() != ".loom":
        raise ValueError("This app version expects a .loom file.")

    adata = read_loom_safely(p)

    if do_normalize_log:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    return adata


def calculate_umap_if_needed(adata: ad.AnnData, n_neighbors: int, n_pcs: int, min_dist: float) -> np.ndarray:
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=safe_n_comps(adata, n_pcs), svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=safe_n_pcs(adata, n_pcs))
    sc.tl.umap(adata, min_dist=min_dist)
    return adata.obsm["X_umap"]


def get_default_coords(adata: ad.AnnData) -> tuple[str, np.ndarray | None]:
    if "X_umap" in adata.obsm:
        return "UMAP from loom", adata.obsm["X_umap"]
    if "X_tsne" in adata.obsm:
        return "t-SNE from loom", adata.obsm["X_tsne"]
    return "No embedding found", None


# =========================
# Sidebar: input and style
# =========================
with st.sidebar:
    st.header("1. Load loom")

    input_mode = st.radio(
        "Input mode",
        ["Path on this computer/server", "Upload loom file"],
        index=0,
    )

    do_normalize_log = st.checkbox(
        "Normalize + log1p",
        value=True,
        help="Keep ON for raw count loom files. Turn OFF if your loom is already log-normalized.",
    )

    uploaded_loom = None
    loom_path = ""

    if input_mode == "Path on this computer/server":
        loom_path = st.text_input(
            "Loom file path",
            "malpighian_tubule_10x.loom",
            help="Example Windows path: D:/RNA loom/file.loom",
        )
    else:
        uploaded_loom = st.file_uploader("Upload .loom", type=["loom"])

    load_clicked = st.button("Load / reload loom", type="primary")

    st.divider()
    st.header("2. Style")
    boost = st.slider("Visual expression boost", 0.5, 10.0, 2.0, step=0.5)
    point_size = st.slider("Point size", 1.0, 20.0, 5.0, step=0.5)
    show_colorbar = st.checkbox(
        "Show side color bar",
        value=False,
        help="OFF gives the clean landscape format like your second picture. Turn ON when you need the scale bar.",
    )
    show_positive_count = st.checkbox(
        "Show positive cell count in title",
        value=False,
        help="OFF gives a clean title like 'Hector'. Turn ON to show positive cell number.",
    )
    output_dpi = st.slider("PNG resolution / DPI", 100, 600, 300, step=50)

    st.divider()
    st.header("3. Map shape")
    panel_width = st.slider(
        "Panel width",
        3.0, 12.0, 6.3, step=0.2,
        help="Increase this to make each UMAP panel wider. Default is now softer, less stretched than the previous picture-2 version.",
    )
    panel_height = st.slider(
        "Panel height",
        2.5, 8.0, 5.2, step=0.2,
        help="Increase this a little if the map still looks too wide/stretched.",
    )
    x_stretch = st.slider(
        "Widen UMAP shape / X stretch",
        0.5, 3.0, 1.0, step=0.05,
        help="Visual-only: stretches the UMAP horizontally. Use 1.0 for the original shape.",
    )
    y_stretch = st.slider(
        "Y stretch",
        0.5, 2.0, 1.0, step=0.05,
        help="Usually keep this at 1.0.",
    )
    keep_equal_aspect = st.checkbox(
        "Keep equal X/Y aspect",
        value=False,
        help="Turn ON for a less distorted scientific shape. Default OFF keeps the nice preview style, but now with a softer width.",
    )
    st.caption("Soft-wide default: 6.3 × 5.2. Previous wider version was 7.5 × 5.0.")


# =========================
# Load data
# =========================
if load_clicked:
    load_loom_cached.clear()
    for key in ["adata", "coords", "coords_label", "load_key"]:
        st.session_state.pop(key, None)

try:
    if input_mode == "Upload loom file":
        if uploaded_loom is None:
            st.info("Upload a .loom file to start.")
            st.stop()

        # Save upload to a temporary file. Streamlit upload objects are not normal file paths.
        tmp_dir = Path(tempfile.gettempdir()) / "streamlit_loom_viewer"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / uploaded_loom.name
        tmp_path.write_bytes(uploaded_loom.getbuffer())

        load_key = (str(tmp_path), do_normalize_log, uploaded_loom.size)
        if st.session_state.get("load_key") != load_key:
            st.session_state.load_key = load_key
            st.session_state.adata = load_loom_cached(str(tmp_path), do_normalize_log)
            label, coords = get_default_coords(st.session_state.adata)
            st.session_state.coords_label = label
            st.session_state.coords = coords

    else:
        load_key = (clean_path(loom_path), do_normalize_log)
        if st.session_state.get("load_key") != load_key:
            st.session_state.load_key = load_key
            st.session_state.adata = load_loom_cached(loom_path, do_normalize_log)
            label, coords = get_default_coords(st.session_state.adata)
            st.session_state.coords_label = label
            st.session_state.coords = coords

except Exception as e:
    st.error("Could not load loom.")
    st.exception(e)
    st.stop()

adata = st.session_state.adata
coords = st.session_state.coords
coords_label = st.session_state.coords_label

st.success(f"Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes/features")


# =========================
# Coordinates
# =========================
st.subheader("Coordinates")

if coords is not None:
    st.caption(f"Using: {coords_label}")
else:
    st.warning("No UMAP/t-SNE coordinates were found inside the loom. Calculate UMAP below.")

with st.expander("Calculate new UMAP if needed / wanted", expanded=(coords is None)):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_neighbors = st.slider("n_neighbors", 2, 100, 15)
    with c2:
        n_pcs = st.slider("n_pcs", 2, 100, 30)
    with c3:
        min_dist = st.slider("min_dist", 0.0, 1.0, 0.5, step=0.01)
    with c4:
        recalc_umap = st.button("Calculate UMAP")

    if recalc_umap:
        with st.spinner("Calculating UMAP..."):
            st.session_state.coords = calculate_umap_if_needed(
                adata, n_neighbors=int(n_neighbors), n_pcs=int(n_pcs), min_dist=float(min_dist)
            )
            st.session_state.coords_label = "New calculated UMAP"
            coords = st.session_state.coords
            coords_label = st.session_state.coords_label
        st.success("UMAP calculated.")

if coords is None:
    st.stop()


# =========================
# Gene controls
# =========================
st.subheader("Genes and colors")

genes = list(adata.var_names)
if len(genes) == 0:
    st.error("No genes/features available.")
    st.stop()

cols = st.columns(4)
gene_names: list[str] = []
colors: list[str] = []
vmins: list[float] = []
vmaxs: list[float] = []

for i in range(4):
    with cols[i]:
        default_preset = DEFAULT_GENES[i]
        default_gene = find_default_gene(adata, default_preset)
        default_index = genes.index(default_gene) if default_gene in genes else 0

        st.markdown(f"**Gene {i + 1}**")
        search_text = st.text_input(
            f"Search/alias {i + 1}",
            default_preset,
            key=f"search_{i}",
            help="You can type a symbol, TC ID, FlyBase ID, or CG ID.",
        )

        found_gene, found_col, found_value = find_gene_in_adata(
            adata,
            TARGET_PRESETS.get(search_text, [search_text]),
        )

        if found_gene is not None and found_gene in genes:
            default_index = genes.index(found_gene)
            st.caption(f"Matched: `{found_gene}` via {found_col}")

        gene = st.selectbox(
            f"Choose gene {i + 1}",
            genes,
            index=default_index,
            key=f"gene_select_{i}",
        )
        gene_names.append(gene)

        color = st.color_picker(f"Color {i + 1}", DEFAULT_COLORS[i], key=f"color_{i}")
        colors.append(color)

        expr = vector_for_gene(adata, gene)
        boosted_expr = expr * float(boost)
        expr_max = float(np.max(boosted_expr)) if boosted_expr.size else 1.0
        expr_max = expr_max if expr_max > 0 else 1.0

        positive = boosted_expr[boosted_expr > 0]
        expr_q = float(np.quantile(positive, 0.995)) if positive.size else expr_max
        expr_q = min(max(expr_q, 0.01), expr_max)

        step = max(float(expr_max) / 200.0, 0.01)
        vmin_vmax = st.slider(
            f"Color range {i + 1}",
            min_value=0.0,
            max_value=float(expr_max),
            value=(0.0, float(expr_q)),
            step=step,
            key=f"range_{i}",
        )
        vmins.append(float(vmin_vmax[0]))
        vmaxs.append(float(vmin_vmax[1]))


# =========================
# Plot
# =========================
fig_overview = make_gene_fig(
    coords=coords,
    adata=adata,
    gene_names=gene_names,
    colors=colors,
    vmins=vmins,
    vmaxs=vmaxs,
    boost=boost,
    point_size=point_size,
    show_colorbar=show_colorbar,
    panel_width=panel_width,
    panel_height=panel_height,
    x_stretch=x_stretch,
    y_stretch=y_stretch,
    keep_equal_aspect=keep_equal_aspect,
    show_positive_count=show_positive_count,
)

st.pyplot(fig_overview, use_container_width=True)


# =========================
# Export
# =========================
st.divider()
st.subheader("Export")

safe_name = "_".join([g.replace("/", "_").replace("\\", "_").replace(" ", "_") for g in gene_names])
png_name = f"gene_expression_map_{safe_name}.png"
pdf_name = f"gene_expression_map_{safe_name}.pdf"

png_bytes = fig_to_png_bytes(fig_overview, dpi=int(output_dpi))
pdf_bytes = build_pdf_bytes(
    overview_fig=fig_overview,
    coords=coords,
    adata=adata,
    gene_names=gene_names,
    colors=colors,
    vmins=vmins,
    vmaxs=vmaxs,
    boost=boost,
    point_size=point_size,
    show_colorbar=show_colorbar,
    panel_width=panel_width,
    panel_height=panel_height,
    x_stretch=x_stretch,
    y_stretch=y_stretch,
    keep_equal_aspect=keep_equal_aspect,
    show_positive_count=show_positive_count,
)

d1, d2 = st.columns(2)
with d1:
    st.download_button(
        "Download PNG",
        data=png_bytes,
        file_name=png_name,
        mime="image/png",
    )

with d2:
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=pdf_name,
        mime="application/pdf",
    )

plt.close(fig_overview)
