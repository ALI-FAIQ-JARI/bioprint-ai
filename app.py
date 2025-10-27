
# BioPrint AI (v2, Neo-Brutalism, multi-accent, styled plots)
# Version without Confidence display

from pathlib import Path
from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import streamlit as st
import plotly.express as px
from plotly.graph_objects import Figure

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests


# ----------------------------- #
# Config & constants
# ----------------------------- #
DATA_REAL = Path("data/bioink_ml_ready_engineered.csv")
DATA_ALL  = Path("data/bioink_ml_ready_augmented_v2_10k.csv")
MAIN_NUMS = ["Pressure_kPa","Temperature_C","Speed_mms","Needle_um"]
HAS_CELLS = "Has_Cells"
CELL_CONC = "Cells (e6/ml)"
PHYS      = ["volumetric_flow_rate","wall_shear_rate","specific_pressure"]
TEMP_CAP_IF_CELLS = 37.0
EPS_MATCH = 1e-9

st.set_page_config(page_title="BioPrint AI — Intelligent Recommender", layout="wide")

# =======================================================================
# NEO-BRUTALISM UI (always on) — multi-accent
# =======================================================================
NB_COLORS = {
    "bg": "#FAFAFA",         # page background (off-white)
    "ink": "#111111",        # main text
    "muted": "#6B7280",      # secondary text
    "stroke": "#111111",     # outlines
    "surface": "#FFFFFF",    # card surface
    "accent1": "#00E5FF",    # cyan
    "accent2": "#FF2E63",    # magenta/pink
    "accent3": "#FFB020",    # amber
    "accent4": "#00C853",    # green
    "accent5": "#111111",    # ink as accent
}

# Plotly defaults (multi-accent)
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = [
    NB_COLORS["accent1"], NB_COLORS["accent2"],
    NB_COLORS["accent3"], NB_COLORS["accent4"], NB_COLORS["accent5"]
]

def apply_plotly_neo(fig: Figure) -> Figure:
    """Chunky axes, hard outlines, flat bg, big markers/lines."""
    fig.update_layout(
        paper_bgcolor=NB_COLORS["bg"],
        plot_bgcolor=NB_COLORS["surface"],
        font=dict(color=NB_COLORS["ink"], family="Inter, ui-sans-serif", size=14),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(
        showline=True, linewidth=3, linecolor=NB_COLORS["stroke"],
        showgrid=False, zeroline=False, ticks="outside", tickwidth=2
    )
    fig.update_yaxes(
        showline=True, linewidth=3, linecolor=NB_COLORS["stroke"],
        showgrid=False, zeroline=False, ticks="outside", tickwidth=2
    )
    # Bolder legend
    fig.update_layout(
        legend=dict(
            bgcolor=NB_COLORS["surface"],
            bordercolor=NB_COLORS["stroke"],
            borderwidth=3,
            itemclick="toggleothers"
        ),
        title=dict(font=dict(size=20, family="Inter, ui-sans-serif")),
    )
    # Marker outlines
    fig.update_traces(marker=dict(line=dict(width=2, color=NB_COLORS["stroke"])), selector=dict(mode="markers"))
    return fig

def apply_neo_brutalism():
    """Inject CSS for neo-brutalist multi-accent UI."""
    css = f"""
    <style>
    :root {{
      --nb-bg: {NB_COLORS['bg']};
      --nb-ink: {NB_COLORS['ink']};
      --nb-muted: {NB_COLORS['muted']};
      --nb-stroke: {NB_COLORS['stroke']};
      --nb-surface: {NB_COLORS['surface']};
      --nb-a1: {NB_COLORS['accent1']};
      --nb-a2: {NB_COLORS['accent2']};
      --nb-a3: {NB_COLORS['accent3']};
      --nb-a4: {NB_COLORS['accent4']};
      --nb-a5: {NB_COLORS['accent5']};
      --nb-radius: 18px;
      --nb-gap: 14px;
      --nb-shadow: 8px 8px 0 0 var(--nb-stroke);
    }}

    /* Load Google font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Page & typography */
    .stApp {{
      background: var(--nb-bg) !important;
      color: var(--nb-ink) !important;
    }}
    html, body, [class^="css"] {{
      font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Apple Color Emoji','Segoe UI Emoji' !important;
    }}
    h1, h2, h3, h4 {{
      font-weight: 800 !important;
      letter-spacing: -0.5px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] > div:first-child {{
      background: var(--nb-surface);
      border: 3px solid var(--nb-stroke);
      border-radius: var(--nb-radius);
      box-shadow: var(--nb-shadow);
      padding: 14px;
    }}

    /* Color legend badges in sidebar */
    .nb-badges {{
      display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px; margin-bottom: 6px;
    }}
    .nb-badge {{
      padding: 4px 8px; font-weight: 800; border: 3px solid var(--nb-stroke);
      border-radius: 12px; box-shadow: var(--nb-shadow);
      color: #000;
    }}
    .nb-badge.a1 {{ background: var(--nb-a1); }}
    .nb-badge.a2 {{ background: var(--nb-a2); }}
    .nb-badge.a3 {{ background: var(--nb-a3); }}
    .nb-badge.a4 {{ background: var(--nb-a4); }}
    .nb-badge.a5 {{ background: var(--nb-a5); color: #fff; }}

    /* Buttons / download buttons (alternate accents using nth-of-type) */
    .stButton > button, .stDownloadButton > button {{
      color: #000 !important;
      border: 3px solid var(--nb-stroke) !important;
      border-radius: 14px !important;
      box-shadow: var(--nb-shadow) !important;
      font-weight: 800 !important;
      text-transform: uppercase;
      letter-spacing: 0.6px;
    }}
    .stButton > button:nth-of-type(5n+1), .stDownloadButton > button:nth-of-type(5n+1) {{ background: var(--nb-a1) !important; }}
    .stButton > button:nth-of-type(5n+2), .stDownloadButton > button:nth-of-type(5n+2) {{ background: var(--nb-a2) !important; }}
    .stButton > button:nth-of-type(5n+3), .stDownloadButton > button:nth-of-type(5n+3) {{ background: var(--nb-a3) !important; }}
    .stButton > button:nth-of-type(5n+4), .stDownloadButton > button:nth-of-type(5n+4) {{ background: var(--nb-a4) !important; }}
    .stButton > button:nth-of-type(5n+5), .stDownloadButton > button:nth-of-type(5n+5) {{ background: var(--nb-a5) !important; color: #fff !important; }}
    .stButton > button:active, .stDownloadButton > button:active {{
      transform: translate(4px, 4px);
      box-shadow: 4px 4px 0 0 var(--nb-stroke) !important;
    }}

    /* Inputs */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, .stNumberInput > div {{
      background: var(--nb-surface) !important;
      border: 3px solid var(--nb-stroke) !important;
      border-radius: 12px !important;
      box-shadow: var(--nb-shadow) !important;
    }}

    /* Tabs — alternating accent backgrounds */
    .stTabs [data-baseweb="tab-list"] {{
      gap: var(--nb-gap);
      border-bottom: none !important;
    }}
    .stTabs [data-baseweb="tab"] {{
      color: var(--nb-ink);
      border: 3px solid var(--nb-stroke);
      border-radius: 12px;
      box-shadow: var(--nb-shadow);
      padding: 10px 14px;
      font-weight: 700;
      text-transform: uppercase;
      background: var(--nb-surface);
    }}
    .stTabs [data-baseweb="tab"]:nth-child(5n+1)[aria-selected="true"] {{ background: var(--nb-a1) !important; }}
    .stTabs [data-baseweb="tab"]:nth-child(5n+2)[aria-selected="true"] {{ background: var(--nb-a2) !important; }}
    .stTabs [data-baseweb="tab"]:nth-child(5n+3)[aria-selected="true"] {{ background: var(--nb-a3) !important; }}
    .stTabs [data-baseweb="tab"]:nth-child(5n+4)[aria-selected="true"] {{ background: var(--nb-a4) !important; }}
    .stTabs [data-baseweb="tab"]:nth-child(5n+5)[aria-selected="true"] {{ background: var(--nb-a5) !important; color: #fff; }}

    /* Metrics */
    div[data-testid="stMetricValue"] {{ font-weight: 800 !important; }}
    div[data-testid="stMetric"] > div {{
      background: var(--nb-surface);
      border: 3px solid var(--nb-stroke);
      border-radius: 16px;
      padding: 10px 14px;
      box-shadow: var(--nb-shadow);
      position: relative;
    }}
    /* Left accent strips rotate colors per metric block */
    div[data-testid="stMetric"] > div::before {{
      content: "";
      position: absolute; left: -3px; top: -3px; bottom: -3px; width: 10px;
      background: var(--nb-a1);
      border-right: 3px solid var(--nb-stroke);
      border-top-left-radius: 14px; border-bottom-left-radius: 14px;
    }}
    /* Alternate strip colors for variety */
    [data-testid="stMetric"]:nth-of-type(5n+2) > div::before {{ background: var(--nb-a2); }}
    [data-testid="stMetric"]:nth-of-type(5n+3) > div::before {{ background: var(--nb-a3); }}
    [data-testid="stMetric"]:nth-of-type(5n+4) > div::before {{ background: var(--nb-a4); }}
    [data-testid="stMetric"]:nth-of-type(5n+5) > div::before {{ background: var(--nb-a5); }}

    /* Alerts / info boxes */
    .stAlert {{
      border: 3px solid var(--nb-stroke) !important;
      border-radius: 14px !important;
      box-shadow: var(--nb-shadow) !important;
    }}

    /* Dataframe container */
    div[data-testid="stDataFrame"] {{
      border: 3px solid var(--nb-stroke);
      border-radius: 14px;
      box-shadow: var(--nb-shadow);
    }}

    /* Expander */
    details {{
      background: var(--nb-surface);
      border: 3px solid var(--nb-stroke);
      border-radius: 14px;
      box-shadow: var(--nb-shadow);
      padding: 4px 8px;
    }}

    /* Chips (multiselect pills) — rotate accents for each chip */
    span[data-baseweb="tag"] {{
      color: #000 !important;
      border: 3px solid var(--nb-stroke) !important;
      border-radius: 12px !important;
      box-shadow: var(--nb-shadow) !important;
      font-weight: 700;
      background: var(--nb-a1) !important;
    }}
    span[data-baseweb="tag"]:nth-child(5n+2) {{ background: var(--nb-a2) !important; }}
    span[data-baseweb="tag"]:nth-child(5n+3) {{ background: var(--nb-a3) !important; }}
    span[data-baseweb="tag"]:nth-child(5n+4) {{ background: var(--nb-a4) !important; }}
    span[data-baseweb="tag"]:nth-child(5n+5) {{ background: var(--nb-a5) !important; color: #fff !important; }}

    /* Divider */
    hr {{ border: none; border-top: 3px solid var(--nb-stroke); }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def neo_card_start(title: str | None = None, accent: str = "a1"):
    """Start a chunky bordered card with optional title + accent stripe."""
    stripe = f"var(--nb-{accent})" if accent in {"a1","a2","a3","a4","a5"} else "var(--nb-a1)"
    st.markdown(
        f"""
        <div style='background: var(--nb-surface); border: 3px solid var(--nb-stroke); border-radius: 18px; box-shadow: var(--nb-shadow); padding: 16px; margin: 10px 0; position: relative;'>
          <div style="position:absolute; left:-3px; top:-3px; bottom:-3px; width:12px; background:{stripe}; border-right:3px solid var(--nb-stroke); border-top-left-radius:16px; border-bottom-left-radius:16px;"></div>
        """,
        unsafe_allow_html=True,
    )
    if title:
        st.markdown(f"<div style='font-weight:800; font-size:1.1rem; margin-left:6px; margin-bottom:8px;'>{title}</div>", unsafe_allow_html=True)

def neo_card_end():
    st.markdown("</div>", unsafe_allow_html=True)

# Apply style immediately
apply_neo_brutalism()

# ----------------------------- #
# Physics & helpers
# ----------------------------- #
def compute_physics(row: Dict[str, float]) -> Tuple[float, float, float]:
    d_um = float(max(row.get("Needle_um", 1.0), 1.0))
    v    = float(max(row.get("Speed_mms", 1e-9), 1e-9))
    p    = float(max(row.get("Pressure_kPa", 0.0), 0.0))
    d_mm = d_um / 1000.0
    A = math.pi * (d_mm**2) / 4.0
    Q_mm3_s = A * v
    Q_mL_s  = Q_mm3_s * 1e-3
    R = d_mm / 2.0
    shear = (4.0 * Q_mm3_s) / (math.pi * (R**3) + 1e-12)
    specP = p / (Q_mL_s + 1e-12)
    return Q_mL_s, shear, specP

def cap_for_cells(params: Dict[str, float], has_cells: int) -> Dict[str, float]:
    r = dict(params)
    if has_cells == 1 and r.get("Temperature_C", 0.0) > TEMP_CAP_IF_CELLS:
        r["Temperature_C"] = TEMP_CAP_IF_CELLS
    return r

@st.cache_data
def load_data():
    df_real = pd.read_csv(DATA_REAL)
    df_all  = pd.read_csv(DATA_ALL) if DATA_ALL.exists() else df_real.copy()
    if HAS_CELLS not in df_real.columns:
        df_real[HAS_CELLS] = (~df_real.get(CELL_CONC, pd.Series([np.nan]*len(df_real))).isna()).astype(int)
    if HAS_CELLS not in df_all.columns:
        df_all[HAS_CELLS] = (~df_all.get(CELL_CONC, pd.Series([np.nan]*len(df_all))).isna()).astype(int)
    for df in (df_real, df_all):
        for col in PHYS:
            if col not in df.columns:
                df[col] = np.nan
        miss = df[PHYS].isna().any(axis=1)
        if miss.any():
            df.loc[miss, PHYS] = df.loc[miss].apply(lambda r: compute_physics(r), axis=1, result_type="expand")
    bands = df_real[MAIN_NUMS].quantile([0.05,0.95]).T
    bands.columns = ["P5","P95"]
    if "is_synthetic" not in df_real.columns:
        df_real["is_synthetic"] = 0
    if "is_synthetic" not in df_all.columns:
        df_all["is_synthetic"] = 0
    return df_real, df_all, bands

def _norm_key(p, t, s, n, hc, dec=6):
    return (round(float(p), dec),
            round(float(t), dec),
            round(float(s), dec),
            round(float(n), dec),
            int(hc))

@st.cache_resource
def build_real_lock_index(df: pd.DataFrame) -> dict:
    idx = {}
    hc_col = HAS_CELLS if HAS_CELLS in df.columns else None
    for _, r in df.iterrows():
        key = _norm_key(r["Pressure_kPa"], r["Temperature_C"], r["Speed_mms"],
                        r["Needle_um"], int(r[hc_col]) if hc_col else 0)
        idx[key] = r
    return idx

df_real, df_all, bands = load_data()
COMP_FLAGS = [c for c in df_all.columns if c.startswith("contains_")]
MAT_META   = [c for c in ["dominant_material_type","has_synthetic_polymer","has_natural_polymer"] if c in df_all.columns]
FLAG2LABEL = {flag: flag.replace("contains_", "").replace("_", " ").strip().title() for flag in COMP_FLAGS}
LABEL2FLAG = {v:k for k,v in FLAG2LABEL.items()}
SIM_COLS   = COMP_FLAGS + MAT_META + [HAS_CELLS]
REAL_LOCK = build_real_lock_index(df_real)

@st.cache_resource
def fit_knn_comp(df: pd.DataFrame, sim_cols: List[str]):
    X = df[sim_cols].copy()
    for c in sim_cols:
        if X[c].dtype == "O":
            X[c] = X[c].astype("category").cat.codes
    X = X.fillna(X.median(numeric_only=True))
    nbrs = NearestNeighbors(n_neighbors=50, metric="euclidean")
    nbrs.fit(X.values)
    med = X.median(numeric_only=True)
    return nbrs, med

KNN_COMP, SIM_MEDIANS = fit_knn_comp(df_all, SIM_COLS)

def encode_query_comp(q: pd.DataFrame) -> np.ndarray:
    x = q.copy()
    for c in SIM_COLS:
        if x[c].dtype == "O":
            x[c] = x[c].astype("category").cat.codes
    x = x.fillna(SIM_MEDIANS)
    return x.values

def _comp_key_from_row(row: pd.Series) -> tuple:
    key_parts = []
    for c in SIM_COLS:
        v = row.get(c, np.nan)
        if pd.isna(v):
            key_parts.append(None)
        else:
            if isinstance(v, str):
                key_parts.append(v.strip().lower())
            else:
                try:
                    if float(v).is_integer():
                        key_parts.append(int(v))
                    else:
                        key_parts.append(float(v))
                except Exception:
                    key_parts.append(v)
    return tuple(key_parts)

@st.cache_resource
def build_comp_lock_index(df: pd.DataFrame, sim_cols: List[str]) -> dict:
    idx = {}
    for i, r in df.iterrows():
        key = _comp_key_from_row(r)
        idx.setdefault(key, []).append(i)
    return idx

COMP_LOCK = build_comp_lock_index(df_real, SIM_COLS)

def comp_lock_lookup(inputs_comp: pd.DataFrame) -> pd.DataFrame | None:
    if inputs_comp is None or len(inputs_comp) == 0:
        return None
    key = _comp_key_from_row(inputs_comp.iloc[0])
    hit_idxs = COMP_LOCK.get(key)
    if not hit_idxs:
        return None
    return df_real.iloc[hit_idxs].copy()

def nearest_compositions(df: pd.DataFrame, qrow: pd.DataFrame, k=25):
    Xq = encode_query_comp(qrow[SIM_COLS])
    dist, idx = KNN_COMP.kneighbors(Xq, n_neighbors=k, return_distance=True)
    return dist[0], idx[0]

def build_comp_query(selected_labels: List[str], meta_choice: str, has_cells: int) -> pd.DataFrame:
    row = {c: 0 for c in SIM_COLS}
    for lab in selected_labels:
        f = LABEL2FLAG.get(lab)
        if f in row:
            row[f] = 1
    if 'dominant_material_type' in SIM_COLS:
        row['dominant_material_type'] = meta_choice
    row[HAS_CELLS] = int(has_cells)
    return pd.DataFrame([row])[SIM_COLS]

def recommend_simple(inputs_comp: pd.DataFrame, k_neighbors: int = 25):
    if inputs_comp is None or inputs_comp.empty:
        base = df_real
        rec = {c: float(base[c].median()) for c in MAIN_NUMS}
        has_cells = int(base[HAS_CELLS].median()) if HAS_CELLS in base.columns else 0
        Q, shear, specP = compute_physics(rec)
        metrics = {"volumetric_flow_rate": Q, "wall_shear_rate": shear, "specific_pressure": specP}
        ev = base.head(5)[MAIN_NUMS + [HAS_CELLS]].copy()
        return rec, metrics, ev

    has_cells = int(inputs_comp.iloc[0][HAS_CELLS]) if HAS_CELLS in inputs_comp.columns else 0

    real_hit = comp_lock_lookup(inputs_comp)
    if real_hit is not None and len(real_hit) > 0:
        if len(real_hit) == 1:
            row = real_hit.iloc[0]
            rec = {c: float(row[c]) for c in MAIN_NUMS}
        else:
            rec = {c: float(real_hit[c].median()) for c in MAIN_NUMS}
        Q, shear, specP = compute_physics(rec)
        metrics = {"volumetric_flow_rate": Q, "wall_shear_rate": shear, "specific_pressure": specP}
        ev = real_hit.head(5)[MAIN_NUMS + [HAS_CELLS]].copy()
        return rec, metrics, ev

    dist, idx = nearest_compositions(df_all, inputs_comp, k=k_neighbors)
    neigh = df_all.iloc[idx].copy()
    rec = {c: float(neigh[c].median()) for c in MAIN_NUMS}
    rec = cap_for_cells(rec, has_cells)
    for c in MAIN_NUMS:
        p5, p95 = float(bands.loc[c, "P5"]), float(bands.loc[c, "P95"])
        rec[c] = float(np.clip(rec[c], p5, p95))
    Q, shear, specP = compute_physics(rec)
    metrics = {"volumetric_flow_rate": Q, "wall_shear_rate": shear, "specific_pressure": specP}
    ev = neigh.head(5)[MAIN_NUMS + [HAS_CELLS]].copy()
    return rec, metrics, ev

def real_lock_lookup(params: Dict[str, float]) -> pd.Series | None:
    key = _norm_key(params["Pressure_kPa"], params["Temperature_C"],
                    params["Speed_mms"], params["Needle_um"],
                    params.get(HAS_CELLS, 0))
    return REAL_LOCK.get(key)
# ----------------------------- #
# Entry Page (Neo-Brutalism transition)
# ----------------------------- #
if "entered" not in st.session_state:
    st.session_state.entered = False

from textwrap import dedent

# ----------------------------- #
# Entry Page (Neo-Brutalism transition)
# ----------------------------- #
if "entered" not in st.session_state:
    st.session_state.entered = False

if "entered" not in st.session_state:
    st.session_state.entered = False

if "entered" not in st.session_state:
    st.session_state.entered = False

if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.markdown("<h1 style='text-align:center; font-size:3rem; font-weight:800;'>BioPrint-AI</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:gray; font-size:1.5rem;'>Less Trial and Error, More Viable Cells</h3>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        enter_clicked = st.button("🚀 Enter App", use_container_width=True)

    if enter_clicked:
        st.session_state.entered = True
        st.rerun()  # ✅ Use this instead of experimental_rerun()

    st.stop()








# ----------------------------- #
# App UI
# ----------------------------- #
st.image("assets/Header_Banner.png", use_container_width=True)
st.caption("Composition → Parameters with physics-aware recommendations")

# Sidebar — Composition
with st.sidebar:
    st.header("Bioink Composition")
    st.caption("Select all materials present in your formulation.")
    # Accent legend
    st.markdown(
        """
        <div class="nb-badges">
          <div class="nb-badge a1">Cyan</div>
          <div class="nb-badge a2">Magenta</div>
          <div class="nb-badge a3">Amber</div>
          <div class="nb-badge a4">Green</div>
          <div class="nb-badge a5">Ink</div>
        </div>
        """, unsafe_allow_html=True
    )
    options_labels = sorted(list(LABEL2FLAG.keys()))
    selected_labels = st.multiselect("Materials (multi-select)", options=options_labels)
    meta_choice = ""
    if "dominant_material_type" in MAT_META:
        modes = df_all["dominant_material_type"].dropna().astype(str).value_counts().index.tolist()
        meta_choice = st.selectbox("Dominant Material Type (optional)", options=[""] + modes[:10])
    has_cells_label = st.radio("Has Cells?", options=["Yes","No"], index=1)
    has_cells = 1 if has_cells_label == "Yes" else 0
    cells_e6 = st.number_input("Cells (e6/ml) (optional)", min_value=0.0, max_value=50.0, value=0.0, step=0.5)
    avail_needles = st.multiselect("Available Needles (µm)", options=sorted(df_all["Needle_um"].round().unique()))

comp_q = build_comp_query(selected_labels, meta_choice, has_cells)

# Tabs
T1, T2, T3, T4, T5 = st.tabs(["Recommender", "Viability & Risks", "Knowledge Hub / Learn", "Feedback & Collaboration", "About / Mission"])

with T1:
    rec, metrics, evidence = recommend_simple(comp_q)

    neo_card_start("Recommended Parameters", accent="a2")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pressure (kPa)", f"{rec['Pressure_kPa']:.1f}")
    c2.metric("Temperature (°C)", f"{rec['Temperature_C']:.1f}")
    c3.metric("Speed (mm/s)", f"{rec['Speed_mms']:.2f}")
    c4.metric("Needle (µm)", f"{rec['Needle_um']:.0f}")
    neo_card_end()

    neo_card_start("Physics", accent="a3")
    m1, m2, m3 = st.columns(3)
    m1.metric("Flow (mL/s)", f"{metrics['volumetric_flow_rate']:.4f}")
    m2.metric("Wall shear (1/s)", f"{metrics['wall_shear_rate']:.0f}")
    m3.metric("Specific Pressure", f"{metrics['specific_pressure']:.0f}")
    neo_card_end()

    # CSV export
    out_df = pd.DataFrame([{**rec, **metrics, HAS_CELLS: has_cells, CELL_CONC: (cells_e6 if cells_e6>0 else np.nan)}])
    st.download_button(
        "Download Recommendation (CSV)",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="bioprint_ai_recommendation.csv",
        mime="text/csv",
        key="dl_csv_main"
    )
    # PDF export
    def build_pdf_lab_sheet():
        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
        )
        styles = getSampleStyleSheet()
        story = []

        # --- Header ---
        title = "BioPrint AI — Lab Sheet (v2)"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
        story.append(Paragraph(f"Date: {ts}", styles["Normal"]))
        story.append(Spacer(1, 12))

        # --- Experiment Summary ---
        comp_text = ", ".join(selected_labels) if selected_labels else "(none specified)"
        meta_txt = meta_choice if meta_choice else "(unspecified)"
        has_cells_txt = "Yes" if has_cells else "No"
        goal_txt = "(not specified)"

        story.append(Paragraph("<b>Experiment Summary</b>", styles["Heading3"]))
        story.append(Paragraph(f"<b>Composition:</b> {comp_text}", styles["Normal"]))
        story.append(Paragraph(f"<b>Dominant Material Type:</b> {meta_txt}", styles["Normal"]))
        story.append(Paragraph(f"<b>Has Cells:</b> {has_cells_txt}", styles["Normal"]))
        story.append(Paragraph(f"<b>User Goals:</b> {goal_txt}", styles["Normal"]))
        story.append(Spacer(1, 10))

        # --- Dataset bands / stats ---
        p5 = bands["P5"].to_dict()
        p95 = bands["P95"].to_dict()
        med = {
            c: float(df_real[c].median())
            for c in ["Pressure_kPa","Temperature_C","Speed_mms","Needle_um"]
        }

        def param_comment(col, value):
            if col == "Temperature_C":
                if has_cells and value >= TEMP_CAP_IF_CELLS - 1e-9:
                    return "✅ Capped for viability"
                return "Within safe range"
            if col == "Speed_mms":
                return (
                    "Favors throughput"
                    if value > med[col]
                    else "Acceptable for resolution"
                )
            if col == "Needle_um":
                return (
                    "Fine detail"
                    if value <= med[col]
                    else "Standard"
                )
            if col == "Pressure_kPa":
                return (
                    "Within safe median range"
                    if p5[col] <= value <= p95[col]
                    else "Out-of-usual range"
                )
            return ""

        # --- Recommended Parameters Table ---
        story.append(Paragraph("<b>Recommended Parameters</b>", styles["Heading3"]))
        table_rows = [
            ["Parameter", "Recommended", "5–95% Dataset Range", "Comment"],
            [
                "Pressure (kPa)",
                f"{rec['Pressure_kPa']:.1f}",
                f"{p5['Pressure_kPa']:.0f}–{p95['Pressure_kPa']:.0f}",
                param_comment("Pressure_kPa", rec["Pressure_kPa"]),
            ],
            [
                "Temperature (°C)",
                f"{rec['Temperature_C']:.1f}",
                f"{p5['Temperature_C']:.0f}–{p95['Temperature_C']:.0f}",
                param_comment("Temperature_C", rec["Temperature_C"]),
            ],
            [
                "Speed (mm/s)",
                f"{rec['Speed_mms']:.2f}",
                f"{p5['Speed_mms']:.0f}–{p95['Speed_mms']:.0f}",
                param_comment("Speed_mms", rec["Speed_mms"]),
            ],
            [
                "Needle (µm)",
                f"{rec['Needle_um']:.0f}",
                f"{p5['Needle_um']:.0f}–{p95['Needle_um']:.0f}",
                param_comment("Needle_um", rec["Needle_um"]),
            ],
        ]
        rtab = Table(table_rows, colWidths=[120,100,140,160])
        rtab.setStyle(TableStyle([
            ("BOX", (0,0), (-1,-1), 0.7, colors.black),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (1,1), (2,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(rtab)
        story.append(Spacer(1, 10))

        # --- Derived Physics & Risks ---
        if (df_real[HAS_CELLS]==1).any():
            shear_cap = float(
                df_real.loc[df_real[HAS_CELLS]==1, "wall_shear_rate"].quantile(0.80)
            )
        else:
            shear_cap = float(np.percentile(df_real["wall_shear_rate"], 80))

        temp_bad  = (has_cells == 1 and rec["Temperature_C"] > TEMP_CAP_IF_CELLS)
        shear_bad = (has_cells == 1 and metrics["wall_shear_rate"] > shear_cap)
        clog_bad  = metrics["specific_pressure"] > np.percentile(
            df_real["specific_pressure"], 80
        )

        risk_text = [
            f"Temperature: {'High' if temp_bad else 'Safe'}",
            f"Shear: {'High' if shear_bad else 'Safe'}",
            f"Clogging Proxy: {'Elevated' if clog_bad else 'Normal'}",
        ]

        story.append(Paragraph("<b>Derived Physics & Risks</b>", styles["Heading3"]))
        ptab = Table(
            [
                ["Flow (mL/s)", "Wall Shear (1/s)", "Specific Pressure"],
                [
                    f"{metrics['volumetric_flow_rate']:.4f}",
                    f"{metrics['wall_shear_rate']:.0f}",
                    f"{metrics['specific_pressure']:.0f}",
                ],
            ],
            colWidths=[120,140,140],
        )
        ptab.setStyle(TableStyle([
            ("BOX", (0,0), (-1,-1), 0.5, colors.black),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(ptab)
        story.append(Paragraph(" → " + " | ".join(risk_text), styles["Normal"]))
        story.append(Spacer(1, 8))

        # --- Data Insight Snapshot ---
        story.append(Paragraph("<b>Data Insight Snapshot</b>", styles["Heading3"]))
        medtab = Table(
            [
                ["Feature", "Median", "5–95% Range"],
                [
                    "Pressure (kPa)",
                    f"{med['Pressure_kPa']:.1f}",
                    f"{p5['Pressure_kPa']:.0f}–{p95['Pressure_kPa']:.0f}",
                ],
                [
                    "Temperature (°C)",
                    f"{med['Temperature_C']:.1f}",
                    f"{p5['Temperature_C']:.0f}–{p95['Temperature_C']:.0f}",
                ],
                [
                    "Speed (mm/s)",
                    f"{med['Speed_mms']:.1f}",
                    f"{p5['Speed_mms']:.0f}–{p95['Speed_mms']:.0f}",
                ],
                [
                    "Needle (µm)",
                    f"{med['Needle_um']:.0f}",
                    f"{p5['Needle_um']:.0f}–{p95['Needle_um']:.0f}",
                ],
            ],
            colWidths=[140,100,160],
        )
        medtab.setStyle(TableStyle([
            ("BOX", (0,0), (-1,-1), 0.5, colors.black),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("INNERGRID", (0,0), (-1,-1), 0.25, colors.grey),
        ]))
        story.append(medtab)
        story.append(Spacer(1, 8))

        # --- Notes & Interpretations ---
        notes = []
        if has_cells and rec["Temperature_C"] <= TEMP_CAP_IF_CELLS:
            notes.append("Temperature is within safe range for cellular inks.")
        if metrics["wall_shear_rate"] > med["Pressure_kPa"]:
            notes.append("Shear slightly above median; consider a slower speed or larger needle.")
        if rec["Speed_mms"] > med["Speed_mms"]:
            notes.append("Speed favors throughput; if resolution is critical, reduce speed slightly.")
        if rec["Needle_um"] <= med["Needle_um"]:
            notes.append("Chosen needle supports finer resolution; monitor clogging risk.")
        if not notes:
            notes.append("All parameters are within typical dataset ranges.")

        story.append(Paragraph("<b>Notes & Interpretations</b>", styles["Heading3"]))
        for n in notes:
            story.append(Paragraph("• " + n, styles["Normal"]))
        story.append(Spacer(1, 12))

        # --- Footer and Authorship Credits ---
        story.append(Spacer(1, 16))
        story.append(Paragraph(
            "Generated by <b>BioPrint AI</b> — Intelligent Bioprinting Parameter Recommender",
            styles["Normal"]
        ))
        story.append(Paragraph(
            "Contact: <b>bioprintai.contact@gmail.com</b>",
            styles["Normal"]
        ))
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "<b>Developed by:</b> Ali Faiq Jari (Founder) & Dr. Bora Büyüksaraç (Co-Founder)",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))
        story.append(Paragraph(
            "<i>BioPrint AI | Less Trial and Error, More Viable Cells</i>",
            styles["Normal"]
        ))

        # --- Build & return bytes ---
        doc.build(story)
        pdf_bytes = buf.getvalue()
        buf.close()
        return pdf_bytes

    pdf_bytes = build_pdf_lab_sheet()
    st.download_button(
        "Download PDF Lab Sheet",
        data=pdf_bytes,
        file_name="bioprint_ai_lab_sheet.pdf",
        mime="application/pdf",
        key="dl_pdf_lab"
    )

with T2:
    st.subheader("Viability & Risk Checker")

    # Compose with previously calculated `rec` from T1
    rec_safe = cap_for_cells(rec, has_cells)
    Q, shear, specP = compute_physics(rec_safe)
    shear_cap = float(df_real.loc[df_real[HAS_CELLS]==1, "wall_shear_rate"].quantile(0.80)) if (df_real[HAS_CELLS]==1).any() else np.percentile(df_real["wall_shear_rate"], 80)
    temp_bad = (has_cells==1 and rec_safe["Temperature_C"]>TEMP_CAP_IF_CELLS)
    shear_bad = (has_cells==1 and shear>shear_cap)
    clog_risk = "Yellow" if specP>np.percentile(df_real["specific_pressure"],80) else "Green"

    neo_card_start("Derived Physics & Status", accent="a4")
    c1,c2,c3 = st.columns(3)
    c1.metric("Flow (mL/s)",f"{Q:.4f}")
    c2.metric("Wall shear (1/s)",f"{shear:.0f}")
    c3.metric("Specific Pressure",f"{specP:.0f}")
    neo_card_end()

    st.write(f"Temperature status: {'🔴' if temp_bad else '🟢'} — cap {TEMP_CAP_IF_CELLS}°C when cells.")
    st.write(f"Shear status: {'🔴' if shear_bad else '🟢'} — 80th pct cap from cellular data.")
    st.write(f"Clogging proxy: {'🟡' if clog_risk=='Yellow' else '🟢'}")

with T3:
    st.markdown("## 🔬 Knowledge Hub / Learn")

    # 🟦 Neo‑Brutalism badge bar
    st.markdown(
        """
        <div class="nb-badges" style="margin-bottom: 1rem;">
          <div class="nb-badge a1">Science</div>
          <div class="nb-badge a2">AI</div>
          <div class="nb-badge a3">Bioinks</div>
          <div class="nb-badge a4">Optimization</div>
          <div class="nb-badge a5">Viability</div>
        </div>
        """, unsafe_allow_html=True
    )

    # Intro blurb
    st.markdown("""
Welcome to the **Knowledge Hub** — a curated space to explore the science, technology, and impact behind **BioPrint AI**. Whether you're a researcher, clinician, or curious innovator, this section offers clear, evidence-backed insights into bioprinting and AI-powered optimization.

---
""")

    with st.expander("🎯 What is BioPrint AI?"):
        st.markdown("""
**BioPrint AI** is an intelligent recommendation system designed to optimize key bioprinting parameters:

- 🧪 **Pressure** (kPa)
- 🌡️ **Temperature** (°C)
- ⚡ **Print Speed** (mm/s)
- 🧵 **Needle Diameter** (µm)

Based on the bioink composition and cellular context, our engine reduces **trial-and-error**, enhances **cell viability**, and accelerates **reproducible tissue fabrication**.
        """)

    with st.expander("🧫 What are Bioinks?"):
        st.markdown("""
**Bioinks** are hydrogel-based materials that encapsulate living cells during 3D bioprinting. Their properties directly affect:

- Cell viability 🧬
- Structural integrity 🧱
- Print resolution 📐

Common bioinks include:
- Alginate-Gelatin
- GelMA
- Fibrinogen + Collagen
- ECM-derived blends

Factors like **viscosity**, **crosslinking behavior**, and **thermal sensitivity** must be tuned carefully for each bioprinting context.
        """)

    with st.expander("🤖 Why Use AI in Bioprinting?"):
        st.markdown("""
Traditional bioprinting involves **trial-and-error** with dozens of variables. This leads to:

- Wasted materials 💸
- Inconsistent results ❌
- Poor reproducibility 🔁

**AI-based modeling** enables:

- 📊 **Data-driven predictions** of optimal parameters
- ⏱️ **Reduced print failures**
- 💡 **Accelerated experimental cycles**

Our model learns from real-world and synthetic datasets, creating a smarter path to functional tissue engineering.
        """)

    with st.expander("📊 How Our Recommendation Engine Works"):
        st.markdown("""
Our AI uses a **Nearest Neighbors** model trained on curated datasets of bioink compositions and printing outcomes. The steps:

1. 🔍 Analyze user inputs (bioink + cell type + desired output)
2. 🤝 Match to similar known examples
3. 🎯 Return optimal Pressure, Speed, Temperature, Needle

All results are:
- Based on **biophysical similarity**
- Evaluated using **viability-focused metrics**
- Generated in **real-time**

> "Less Trial and Error, More Viable Cells."
        """)

    with st.expander("🧠 Bioprinting Glossary"):
        st.markdown("""
- **Bioink**: Printable substance made of biomaterials and live cells.
- **Shear Stress**: Force affecting cells as they're extruded; too high = cell death.
- **Crosslinking**: Process to stabilize printed structures (via light, ions, heat).
- **Viability**: Percentage of cells alive post-printing.
- **Extrusion Bioprinting**: Most common technique; pushes bioink through a nozzle.
        """)

with T4:
    st.subheader("🤝 Feedback & Collaboration")
    st.caption("We value your insights. Share ideas, report issues, or propose collaborations.")

    # Neo-Brutalist Card – Feedback
    neo_card_start("📩 Send Us Feedback", accent="a2")
    st.markdown("""
We’re building **BioPrint AI** with researchers, not just for them.
Whether you're experimenting with bioinks, teaching bioprinting, or optimizing protocols — your thoughts matter.

🧾 **Share your feedback, ideas, or bug reports via email**:

📧 [bioprintai.contact@gmail.com](mailto:bioprintai.contact@gmail.com)

Or just include:
- 👤 Your name & lab (optional)
- 🧪 How you’re using BioPrint AI
- 💡 What works / what doesn't
- 🧠 Any ideas for new features

""")
    st.success("💬 Just send us an email")
    neo_card_end()

    # Neo-Brutalist Card – Collaboration
    neo_card_start("🤓 Contribute & Collaborate", accent="a3")
    st.markdown("""
If you’re interested in sharing anonymized print logs, real-world datasets, or collaborating on optimization strategies, let’s team up.

📮 **Reach out at:**
📧 [bioprintai.contact@gmail.com](mailto:bioprintai.contact@gmail.com)

We prioritize ethical, transparent, and opt-in scientific collaboration.
    """)
    neo_card_end()






with T5:
    st.subheader("📘 About BioPrint AI")

    # --- Mission Card ---
    neo_card_start("🎯 Mission & Purpose", accent="a1")
    st.markdown("""
**BioPrint-AI** is a smart recommender system that predicts **bioprinting parameters** —
**Pressure (kPa)**, **Temperature (°C)**, **Speed (mm/s)**, and **Needle Size (µm)** — from bioink composition.

The project aims to:
- 🔁 **Minimize trial-and-error** in lab setups
- 🧬 **Increase cell viability** and reproducibility
- 📈 **Accelerate** data-driven biofabrication workflows

It combines **machine learning**, **composition similarity search**, and **physics-aware logic**
to deliver lab-ready, explainable predictions.
    """)
    neo_card_end()

    # --- Founder Card ---
    neo_card_start("🧬 Founder and Developer", accent="a2")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.image("assets/ali_portrait_bw.png", width=140)
    with c2:
        st.markdown("""
**👤 Name:** Ali Faiq Jari Alsaadawi
**🎓 Field:** Biomedical Engineering (Bahçeşehir University, Istanbul)
**💼 Role:** Founder, Developer & Researcher
**🧠 Focus:** AI-powered diagnostics, biomechanics, digital health tools
**📧 Contact:** [alifaiqjari@gmail.com](mailto:alifaiqjari@gmail.com)
        """)
    neo_card_end()

    # --- Co-Founder (Supervisor) Card ---
    neo_card_start("👥 Co‑Founder", accent="a3")
    c3, c4 = st.columns([1, 3])
    with c3:
        st.image("assets/supervisor_portrait_bw.png", width=140)
    with c4:
        st.markdown("""
**👨‍🏫 Name:** Dr. Öğr. Üyesi Bora Büyüksaraç
**🏛️ Institution:** Bahçeşehir University — Faculty of Engineering and Natural Sciences
**🏢 Department:** Biomedical Engineering
**📧 Contact:** [bora.buyuksarac@eng.bau.edu.tr](mailto:bora.buyuksarac@eng.bau.edu.tr)

**🔬 Research Focus:**
MRI • Medical Image Processing • Perfusion Imaging • 3D Modeling
        """)
    neo_card_end()

    # --- Tech Stack Card ---
    neo_card_start("🧩 Tech & Dataset", accent="a4")
    st.markdown("""
**🛠️ Tech Stack:**
Python | Streamlit | Scikit-learn | Pandas | Plotly | NumPy

**📊 Dataset:**
- 807 curated empirical records
- +10 000 physics-augmented synthetic data points

**🎨 UI Language:**
Neo‑Brutalism — bold geometry, visible structure, and vivid accent colors.
    """)
    neo_card_end()



