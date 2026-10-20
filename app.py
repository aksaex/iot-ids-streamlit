import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime

# ============================================================
# Konfigurasi halaman
# ============================================================
st.set_page_config(
    page_title="SENTINEL — IoT Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS TAKTIS — Dark Command Center
# ============================================================
st.markdown("""
<style>
/* 1. Sembunyikan header, footer, menu bawaan Streamlit */
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* 2. Judul utama — efek neon */
h1 {
    color: #00ff9d !important;
    text-shadow: 0 0 12px rgba(0, 255, 157, 0.4);
    letter-spacing: 1.5px;
    font-weight: 700 !important;
}

/* 3. Subjudul seksi */
h2, h3 {
    color: #00ff9d !important;
    letter-spacing: 0.8px;
}

/* 4. Kartu metrik gaya HUD */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #0f172a, #111827);
    border: 1px solid #1f2937;
    border-left: 3px solid #00ff9d;
    border-radius: 8px;
    padding: 16px 20px;
    box-shadow: 0 0 12px rgba(0, 255, 157, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.03);
}
div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-size: 0.75rem !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #00ff9d !important;
    font-weight: 700;
    text-shadow: 0 0 8px rgba(0, 255, 157, 0.3);
}

/* 5. Tombol primer — glow hijau */
div.stButton > button[kind="primary"] {
    background: #00ff9d !important;
    color: #0a0e17 !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    padding: 12px 24px;
    box-shadow: 0 0 14px rgba(0, 255, 157, 0.35);
    transition: all 0.2s ease;
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 24px rgba(0, 255, 157, 0.65);
    transform: translateY(-1px);
}

/* 6. Tombol sekunder */
div.stButton > button {
    border: 1px solid #00ff9d;
    color: #00ff9d;
    background: transparent;
    letter-spacing: 0.8px;
}
div.stButton > button:hover {
    background: rgba(0, 255, 157, 0.1);
    color: #00ff9d;
}

/* 7. Dataframe menyatu dengan latar gelap */
div[data-testid="stDataFrame"] {
    border: 1px solid #1f2937;
    border-radius: 6px;
    overflow: hidden;
}

/* 8. Slider — aksen neon */
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #00ff9d !important;
    box-shadow: 0 0 10px rgba(0, 255, 157, 0.6) !important;
}

/* 9. Sidebar */
section[data-testid="stSidebar"] {
    border-right: 1px solid #1f2937;
    background: #0d1117;
}

/* 10. Divider */
hr { border-color: #1f2937 !important; }

/* 11. Caption */
div[data-testid="stCaptionContainer"], .stCaption, small {
    color: #64748b !important;
    letter-spacing: 0.5px;
}

/* 12. Alert */
div[data-testid="stAlert"] {
    border-radius: 6px;
    border-left-width: 4px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Konstanta
# ============================================================
CAT_COLS = ["proto", "service"]
MODEL_DIR = Path(__file__).parent / "models"

# ============================================================
# Load model (cached)
# ============================================================
@st.cache_resource(show_spinner="🔄 Memuat model ...")
def load_models():
    xgb = joblib.load(MODEL_DIR / "xgboost_best.pkl")
    cb  = joblib.load(MODEL_DIR / "catboost_best.pkl")
    meta = {}
    for name in ["xgboost_best_params", "catboost_best_params", "evaluation_summary"]:
        p = MODEL_DIR / f"{name}.json"
        if p.exists():
            with open(p) as f:
                meta[name] = json.load(f)
    return xgb, cb, meta

try:
    xgb_model, cb_model, meta = load_models()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

XGB_FEATURES = list(xgb_model.feature_names_in_)
REQUIRED_COLS = XGB_FEATURES

# ============================================================
# Sidebar — pengaturan
# ============================================================
st.sidebar.title("⚙️ Pengaturan")

model_choice = st.sidebar.radio(
    "Model",
    ["CatBoost", "XGBoost"],
    index=0,
    help="CatBoost: native categorical. XGBoost: One-Hot Encoding."
)

threshold = st.sidebar.slider(
    "Threshold (probabilitas → Attack)",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    help="Turunkan untuk recall Attack lebih tinggi (lebih sensitif). "
         "Naikkan untuk mengurangi false alarm."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Info Model")

if model_choice == "CatBoost":
    p = meta.get("catboost_best_params", {})
    st.sidebar.caption(f"CV Macro-F1: **{p.get('cv_macro_f1', 'N/A'):.4f}**")
elif model_choice == "XGBoost":
    p = meta.get("xgboost_best_params", {})
    st.sidebar.caption(f"CV Macro-F1: **{p.get('cv_macro_f1', 'N/A'):.4f}**")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**RT-IoT2022** — Deteksi Intrusi IoT\n\n"
    "Skripsi — Model ML untuk klasifikasi trafik benign vs attack."
)

# ============================================================
# Header + HUD Status
# ============================================================
st.title("🛡️ SENTINEL — IoT Threat Detection")
st.caption("TACTICAL COMMAND CENTER · RT-IoT2022 · Real-time Traffic Classification")

eval_summary = meta.get("evaluation_summary", {})
f1_key = "catboost" if model_choice == "CatBoost" else "xgboost"
f1 = eval_summary.get(f1_key, {}).get("f1_macro")
f1_str = f"{f1:.4f}" if f1 else "N/A"

st.markdown(f"""
<div style="
    display: flex;
    gap: 28px;
    padding: 10px 16px;
    background: linear-gradient(90deg, rgba(0,255,157,0.10), transparent);
    border-left: 3px solid #00ff9d;
    border-radius: 4px;
    margin: 8px 0 20px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 1px;
">
    <span style="color:#00ff9d;">● SYSTEM ONLINE</span>
    <span style="color:#94a3b8;">MODEL: <b style="color:#00ff9d;">{model_choice.upper()}</b></span>
    <span style="color:#94a3b8;">THRESHOLD: <b style="color:#00ff9d;">{threshold:.2f}</b></span>
    <span style="color:#94a3b8;">LOCKBOX F1: <b style="color:#00ff9d;">{f1_str}</b></span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", model_choice)
with col2:
    st.metric("Threshold", f"{threshold:.2f}")
with col3:
    st.metric("Lockbox Macro-F1", f1_str)

st.markdown("---")

# ============================================================
# Mode input
# ============================================================
input_mode = st.radio(
    "Sumber data:",
    ["📁 Upload CSV", "🎲 Sample acak (untuk demo)"],
    horizontal=True,
)

df_input = None

if input_mode.startswith("📁"):
    uploaded = st.file_uploader(
        "Upload file CSV berisi trafik IoT",
        type=["csv"],
        help=f"File harus memiliki minimal kolom: {', '.join(REQUIRED_COLS[:5])} ... (total {len(REQUIRED_COLS)} kolom)"
    )
    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
            st.success(f"✅ File ter-upload: {df_input.shape[0]:,} baris × {df_input.shape[1]} kolom")
        except Exception as e:
            st.error(f"❌ Gagal baca CSV: {e}")
else:
    st.info("Klik tombol di bawah untuk mengambil 20 baris acak dari lockbox sebagai demo.")
    if st.button("🎲 Ambil sample"):
        lockbox_path = Path("/content/drive/MyDrive/iot_ids_skripsi/data/lockbox_test.csv")
        local_path = Path(__file__).parent / "sample_input.csv"
        if local_path.exists():
            df_sample = pd.read_csv(local_path)
            df_input = df_sample.head(20) if len(df_sample) >= 20 else df_sample
            st.session_state["df_input"] = df_input
            st.success(f"✅ Sample dimuat: {df_input.shape}")
        elif lockbox_path.exists():
            df_sample = pd.read_csv(lockbox_path)
            df_input = df_sample.sample(20, random_state=42)
            st.session_state["df_input"] = df_input
            st.success(f"✅ Sample dimuat: {df_input.shape}")
        else:
            st.error("❌ File sample tidak tersedia. Upload CSV manual atau taruh `sample_input.csv` di folder repo.")

    if "df_input" in st.session_state and df_input is None:
        df_input = st.session_state["df_input"]

# ============================================================
# Preview + Prediksi
# ============================================================
if df_input is not None:
    st.markdown("### 📋 Preview Data")
    st.dataframe(df_input.head(10), use_container_width=True)

    df_features = df_input.copy()
    drop_cols = [c for c in ["Unnamed: 0", "Attack_type", "label"] if c in df_features.columns]
    if drop_cols:
        st.info(f"ℹ️ Kolom berikut otomatis di-drop (tidak boleh jadi fitur): {drop_cols}")
        df_features = df_features.drop(columns=drop_cols)

    missing = [c for c in REQUIRED_COLS if c not in df_features.columns]
    if missing:
        st.error(f"❌ Kolom wajib TIDAK ada di file: {missing[:10]}{' ...' if len(missing)>10 else ''}")
        st.stop()

    df_features = df_features[REQUIRED_COLS]
    for c in CAT_COLS:
        df_features[c] = df_features[c].astype(str)

    st.markdown("---")
    if st.button("🔮 Prediksi", type="primary", use_container_width=True):
        model = cb_model if model_choice == "CatBoost" else xgb_model

        with st.spinner(f"⏳ Prediksi {len(df_features):,} baris dengan {model_choice} ..."):
            proba = model.predict_proba(df_features)[:, 1]
            pred = (proba >= threshold).astype(int)

        df_out = df_input.copy()
        df_out["prediction"] = pred
        df_out["prediction_label"] = df_out["prediction"].map({0: "🟢 Normal", 1: "🔴 Attack"})
        df_out["attack_probability"] = proba.round(6)

        st.markdown("### 📊 Hasil Prediksi")

        n_normal = int((pred == 0).sum())
        n_attack = int((pred == 1).sum())
        pct_attack = n_attack / len(pred) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Total baris", f"{len(pred):,}")
        c2.metric("🟢 Normal", f"{n_normal:,}", f"{100-pct_attack:.2f}%")
        c3.metric("🔴 Attack", f"{n_attack:,}", f"{pct_attack:.2f}%")

        st.markdown("### 🗂️ Detail per Baris")
        show_cols = ["prediction_label", "attack_probability"] + [
            c for c in df_input.columns[:5] if c not in ["prediction_label", "attack_probability"]
        ]
        st.dataframe(
            df_out[show_cols].head(100),
            use_container_width=True,
            height=350,
        )

        st.markdown("### 💾 Download Hasil")
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV (hasil + probabilitas)",
            data=csv_bytes,
            file_name=f"predictions_{model_choice.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.success(f"✅ Prediksi selesai. Threshold = {threshold:.2f}")

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "🛡️ SENTINEL · RT-IoT2022 · "
    f"Model: {model_choice} · "
    "Dibangun dengan Streamlit + XGBoost + CatBoost"
)
