
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
    page_title="IoT IDS — RT-IoT2022",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CAT_COLS = ["proto", "service"]
MODEL_DIR = Path(__file__).parent / "models"

# ============================================================
# Load model (cached — sekali saja per sesi)
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

# Ambil daftar kolom wajib dari model XGB (pipeline sudah menyimpan urutan fitur)
XGB_FEATURES = list(xgb_model.feature_names_in_)
REQUIRED_COLS = XGB_FEATURES  # 83 kolom

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
# Header
# ============================================================
st.title("🛡️ IoT Intrusion Detection System")
st.caption("Deteksi serangan siber pada trafik jaringan IoT menggunakan model ML terlatih.")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model", model_choice)
with col2:
    st.metric("Threshold", f"{threshold:.2f}")
with col3:
    eval_summary = meta.get("evaluation_summary", {})
    f1_key = "catboost" if model_choice == "CatBoost" else "xgboost"
    f1 = eval_summary.get(f1_key, {}).get("f1_macro")
    st.metric("Lockbox Macro-F1", f"{f1:.4f}" if f1 else "N/A")

st.markdown("---")

# ============================================================
# Mode input: upload atau sample
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
# Preview
# ============================================================
if df_input is not None:
    st.markdown("### 📋 Preview Data")
    st.dataframe(df_input.head(10), use_container_width=True)
    
    # Persiapkan fitur
    df_features = df_input.copy()
    
    # Drop kolom terlarang kalau ada
    drop_cols = [c for c in ["Unnamed: 0", "Attack_type", "label"] if c in df_features.columns]
    if drop_cols:
        st.info(f"ℹ️ Kolom berikut otomatis di-drop (tidak boleh jadi fitur): {drop_cols}")
        df_features = df_features.drop(columns=drop_cols)
    
    # Validasi kolom
    missing = [c for c in REQUIRED_COLS if c not in df_features.columns]
    if missing:
        st.error(f"❌ Kolom wajib TIDAK ada di file: {missing[:10]}{' ...' if len(missing)>10 else ''}")
        st.stop()
    
    # Urutkan kolom sesuai training
    df_features = df_features[REQUIRED_COLS]
    
    # Casting kategorikal
    for c in CAT_COLS:
        df_features[c] = df_features[c].astype(str)
    
    # ============================================================
    # Prediksi
    # ============================================================
    st.markdown("---")
    if st.button("🔮 Prediksi", type="primary", use_container_width=True):
        model = cb_model if model_choice == "CatBoost" else xgb_model
        
        with st.spinner(f"⏳ Prediksi {len(df_features):,} baris dengan {model_choice} ..."):
            proba = model.predict_proba(df_features)[:, 1]
            pred = (proba >= threshold).astype(int)
        
        # Hasil
        df_out = df_input.copy()
        df_out["prediction"] = pred
        df_out["prediction_label"] = df_out["prediction"].map({0: "🟢 Normal", 1: "🔴 Attack"})
        df_out["attack_probability"] = proba.round(6)
        
        # Ringkasan
        st.markdown("### 📊 Hasil Prediksi")
        
        n_normal = int((pred == 0).sum())
        n_attack = int((pred == 1).sum())
        pct_attack = n_attack / len(pred) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total baris", f"{len(pred):,}")
        c2.metric("🟢 Normal", f"{n_normal:,}", f"{100-pct_attack:.2f}%")
        c3.metric("🔴 Attack", f"{n_attack:,}", f"{pct_attack:.2f}%")
        
        # Tabel hasil
        st.markdown("### 🗂️ Detail per Baris")
        show_cols = ["prediction_label", "attack_probability"] + [
            c for c in df_input.columns[:5] if c not in ["prediction_label", "attack_probability"]
        ]
        st.dataframe(
            df_out[show_cols].head(100),
            use_container_width=True,
            height=350,
        )
        
        # Download
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
    "🛡️ IoT IDS · RT-IoT2022 · "
    f"Model: {model_choice} · "
    "Dibangun dengan Streamlit + XGBoost + CatBoost"
)
