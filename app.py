# app.py - BigMart Sales Prediction (production-ready)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import traceback
from datetime import datetime

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="BigMart Sales Prediction",
    page_icon="🛒",
    layout="wide",
)

# ---------------------------
# Polished CSS theme
# ---------------------------
st.markdown("""
<style>
/* App background & base */
.stApp {
    background: linear-gradient(180deg, #f9fafb 0%, #edf1f4 100%);
    color: #0f172a;
    font-family: Inter, "Segoe UI", Roboto, system-ui, -apple-system, "Helvetica Neue", Arial;
}

/* Header & title */
h1, h2, h3 { text-align: center; font-weight: 800; color: #0a4b78; margin: 0; }
h1 { font-size: 2.0rem; margin-top: 6px; }

/* Cards */
.card { background: #ffffff; padding: 18px; border-radius: 12px; box-shadow: 0 6px 20px rgba(13,30,49,0.06); }

/* Inputs styling (best-effort selectors) */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] select,
textarea {
    border-radius: 10px !important;
    border: 1px solid #e6eef6 !important;
    background-color: #ffffff !important;
    padding: 12px !important;
    color: #0f172a !important;
}

/* Slider */
[data-baseweb="slider"] { margin-top: 8px; }

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #0a4b78, #2563eb);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    font-size: 16px;
    padding: 10px 24px;
    transition: transform 0.18s ease;
}
.stButton>button:hover { transform: scale(1.03); }

/* Result card */
.result-card {
    background: #ffffff;
    border-radius: 14px;
    box-shadow: 0 10px 30px rgba(2,6,23,0.06);
    padding: 20px;
    margin-top: 12px;
    text-align: center;
    animation: fadeIn 0.45s ease-in-out;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px);} to { opacity:1; transform: translateY(0);} }

/* Footer */
.dev-footer { text-align:center; margin-top: 18px; color:#475569; font-size:13px; }
.dev-footer a { color: #0a4b78; text-decoration:none; }
.dev-footer a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Utility: model loading (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(path="bigmart_best_model.pkl"):
    """
    Load a variety of possible pickle structures:
    - plain model object
    - (model, sklearn_version)
    - dict {'model':..., 'preprocessor':..., 'metadata':...}
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    model = None
    preprocessor = None
    metadata = {}

    if isinstance(obj, tuple) and len(obj) == 2:
        model, metadata["sklearn_version"] = obj
    elif isinstance(obj, dict):
        model = obj.get("model") or obj.get("estimator")
        preprocessor = obj.get("preprocessor")
        metadata = obj.get("metadata", {})
    else:
        model = obj

    return {"model": model, "preprocessor": preprocessor, "metadata": metadata}


def df_to_download_link(df: pd.DataFrame, name="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}"


# ---------------------------
# Load model - graceful error messages
# ---------------------------
with st.spinner("Loading model..."):
    try:
        bundle = load_model()
        model = bundle["model"]
        preprocessor = bundle.get("preprocessor")
        meta = bundle.get("metadata", {})
        sklearn_version = meta.get("sklearn_version", "unknown")
    except FileNotFoundError:
        st.error("Model file `bigmart_best_model.pkl` not found. Upload it to the app folder.")
        st.stop()
    except Exception as e:
        st.error("Failed to load model. See error below.")
        st.exception(e)
        st.stop()

# ---------------------------
# Header
# ---------------------------
st.title("🛒 BigMart Sales Prediction")
st.markdown(f"<div style='text-align:center;color:#475569;'>Using model: `<b>{getattr(model,'__class__', type(model)).__name__}</b>` • scikit-learn `{sklearn_version}`</div>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Controls")
    use_sample = st.checkbox("Use sample single input", value=True)
    uploaded_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
    show_raw = st.checkbox("Show raw input dataframe", value=False)
    enable_download = st.checkbox("Enable CSV download of predictions", value=True)
    st.markdown("---")
    st.markdown("**Developer**: Tejas Gholap")
    st.markdown("[GitHub](https://github.com/tejasgholap45) • [LinkedIn](https://linkedin.com/in/tejas-gholap-bb3417300)")

# ---------------------------
# Build single-row input form
# ---------------------------
def build_single_input():
    c1, c2 = st.columns(2)
    with c1:
        Item_Identifier = st.text_input("Item Identifier", value="FDA15", help="Product code")
        Item_Weight = st.number_input("Item Weight (kg)", min_value=0.0, value=12.5, format="%.2f")
        Item_Fat_Content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"], index=0)
        Item_Visibility = st.number_input("Item Visibility (0-1)", min_value=0.0, max_value=1.0, value=0.065, format="%.5f")
        Item_Type = st.selectbox("Item Type", [
            "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
            "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", 
            "Health and Hygiene", "Hard Drinks", "Canned", "Breads", 
            "Starchy Foods", "Others", "Seafood"
        ])
    with c2:
        Item_MRP = st.number_input("Item MRP (₹)", min_value=0.0, value=150.0, format="%.2f")
        Outlet_Identifier = st.selectbox("Outlet Identifier", [
            "OUT027", "OUT013", "OUT049", "OUT035", "OUT046", 
            "OUT017", "OUT045", "OUT018", "OUT019", "OUT010"
        ])
        Outlet_Size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
        Outlet_Location_Type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        Outlet_Type = st.selectbox("Outlet Type", [
            "Supermarket Type1", "Supermarket Type2", 
            "Supermarket Type3", "Grocery Store"
        ])
        Outlet_Age = st.slider("Outlet Age (Years)", 0, 60, 15)

    row = {
        "Item_Identifier": Item_Identifier,
        "Item_Weight": float(Item_Weight),
        "Item_Fat_Content": Item_Fat_Content,
        "Item_Visibility": float(Item_Visibility),
        "Item_Type": Item_Type,
        "Item_MRP": float(Item_MRP),
        "Outlet_Identifier": Outlet_Identifier,
        "Outlet_Size": Outlet_Size,
        "Outlet_Location_Type": Outlet_Location_Type,
        "Outlet_Type": Outlet_Type,
        "Outlet_Age": int(Outlet_Age)
    }
    return pd.DataFrame([row])


# ---------------------------
# Prepare input dataframe (single or batch)
# ---------------------------
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(input_df)} rows for batch prediction")
    except Exception as e:
        st.error("Failed to read CSV. Ensure it is valid and has expected columns.")
        st.exception(e)
        st.stop()
else:
    if use_sample:
        input_df = build_single_input()
    else:
        input_df = build_single_input()

if show_raw:
    with st.expander("📋 Raw input data"):
        st.dataframe(input_df)

# ---------------------------
# Prediction logic & UI
# ---------------------------
predict_btn = st.button("🔮 Predict Sales")

if predict_btn:
    with st.spinner("Running prediction..."):
        try:
            X = input_df.copy()

            # If user saved a preprocessing pipeline, use it
            if preprocessor is not None:
                try:
                    X_trans = preprocessor.transform(X)
                except Exception:
                    # fallback: some preprocessors expect fit_transform during mismatch (not ideal)
                    X_trans = preprocessor.fit_transform(X)
                preds = model.predict(X_trans)
            else:
                # If the model expects engineered features, the user should have saved a pipeline.
                preds = model.predict(X)

            results = input_df.copy()
            results["Predicted_Sales"] = np.round(preds.astype(float), 2)

            # Result display
            st.markdown(f"""
                <div class="result-card">
                    <h2>📈 Predicted Sales</h2>
                    <h1 style="color:#0a4b78;margin:6px 0;">₹{results["Predicted_Sales"].mean():,.2f}</h1>
                    <div style="color:#6b7280;font-size:14px;">Estimated by <b>{getattr(model,'__class__', type(model)).__name__}</b> • scikit-learn {sklearn_version}</div>
                </div>
            """, unsafe_allow_html=True)

            # Show detailed table and metric
            cA, cB = st.columns([2, 1])
            with cA:
                st.markdown("#### Detailed results")
                st.dataframe(results)
            with cB:
                st.metric("Average Predicted Sales (₹)", f"{results['Predicted_Sales'].mean():,.2f}")
                st.markdown("#### Quick actions")
                if enable_download:
                    csv_bytes = results.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️ Download CSV",
                        data=csv_bytes,
                        file_name=f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

            # Lightweight explainability: feature importances / coefficients if available
            try:
                fi = None
                if hasattr(model, "feature_importances_"):
                    fi = model.feature_importances_
                elif hasattr(model, "coef_"):
                    fi = np.abs(model.coef_).ravel()

                if fi is not None:
                    try:
                        feature_names = (preprocessor.get_feature_names_out() if preprocessor is not None else X.columns)
                    except Exception:
                        feature_names = X.columns
                    fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
                    fi_df = fi_df.sort_values("importance", ascending=False).head(10).reset_index(drop=True)
                    st.markdown("---")
                    st.markdown("#### Top feature importances (approx)")
                    st.table(fi_df.style.hide_index())
            except Exception:
                # not critical - continue
                pass

        except Exception as e:
            st.error("Prediction failed. See details below.")
            st.exception(traceback.format_exc())

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.markdown("""
<div class='dev-footer'>
  Built with ❤️ by <b>Tejas Gholap</b> ·
  <a href='https://github.com/tejasgholap45' target='_blank'>GitHub</a> ·
  <a href='https://linkedin.com/in/tejas-gholap-bb3417300' target='_blank'>LinkedIn</a><br>
  <span style='font-size:12px;'>v1.0 • © 2025 BigMart ML Project</span>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Sample CSV download (sidebar button)
# ---------------------------
if st.sidebar.button("Download sample CSV"):
    sample_df = build_single_input()
    st.sidebar.download_button("Download sample.csv", data=sample_df.to_csv(index=False).encode(), file_name="bigmart_sample_input.csv", mime="text/csv")
