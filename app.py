import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import base64
import textwrap
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
# Custom CSS (clean, accessible)
# ---------------------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%); }
    .card { background: #ffffff; padding: 18px; border-radius: 12px; box-shadow: 0 6px 20px rgba(13,30,49,0.06); }
    .accent { color: #0a4b78; font-weight:700 }
    .muted { color: #6b7280 }
    .small { font-size: 12px }
    .center { text-align:center }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utilities
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str = "bigmart_best_model.pkl"):
    """Load a pickle that may contain either:
    - a single model object
    - a tuple (model, sklearn_version)
    - a dict with {'model':..., 'preprocessor':..., 'metadata':...}
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # normalize structure
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


def df_download_link(df: pd.DataFrame, filename: str = "predictions.csv") -> str:
    """Return a link to download a dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"data:file/csv;base64,{b64}"
    return href


# ---------------------------
# Load model
# ---------------------------
with st.spinner("Loading model..."):
    try:
        model_bundle = load_model()
        model = model_bundle["model"]
        preprocessor = model_bundle.get("preprocessor")
        meta = model_bundle.get("metadata", {})
        sklearn_version = meta.get("sklearn_version", "unknown")
    except FileNotFoundError:
        st.error("Model file `bigmart_best_model.pkl` not found in app folder. Upload the file or place it next to this script.")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

# ---------------------------
# App header
# ---------------------------
st.title("🛒 BigMart Sales Prediction")
st.markdown(f"**Model:** `{getattr(model, '__class__', type(model)).__name__}` • scikit-learn `{sklearn_version}`")
st.write("---")

# ---------------------------
# Sidebar - Inputs & Uploads
# ---------------------------
with st.sidebar:
    st.header("Input / Controls")

    use_sample = st.checkbox("Use sample input (recommended)", value=True)
    upload_file = st.file_uploader("Or upload CSV for batch predictions", type=["csv"])

    st.markdown("---")
    st.markdown("### Model & App options")
    show_raw = st.checkbox("Show raw input dataframe", value=False)
    enable_batch_download = st.checkbox("Enable CSV download of results", value=True)

    st.markdown("---")
    st.markdown("Developed by **Tejas Gholap** — [GitHub](https://github.com/tejasgholap45) | [LinkedIn](https://linkedin.com/in/tejas-gholap-bb3417300)")

# ---------------------------
# Helper: build single-row input
# ---------------------------
def build_single_input():
    # We'll show inputs in two columns for visual balance
    c1, c2 = st.columns(2)
    with c1:
        Item_Identifier = st.text_input("Item Identifier", value="FDA15", help="Unique product code")
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
        Outlet_Size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])  # keep original choices
        Outlet_Location_Type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"]) 
        Outlet_Type = st.selectbox("Outlet Type", [
            "Supermarket Type1", "Supermarket Type2", 
            "Supermarket Type3", "Grocery Store"
        ])
        Outlet_Age = st.slider("Outlet Age (Years)", 0, 60, 15)

    data = {
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
    return pd.DataFrame([data])

# ---------------------------
# Prepare input dataframe
# ---------------------------
if upload_file is not None:
    try:
        input_df = pd.read_csv(upload_file)
        st.success(f"Loaded {len(input_df)} rows for batch prediction")
    except Exception as e:
        st.error("Failed to read uploaded CSV. Make sure it is comma-separated and has correct columns.")
        st.stop()
else:
    if use_sample:
        input_df = build_single_input()
    else:
        st.info("Using blank sample input. Fill the fields on the left and press Predict.")
        input_df = build_single_input()

if show_raw:
    with st.expander("📋 Raw input data"):
        st.dataframe(input_df)

# ---------------------------
# Prediction
# ---------------------------
predict_button = st.button("Predict Sales", key="predict")

if predict_button:
    with st.spinner("Running prediction..."):
        try:
            X = input_df.copy()

            # If a preprocessor was saved with the model, use it
            if preprocessor is not None:
                try:
                    X_trans = preprocessor.transform(X)
                except Exception:
                    # Some preprocessors expect columns in a different order — try fit_transform fallback
                    X_trans = preprocessor.fit_transform(X)
                preds = model.predict(X_trans)
            else:
                # If model expects specific feature order, user should export a pipeline. We'll pass raw X.
                preds = model.predict(X)

            results = input_df.copy()
            results["Predicted_Sales"] = np.round(preds, 2)

            # show results
            st.success("✅ Prediction completed")

            cola, colb = st.columns([2,1])
            with cola:
                st.markdown("#### Prediction Result")
                st.dataframe(results)

            with colb:
                mean_pred = results["Predicted_Sales"].mean()
                st.metric(label="Average Predicted Sales (₹)", value=f"{mean_pred:,.2f}")
                st.markdown("\n")

            # enable CSV download
            if enable_batch_download:
                csv_bytes = results.to_csv(index=False).encode()
                st.download_button("Download predictions as CSV", data=csv_bytes, file_name=f"predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

            # small explainability: if the model has feature_importances_ or coef_
            try:
                fi = None
                if hasattr(model, "feature_importances_"):
                    fi = model.feature_importances_
                elif hasattr(model, "coef_"):
                    fi = np.abs(model.coef_).ravel()

                if fi is not None:
                    # try to recover feature names
                    try:
                        feature_names = (preprocessor.get_feature_names_out() if preprocessor is not None else X.columns)
                    except Exception:
                        feature_names = X.columns

                    fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
                    fi_df = fi_df.sort_values("importance", ascending=False).head(10)
                    st.markdown("---")
                    st.markdown("#### Top feature importances (approx)")
                    st.table(fi_df.style.hide_index())

            except Exception:
                # not critical
                pass

        except Exception as e:
            st.exception(e)

# ---------------------------
# Footer / Extra
# ---------------------------
st.write("---")
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("Built with ❤️ — improvements: input validation, batch upload/download, preprocessing compatibility, simple explainability, and robust model loading.")
with col2:
    st.markdown("*v1.0*")
    st.markdown(f"Model: `{getattr(model, '__class__', type(model)).__name__}`")


# Helpful sample CSV download
if st.sidebar.button("Download sample CSV"):
    sample = build_single_input()
    st.download_button("Download sample.csv", data=sample.to_csv(index=False).encode(), file_name="bigmart_sample_input
