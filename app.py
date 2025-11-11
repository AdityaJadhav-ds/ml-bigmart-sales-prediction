import streamlit as st
import pandas as pd
import pickle

# === Load Model and Version Info ===
with open("bigmart_best_model.pkl", "rb") as f:
    model, sklearn_version = pickle.load(f)

# === Page Config ===
st.set_page_config(page_title="BigMart Sales Prediction", page_icon="🛒", layout="wide")

# === CUSTOM CSS ===
st.markdown("""
<style>
.stApp { background: #f1f5f9; }

h1 {
    text-align: center;
    color: #0a4b78;
    font-weight: 800;
}

.info-bar {
    text-align:center;
    padding: 6px 0;
    margin-top: -12px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0px 5px 18px rgba(0,0,0,0.10);
}

.stButton>button {
    background-color: #0a4b78;
    color: white;
    padding: 10px 25px;
    border-radius: 8px;
    font-size: 18px;
    border: none;
}
.stButton>button:hover {
    background-color: #062d4a;
}
</style>
""", unsafe_allow_html=True)

# === Title ===
st.title("🛒 BigMart Sales Prediction App")
st.markdown(f"**Using trained model (scikit-learn v{sklearn_version}) to predict Item Outlet Sales.**", unsafe_allow_html=True)

# === Developer Profile Bar ===
st.markdown("""
<div class="info-bar">
    <h4>Developed by <b>Tejas Gholap</b></h4>
    <a href="https://linkedin.com/in/tejas-gholap-bb3417300" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin" height="28">
    </a>
    &nbsp;&nbsp;
    <a href="https://github.com/tejasgholap45" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-Profile-black?logo=github" height="28">
    </a>
</div>
""", unsafe_allow_html=True)

st.write("")

# === Form Layout ===
col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        Item_Identifier = st.text_input("Item Identifier", "FDA15")
        Item_Weight = st.number_input("Item Weight", min_value=0.0, value=12.5)
        Item_Fat_Content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
        Item_Visibility = st.slider("Item Visibility", 0.0, 0.3, 0.1)
        Item_Type = st.selectbox("Item Type", [
            "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household",
            "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", 
            "Health and Hygiene", "Hard Drinks", "Canned", "Breads", 
            "Starchy Foods", "Others", "Seafood"
        ])

        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        Item_MRP = st.number_input("Item MRP", min_value=0.0, value=150.0)
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
        Outlet_Age = st.slider("Outlet Age (Years)", 0, 40, 15)

        st.markdown('</div>', unsafe_allow_html=True)

# === Predict Button ===
st.write("")
if st.button("Predict Sales"):
    input_df = pd.DataFrame([{
        "Item_Identifier": Item_Identifier,
        "Item_Weight": Item_Weight,
        "Item_Fat_Content": Item_Fat_Content,
        "Item_Visibility": Item_Visibility,
        "Item_Type": Item_Type,
        "Item_MRP": Item_MRP,
        "Outlet_Identifier": Outlet_Identifier,
        "Outlet_Size": Outlet_Size,
        "Outlet_Location_Type": Outlet_Location_Type,
        "Outlet_Type": Outlet_Type,
        "Outlet_Age": Outlet_Age
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"📈 Predicted Sales: ₹{prediction:,.2f}")
