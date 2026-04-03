import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart House Price Predictor", layout="wide")
st.title("🏠 Smart House Price Prediction System")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("kc_house_data.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

df = load_data()

# =========================
# LOAD MODEL (NO TRAINING)
# =========================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("house_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None

model, scaler = load_model()

# =========================
# SIDEBAR
# =========================
page = st.sidebar.radio("Navigation", [
    "📈 Dashboard",
    "🏡 Prediction",
    "📊 Visualization",
    "🤝 Recommendation",
    "📋 Dataset Info"
])

# =========================
# DASHBOARD
# =========================
if page == "📈 Dashboard":
    st.header("🏠 House Data Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Houses", len(df))
    c2.metric("Avg Price", f"${df['price'].mean():,.0f}")
    c3.metric("Avg Living Area", f"{df['sqft_living'].mean():,.0f} sqft")

    st.markdown("---")

    col1, col2 = st.columns(2)

    fig1 = px.histogram(df, x="price", title="Price Distribution")
    fig2 = px.histogram(df, x="sqft_living", title="Living Area Distribution")

    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

# =========================
# PREDICTION
# =========================
elif page == "🏡 Prediction":
    st.header("Enter House Details")

    bedrooms = st.number_input("Bedrooms", 0, 10, 3)
    bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0)
    sqft_living = st.number_input("Living Area (sqft)", 100, 10000, 2000)
    sqft_lot = st.number_input("Lot Area (sqft)", 500, 50000, 5000)
    floors = st.number_input("Floors", 1.0, 5.0, 1.0)

    if st.button("Predict Price"):
        if model is None or scaler is None:
            st.error("Model not loaded properly")
        else:
            input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors]])
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)
            price = prediction[0][0]  # FIXED

            st.success(f"💰 Predicted Price: ${price:,.2f}")

# =========================
# VISUALIZATION
# =========================
elif page == "📊 Visualization":
    st.header("Data Insights")

    fig, ax = plt.subplots()
    ax.hist(df['price'], bins=50)
    st.pyplot(fig)

# =========================
# RECOMMENDATION
# =========================
elif page == "🤝 Recommendation":
    st.header("Property Recommendation")

    budget = st.slider("Max Budget", 100000, 2000000, 500000)
    min_bedrooms = st.slider("Min Bedrooms", 1, 10, 3)

    filtered = df[(df['price'] <= budget) & (df['bedrooms'] >= min_bedrooms)]

    st.write(f"Showing {len(filtered)} properties")
    st.dataframe(filtered[['price','bedrooms','bathrooms','sqft_living']].head(10))

# =========================
# DATASET INFO
# =========================
elif page == "📋 Dataset Info":
    st.header("Dataset Overview")

    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    st.subheader("Summary")
    st.write(df.describe())

    st.subheader("Timestamp")
    st.write(datetime.datetime.now())