import requests
import os
import joblib
import pandas as pd
import pickle

# ===============================
# Download files if not exists
# ===============================
if not os.path.exists("kc_house_data.csv"):
    url_csv = "https://link-to-your-kc_house_data.csv"
    r = requests.get(url_csv)
    with open("kc_house_data.csv", "wb") as f:
        f.write(r.content)

if not os.path.exists("house_model.pkl"):
    url_model = "https://link-to-your-house_model.pkl"
    r = requests.get(url_model)
    with open("house_model.pkl", "wb") as f:
        f.write(r.content)

if not os.path.exists("scaler.pkl"):
    url_scaler = "https://link-to-your-scaler.pkl"
    r = requests.get(url_scaler)
    with open("scaler.pkl", "wb") as f:
        f.write(r.content)

        # Load model and scaler
with open("house_model.pkl", "rb") as f:
    model = joblib.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = joblib.load(f)

# Load dataset
df = pd.read_csv("kc_house_data.csv")

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import matplotlib.pyplot as plt
import pickle
import plotly.express as px

import requests
import joblib
import os
import pandas as pd

# =========================
# Download files at runtime
# =========================

# URLs to your hosted files
url_csv   = "https://raw.githubusercontent.com/yourusername/HOME-VALUE-/main/kc_house_data.csv"
url_model = "https://raw.githubusercontent.com/yourusername/HOME-VALUE-/main/house_model.pkl"
url_scaler = "https://raw.githubusercontent.com/yourusername/HOME-VALUE-/main/scaler.pkl"

# Download CSV if not exists
if not os.path.exists("kc_house_data.csv"):
    r = requests.get(url_csv)
    with open("kc_house_data.csv", "wb") as f:
        f.write(r.content)

# Download model if not exists
if not os.path.exists("house_model.pkl"):
    r = requests.get(url_model)
    with open("house_model.pkl", "wb") as f:
        f.write(r.content)

# Download scaler if not exists
if not os.path.exists("scaler.pkl"):
    r = requests.get(url_scaler)
    with open("scaler.pkl", "wb") as f:
        f.write(f.content)

# Load model & scaler using joblib
model = joblib.load("house_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
df = pd.read_csv("kc_house_data.csv")

# =========================
# 2. Train or Load Model & Scaler
# =========================
MODEL_FILE = "house_model.pkl"
SCALER_FILE = "scaler.pkl"

if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    # download or create model/scaler
    # Train the model
    # Features and targets
    features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']
    targets = ['price','sqft_living','bedrooms','bathrooms']

    X = df[features]
    y = df[targets]

    # Fill missing values
    X = X.fillna(0)
    y = y.fillna(0)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multi-output Random Forest
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model = MultiOutputRegressor(rf)
    model.fit(X_train_scaled, y_train)

    # Save for future use
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

# =========================
# 3. Streamlit App Layout
# =========================
st.set_page_config(page_title="Smart House Price Predictor", layout="wide")
st.title("🏠 Smart House Price Prediction System")

page = st.sidebar.radio("Navigation", [
    "📈 Dashboard", # new page
    "🏡 Prediction",
    "📊 Visualization",
    "🔍 Explainability (SHAP)",
    "🤝 Recommendation",
    "📋 Dataset Info"
])
# =========================
# Dashboard Page
# =========================
if page == "📈 Dashboard":
    st.header("🏠 House Data Dashboard")

    # KPI Cards
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div style='padding:20px; background-color:#1f77b4; color:white; border-radius:10px; text-align:center;'><h2>{len(df)}</h2><p>Total Houses</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div style='padding:20px; background-color:#ff7f0e; color:white; border-radius:10px; text-align:center;'><h2>${df['price'].mean():,.0f}</h2><p>Average Price</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div style='padding:20px; background-color:#2ca02c; color:white; border-radius:10px; text-align:center;'><h2>{df['sqft_living'].mean():,.0f}</h2><p>Average Living Area (sqft)</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Row 1: Distribution of Bedrooms & Bathrooms
    col1, col2 = st.columns(2)

    bed_counts = df['bedrooms'].value_counts().sort_index().reset_index()
    bed_counts.columns = ['bedrooms','count']
    fig_bed = px.bar(bed_counts, x='bedrooms', y='count', text='count', title="Bedrooms Distribution")
    fig_bed.update_layout(template="plotly_dark")

    bath_counts = df['bathrooms'].value_counts().sort_index().reset_index()
    bath_counts.columns = ['bathrooms','count']
    fig_bath = px.bar(bath_counts, x='bathrooms', y='count', text='count', title="Bathrooms Distribution")
    fig_bath.update_layout(template="plotly_dark")

    col1.plotly_chart(fig_bed, use_container_width=True)
    col2.plotly_chart(fig_bath, use_container_width=True)

    # Row 2: Living Area & Price Distributions
    col3, col4 = st.columns(2)

    fig_sqft = px.histogram(df, x='sqft_living', nbins=50, title="Living Area Distribution")
    fig_sqft.update_layout(template="plotly_dark")

    fig_price = px.histogram(df, x='price', nbins=50, title="Price Distribution")
    fig_price.update_layout(template="plotly_dark")

    col3.plotly_chart(fig_sqft, use_container_width=True)
    col4.plotly_chart(fig_price, use_container_width=True)

    # Row 3: Average Price by Floors
    st.subheader("Average Price by Floors")
    avg_price_floors = df.groupby('floors')['price'].mean().reset_index()
    fig_avg_floors = px.bar(avg_price_floors, x='floors', y='price', text_auto='.2s', title="Average Price by Floors")
    fig_avg_floors.update_layout(template="plotly_dark")
    st.plotly_chart(fig_avg_floors, use_container_width=True)

# =========================
# Prediction Page
# =========================
if page == "🏡 Prediction":
    st.header("Enter House Details")

    bedrooms = st.number_input("Bedrooms", 0, 10, 3, step=1)
    bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0, step=0.25)
    sqft_living = st.number_input("Living Area (sqft)", 100, 10000, 2000, step=50)
    sqft_lot = st.number_input("Lot Area (sqft)", 500, 50000, 5000, step=50)
    floors = st.number_input("Floors", 1.0, 5.0, 1.0, step=0.5)
    zipcode = st.selectbox("Zipcode", sorted(df['zipcode'].unique()))

    if st.button("Predict Price"):
        input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, floors]])
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)  # shape: (1, n_targets)
        prediction_value = prediction[0]if prediction.ndim == 1 else prediction[0][0]  
        prediction_value = float(prediction_value)    # get the first (and only) value

        st.success(f"💰 Predicted Price: ${prediction_value:,.2f}")

        # Save in session state
        st.session_state["last_prediction"] = prediction_value
        st.session_state["last_input"] = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "zipcode": zipcode
        }
# =========================
# Visualization Page
# =========================
elif page == "📊 Visualization":
    st.header("Data Insights")
    st.subheader("Price Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['price'], bins=50, color="#1f77b4", alpha=0.7)
    st.pyplot(fig)

# =========================
# SHAP Explainability
# =========================
elif page == "🔍 Explainability (SHAP)":
    st.header("Feature Importance via SHAP")
    
    features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']
    X_for_shap = df[features].values  # use raw features, not scaled

    targets = ['price','sqft_living','bedrooms','bathrooms']

    for i, target in enumerate(targets):
        st.subheader(f"Feature Importance for {target}")
        single_model = model.estimators_[i]  # get individual RF for this target

        # Initialize explainer (no check_additivity here)
        explainer = shap.Explainer(single_model, X_for_shap)
        
        # Compute SHAP values with check_additivity=False
        shap_values = explainer(X_for_shap[:100], check_additivity=False)

        shap.plots.bar(shap_values, show=False)
        st.pyplot(plt.gcf())
# =========================
# Recommendation Page
# =========================
elif page == "🤝 Recommendation":
    st.header("Property Recommendation")
    budget = st.slider("Max Budget", 100_000, 2_000_000, 500_000, step=10_000)
    min_bedrooms = st.slider("Min Bedrooms", 1, 10, 3)
    filtered = df[(df['price'] <= budget) & (df['bedrooms'] >= min_bedrooms)]
    st.write(f"Showing {len(filtered)} properties")
    st.dataframe(filtered[['price','bedrooms','bathrooms','sqft_living','zipcode']].sort_values('price').head(10))

# =========================
# Dataset Info
# =========================
elif page == "📋 Dataset Info":
    st.header("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write("Missing Values:")
    st.write(df.isnull().sum())
    st.subheader("Summary Statistics")
    st.write(df.describe())
    st.subheader("Timestamp")
    st.write(datetime.datetime.now())