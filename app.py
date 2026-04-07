import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="HomeValuator", layout="wide", initial_sidebar_state="collapsed")

# =========================
# CUSTOM UI (Same as GemValuator)
# =========================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    padding: 2rem; border-radius: 12px; text-align: center;
}
.main-header h1 { color: #00d4ff; }
.metric-card {
    background: #1a1a2e; border-radius: 10px;
    padding: 1rem; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA & MODEL
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("kc_house_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("house_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

df = load_data()
model = load_model()
scaler = load_scaler()

# =========================
# MODEL COMPARISON DATA (REAL VALUES)
# =========================
results_df = pd.DataFrame({
    "Model": [
        "Gradient Boosting",
        "Cat Boost Regression",
        "Random Forest",
        "XGBoost Regressor",
        "Artificial Neural Network",
        "KNN Regression",
        "Decision Tree Regressor",
        "Support Vector Regression"
    ],
    "MAE": [
        155148.51,
        153411.16,
        160604.62,
        155725.56,
        174268.53,
        176406.25,
        213478.83,
        227292.02
    ],
    "RMSE": [
        239419.91,
        247176.84,
        248767.15,
        249941.48,
        264094.52,
        269805.86,
        341908.74,
        387253.82
    ],
    "R2 Score": [
        0.5915,
        0.5646,
        0.5590,
        0.5548,
        0.5030,
        0.4812,
        0.1669,
        -0.0687
    ]
})

# Sort by best model (LOW RMSE)
results_df = results_df.sort_values(by="RMSE")

features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.title("🏠 HomeValuator")
    page = st.radio("", ["Dashboard", "EDA", "Prediction", "Bulk Scanner","Visualization", "Model Comparison", "Model Logs"], label_visibility="collapsed")

# =========================
# HEADER
# =========================
st.markdown("""
<div class="main-header">
<h1>🏠 HomeValuator</h1>
<p>AI-powered house price prediction</p>
</div>
""", unsafe_allow_html=True)

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":

    c1, c2, c3 = st.columns(3)

    c1.markdown(f"<div class='metric-card'><h2>{len(df)}</h2><p>Total Houses</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card'><h2>${df['price'].mean():,.0f}</h2><p>Avg Price</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h2>{df['sqft_living'].mean():,.0f}</h2><p>Avg Sqft Living</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    fig1 = px.histogram(df, x="price", title="Price Distribution")
    fig2 = px.histogram(df, x="sqft_living", title="Living Area Distribution")

    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

# =========================
# EDA PAGE (HOME VERSION)
# =========================
elif page == "EDA":

    st.markdown("## 🏠 Data Intelligence Dashboard")

    # =========================
    # FILTERS
    # =========================
    st.markdown("### 🎛️ Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_bedrooms = st.multiselect(
            "Bedrooms",
            sorted(df["bedrooms"].unique()),
            default=sorted(df["bedrooms"].unique())
        )

    with col2:
        selected_floors = st.multiselect(
            "Floors",
            sorted(df["floors"].unique()),
            default=sorted(df["floors"].unique())
        )

    with col3:
        selected_zipcode = st.multiselect(
            "Zipcode",
            sorted(df["zipcode"].unique()),
            default=sorted(df["zipcode"].unique())
        )

    # =========================
    # FILTERED DATA
    # =========================
    filtered_df = df[
        (df["bedrooms"].isin(selected_bedrooms)) &
        (df["floors"].isin(selected_floors)) &
        (df["zipcode"].isin(selected_zipcode))
    ]

    st.markdown("---")

    # =========================
    # KPIs
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🏠 Total Houses", f"{len(filtered_df):,}")
    c2.metric("💰 Avg Price", f"${filtered_df['price'].mean():,.0f}")
    c3.metric("📐 Avg Sqft Living", f"{filtered_df['sqft_living'].mean():,.0f}")
    c4.metric("🛁 Avg Bathrooms", f"{filtered_df['bathrooms'].mean():.2f}")

    st.markdown("---")

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Distributions", "🔗 Relationships"])

    # =========================
    # TAB 1: OVERVIEW
    # =========================
    with tab1:

        col1, col2 = st.columns(2)

        # Bedrooms Distribution
        fig1 = px.bar(filtered_df, x="bedrooms", color="bedrooms", title="Bedrooms Distribution")
        col1.plotly_chart(fig1, use_container_width=True)

        # Floors Distribution
        fig2 = px.bar(filtered_df, x="floors", color="floors", title="Floors Distribution")
        col2.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        # Zipcode Distribution
        fig3 = px.histogram(filtered_df, x="zipcode", title="Zipcode Distribution")
        col3.plotly_chart(fig3, use_container_width=True)

        # Avg Price by Bedrooms
        avg_price = filtered_df.groupby("bedrooms")["price"].mean().reset_index()
        fig4 = px.bar(avg_price, x="bedrooms", y="price", color="bedrooms", title="Avg Price by Bedrooms")
        col4.plotly_chart(fig4, use_container_width=True)

    # =========================
    # TAB 2: DISTRIBUTIONS
    # =========================
    with tab2:

        feature = st.selectbox(
            "Select Feature",
            filtered_df.select_dtypes(include=np.number).columns
        )

        col1, col2 = st.columns(2)

        # Histogram
        fig5 = px.histogram(filtered_df, x=feature, nbins=50)
        col1.plotly_chart(fig5, use_container_width=True)

        # Boxplot
        fig6 = px.box(filtered_df, y=feature)
        col2.plotly_chart(fig6, use_container_width=True)

    # =========================
    # TAB 3: RELATIONSHIPS
    # =========================
    with tab3:

        st.subheader("🔗 Correlation Heatmap")

        numeric_df = filtered_df.select_dtypes(include=np.number)

        corr = numeric_df.corr()

        fig7 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig7, use_container_width=True)

        st.subheader("📈 Scatter Analysis")

        col1, col2 = st.columns(2)

        x_axis = col1.selectbox("X-axis", numeric_df.columns)
        y_axis = col2.selectbox("Y-axis", numeric_df.columns, index=1)

        fig8 = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color="bedrooms",
            trendline="ols",
            title=f"{x_axis} vs {y_axis}"
        )

        st.plotly_chart(fig8, use_container_width=True)
# =========================
# PREDICTION
# =========================
elif page == "Prediction":

    st.title("House Price Prediction")

    bedrooms = st.number_input("Bedrooms", 0, 10, 3, step=1)
    bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 2.0, step=0.25)
    sqft_living = st.number_input("Living Area (sqft)", 100, 10000, 2000, step=50)
    sqft_lot = st.number_input("Lot Area (sqft)", 500, 50000, 5000, step=50)
    floors = st.number_input("Floors", 1.0, 5.0, 1.0, step=0.5)
    zipcode = st.selectbox("Zipcode", sorted(df['zipcode'].unique()))

    input_data = pd.DataFrame([[bedrooms, bathrooms, sqft_living, sqft_lot, floors]], columns=features)

    st.write("### Input Data")
    st.write(input_data)

    if st.button("Predict Price"):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        st.session_state["prediction"] = prediction[0]

        st.session_state["last_input"] = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "zipcode": zipcode
        }

        if "history" not in st.session_state:
            st.session_state["history"] = []

        st.session_state["history"].append(prediction[0])

        st.success(f"💰 Predicted Price: ${prediction[0]:,.2f}")
# =========================
# BULK SCANNER PAGE (HOME)
# =========================
elif page == "Bulk Scanner":

    st.title("📂 Bulk House Price Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV / Excel / JSON file",
        type=["csv", "xlsx", "json"]
    )

    if uploaded_file is not None:

        try:
            # =========================
            # READ FILE
            # =========================
            if uploaded_file.name.endswith(".csv"):
                bulk_df = pd.read_csv(uploaded_file)

            elif uploaded_file.name.endswith(".xlsx"):
                bulk_df = pd.read_excel(uploaded_file)

            elif uploaded_file.name.endswith(".json"):
                bulk_df = pd.read_json(uploaded_file)

            st.write("### 📄 Uploaded Data")
            st.write(bulk_df.head())

            # =========================
            # REQUIRED FEATURES
            # =========================
            required_cols = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']

            # Check missing columns
            missing = [col for col in required_cols if col not in bulk_df.columns]

            if missing:
                st.error(f"❌ Missing columns: {missing}")
                st.stop()

            # =========================
            # PREPROCESSING
            # =========================
            bulk_processed = bulk_df[required_cols].copy()

            # Fill missing values
            bulk_processed = bulk_processed.fillna(0)

            # Scale (IMPORTANT)
            bulk_scaled = scaler.transform(bulk_processed)

            # =========================
            # PREDICTION
            # =========================
            predictions = model.predict(bulk_scaled)

            # Handle multi-output safely
            if isinstance(predictions[0], (list, np.ndarray)):
                predictions = [float(p[0]) for p in predictions]
            else:
                predictions = [float(p) for p in predictions]

            bulk_df["Predicted Price"] = predictions

            st.success("✅ Predictions generated successfully!")

            # =========================
            # RESULTS
            # =========================
            st.write("### 📊 Results")
            st.write(bulk_df)

            # =========================
            # DOWNLOAD
            # =========================
            csv = bulk_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Download Results",
                data=csv,
                file_name="house_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")


# =========================
# VISUALIZATION PAGE (HOME)
# =========================
elif page == "Visualization":

    st.title("🏠 Data Insights")

    # =========================
    # DATA PREVIEW
    # =========================
    st.subheader("📄 Dataset Preview")
    st.write(df.head())

    st.subheader("📊 Summary Statistics")
    st.write(df.describe())

    # =========================
    # PRICE DISTRIBUTION
    # =========================
    st.subheader("💰 Price Distribution")

    fig1, ax1 = plt.subplots()
    sns.histplot(df["price"], kde=True, ax=ax1)
    st.pyplot(fig1)

    # =========================
    # SCATTER PLOT
    # =========================
    st.subheader("📈 Living Area vs Price")

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df["sqft_living"], y=df["price"], ax=ax2)
    st.pyplot(fig2)

    # =========================
    # HEATMAP (FIXED ✅)
    # =========================
    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["number"])

    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    # =========================
    # BOXPLOT
    # =========================
    st.subheader("📦 Price by Bedrooms")

    fig4 = px.box(df, x="bedrooms", y="price", color="bedrooms")
    st.plotly_chart(fig4, use_container_width=True)

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    st.subheader("⭐ Feature Importance")

    try:
        # If model is RandomForest
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_names = features

        # If model is MultiOutputRegressor
        elif hasattr(model, "estimators_"):
            importance = model.estimators_[0].feature_importances_
            feature_names = features

        else:
            raise Exception("Model does not support feature importance")

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.write(importance_df)

        fig5, ax5 = plt.subplots()
        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            ax=ax5
        )
        st.pyplot(fig5)

    except Exception as e:
        st.error(f"Feature importance error: {e}")

# =========================
# MODEL COMPARISON
# =========================
elif page == "Model Comparison":

    st.title("📊 Model Comparison")

    # RMSE Graph
    st.subheader("RMSE Comparison")
    fig = px.bar(results_df, x="Model", y="RMSE", color="Model", text="RMSE")
    st.plotly_chart(fig, use_container_width=True)

    # MAE Graph
    st.subheader("MAE Comparison")
    fig = px.bar(results_df, x="Model", y="MAE", color="Model", text="MAE")
    st.plotly_chart(fig, use_container_width=True)

    # R2 Score Graph
    st.subheader("R2 Score Comparison")
    fig = px.bar(results_df, x="Model", y="R2 Score", color="Model", text="R2 Score")
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.dataframe(results_df)

    # Best model
    best = results_df.iloc[0]

    st.success(f"""
🏆 Best Model: {best['Model']}

RMSE: {best['RMSE']}
MAE: {best['MAE']}
R² Score: {best['R2 Score']}
""")
    
    # ✅ FIXED POSITION
    st.info(
        "The Gradient Boosting model outperforms others by minimizing prediction errors (lowest RMSE & MAE) while maximizing explained variance (highest R²), making it the most reliable model for this dataset."
    )
#=========================
# MODEL LOGS (HOME VERSION)
# =========================
elif page == "Model Logs":

    st.title("📜 Model Logs & Monitoring")

    # =========================
    # MODEL INFO
    # =========================
    st.subheader("🤖 Model Info")

    model_name = type(model).__name__

    st.write({
        "Model":"GradientBoostingRegressor",
        "Version": "1.0",
        "Algorithm": "Boosting"
    })

    # =========================
    # DATASET INFO
    # =========================
    st.subheader("Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # =========================
    # PERFORMANCE (STATIC / EDIT IF NEEDED)
    # =========================
    st.subheader("📈 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("RMSE", "239419.91")
    c2.metric("MAE", "155148.51")
    c3.metric("R² Score", "0.5915")

    # =========================
    # LAST PREDICTION
    # =========================
    st.subheader("💰 Last Prediction")

    if "prediction" in st.session_state:
        st.success(f"${st.session_state['prediction']:,.2f}")
    else:
        st.warning("No prediction made yet")

    # =========================
    # INPUT LOG
    # =========================
    st.subheader("🧾 Last Input Data")

    if "last_input" in st.session_state:
        st.write(pd.DataFrame([st.session_state["last_input"]]))
    else:
        st.info("No input recorded yet")

    # =========================
    # PREDICTION HISTORY
    # =========================
    st.subheader("📜 Prediction History")

    if "history" in st.session_state and len(st.session_state["history"]) > 0:
        history_df = pd.DataFrame(st.session_state["history"], columns=["Predicted Price"])
        st.dataframe(history_df)
    else:
        st.warning("No prediction history available")

    # =========================
    # TIMESTAMP
    # =========================
    st.subheader("⏱️ Current Timestamp")

    st.write(datetime.datetime.now())