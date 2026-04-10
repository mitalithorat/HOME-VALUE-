import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import datetime
import plotly.graph_objects as go


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="HomeValuator", layout="wide", initial_sidebar_state="collapsed")

# =========================
# GEMVALUATOR STYLE UI
# =========================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
:root {
    --bg-base:#1e2130;
    --bg-card:#252838;
    --bg-sidebar:#1a1d2e;
    --bg-input:#2d3148;
    --accent:#00d4ff;
    --accent-dim:#00a8cc;
    --border:#363a52;
    --text-hi:#eef0f8;
    --text-mid:#a0a8c8;
}

/* GLOBAL */
html, body, .stApp {
    font-family:'Plus Jakarta Sans',sans-serif !important;
    background-color:var(--bg-base) !important;
    color:var(--text-hi) !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background:var(--bg-sidebar) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    border:1px solid var(--border);
    border-radius:8px;
    padding:10px;
    margin-bottom:6px;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    border-color:var(--accent);
    background:rgba(0,212,255,0.1);
}

/* METRICS */
div[data-testid="stMetric"] {
    background:var(--bg-card);
    border:1px solid var(--border);
    border-radius:12px;
    padding:18px;
}
div[data-testid="stMetricValue"] {
    color:var(--accent);
}

/* BUTTON */
.stButton > button {
    background:var(--accent);
    color:white;
    border-radius:8px;
}
.stButton > button:hover {
    background:var(--accent-dim);
}

/* HERO */
.hero {
    background:linear-gradient(135deg,#252838,#2d3148);
    padding:30px;
    border-radius:16px;
    text-align:center;
}

/* RESULT */
.result-box {
    background:#252838;
    border-radius:16px;
    padding:30px;
    text-align:center;
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

features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']



# =========================
# LOG SYSTEM
# =========================
if "logs" not in st.session_state:
    st.session_state.logs = []

def add_log(msg, level="info"):
    t = datetime.datetime.now().strftime("%H:%M:%S")
    icon = "ℹ️"
    if level == "success": icon = "✅"
    if level == "error": icon = "❌"
    st.session_state.logs.append(f"{icon} [{t}] {msg}")

# =========================
# SIDEBAR
# =========================
st.sidebar.markdown('<div style="font-weight:700;">🏠 HOMEVALUATOR</div>', unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["Dashboard","EDA","Prediction","Upload & Explore","Visualization","Model Comparison","Model Logs"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="font-size:0.75rem;color:#a0a8c8;text-align:center;">AI Model · House Price Predictor</p>',
    unsafe_allow_html=True
)

if page == "Dashboard":

    # =========================
    # HERO SECTION (LIKE AQI)
    # =========================
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 36px 32px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 28px;
        border: 1px solid #2a2e4a;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    ">
        <h1 style="color:#00d4ff; font-size:2.2rem; margin-bottom:8px;">
            🏠 HomeValuator
        </h1>
        <p style="color:#a0a8c8; font-size:1rem; margin:0;">
            Smart House Price Prediction & Analytics Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # METRICS (LIKE AQI CARDS)
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("🏘 Total Houses", f"{len(df)}")
    col2.metric("💰 Avg Price", f"${df['price'].mean():,.0f}")
    col3.metric("📐 Avg Living Area", f"{df['sqft_living'].mean():,.0f} sqft")

    st.markdown("---")

    # =========================
    # INFO SECTION
    # =========================
    left, right = st.columns(2)

    # LEFT SIDE
    with left:
        st.markdown("### 🧠 How Prediction Works")

        st.markdown("""
This model predicts house prices using machine learning based on key features:

- Bedrooms 🛏
- Bathrooms 🚿
- Living Area 📐
- Lot Size 🌳
- Floors 🏢

Higher values generally increase price, but relationships are learned from real data.
        """)

        # Small table (like AQI categories)
        info_df = pd.DataFrame({
            "Feature": ["Bedrooms", "Bathrooms", "Sqft Living", "Sqft Lot", "Floors"],
            "Impact": [
                "Moderate",
                "High",
                "Very High",
                "Moderate",
                "Low–Moderate"
            ]
        })
        st.table(info_df)

    # RIGHT SIDE
    with right:
        st.markdown("### 📊 Key Insights")

        insights = [
            "Larger homes → higher prices",
            "Location strongly influences value",
            "More bathrooms increase price significantly",
            "Luxury homes skew average price",
            "Outliers exist in high-end properties"
        ]

        for i in insights:
            st.markdown(f"""
            <div style="
                background:#1a1a2e;
                border:1px solid #2a2e4a;
                border-radius:10px;
                padding:10px 14px;
                margin-bottom:8px;">
                <span style="color:#00d4ff; font-weight:600;">•</span>
                <span style="color:#a0a8c8;"> {i}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # CHARTS (LIKE AQI VISUALS)
    # =========================
    col1, col2 = st.columns(2)

    fig1 = px.histogram(
        df, x="price",
        title="💰 Price Distribution",
        color_discrete_sequence=["#00d4ff"]
    )

    fig2 = px.histogram(
        df, x="sqft_living",
        title="📐 Living Area Distribution",
        color_discrete_sequence=["#5b8dee"]
    )

    col1.plotly_chart(fig1, use_container_width=True)
    col2.plotly_chart(fig2, use_container_width=True)

    # =========================
    # EXTRA SECTION (LIKE SAFETY / ENVIRONMENT)
    # =========================
    st.markdown("---")

    col_env, col_safe = st.columns(2)

    with col_env:
        st.markdown("### 🌍 Market Factors")
        for item in [
            "Interest rates affect demand",
            "Urban areas have higher prices",
            "Infrastructure boosts value",
            "Neighborhood quality matters"
        ]:
            st.markdown(f"- {item}")

    with col_safe:
        st.markdown("### 💡 Tips for Buyers")
        for item in [
            "Check price per sqft",
            "Compare similar properties",
            "Analyze historical trends",
            "Don't rely only on model prediction"
        ]:
            st.markdown(f"- {item}")
# =========================
# EDA PAGE (HOME VERSION)
# =========================
elif page == "EDA":

    st.markdown("""
    <h1 style="display:flex;align-items:center;gap:10px;">
        <span style="color:#e94560;font-size:2rem;">🏠</span>
        Data Intelligence Dashboard
    </h1>
    """, unsafe_allow_html=True)

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
    # FILTER DATA
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
    c3.metric("📐 Avg Living Area", f"{filtered_df['sqft_living'].mean():,.0f}")
    c4.metric("🛏 Avg Bedrooms", f"{filtered_df['bedrooms'].mean():.1f}")

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
        fig1 = px.bar(filtered_df, x="bedrooms", color="bedrooms",
                      title="Bedrooms Distribution")
        col1.plotly_chart(fig1, use_container_width=True)

        # Floors Distribution
        fig2 = px.bar(filtered_df, x="floors", color="floors",
                      title="Floors Distribution")
        col2.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)

        # Zipcode Distribution
        fig3 = px.histogram(filtered_df, x="zipcode",
                            title="Zipcode Distribution")
        col3.plotly_chart(fig3, use_container_width=True)

        # Avg Price by Bedrooms
        avg_price = filtered_df.groupby("bedrooms")["price"].mean().reset_index()
        fig4 = px.bar(avg_price, x="bedrooms", y="price",
                      color="bedrooms",
                      title="Avg Price by Bedrooms")
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
        fig5 = px.histogram(
            filtered_df,
            x=feature,
            nbins=50,
            color_discrete_sequence=["#e94560"]
        )
        col1.plotly_chart(fig5, use_container_width=True)

        # Boxplot
        fig6 = px.box(
            filtered_df,
            y=feature,
            color_discrete_sequence=["#0f3460"]
        )
        col2.plotly_chart(fig6, use_container_width=True)

    # =========================
    # TAB 3: RELATIONSHIPS
    # =========================
    with tab3:

        st.subheader("🔗 Correlation Heatmap")

        corr = filtered_df.select_dtypes(include=np.number).corr()

        fig7 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r"
        )
        st.plotly_chart(fig7, use_container_width=True)

        st.subheader("📈 Scatter Analysis")

        col1, col2 = st.columns(2)

        numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()

        # Smart defaults
        default_x = "sqft_living" if "sqft_living" in numeric_cols else numeric_cols[0]
        default_y = "price" if "price" in numeric_cols else numeric_cols[1]

        x_axis = col1.selectbox("X-axis", numeric_cols,
                               index=numeric_cols.index(default_x))
        y_axis = col2.selectbox("Y-axis", numeric_cols,
                               index=numeric_cols.index(default_y))

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
# PREDICTION PAGE
# =========================
elif page == "Prediction":

    st.markdown("""
    <h1 style="display:flex;align-items:center;gap:10px;">
        <span class="material-icons" style="color:#e94560;font-size:2rem;"></span>
        Predict Home Price
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#a0a8c8; margin-bottom:20px;">
    Enter property features below. The AI model will predict the price instantly.
    </p>
    """, unsafe_allow_html=True)

    st.info("""
📌 Input Guidelines:
- Square Footage (sqft): 100 – 10000  
- Bedrooms: 1 – 10  
- Bathrooms: 1 – 10  
- Year Built: 1900 – Current Year  
- Garage Spots: 0 – 10  
- Lot Size (sqft): 100 – 50000  
- Floor (for apartments): 0 – 50  
""")

    # =========================
    # FORM INPUT (ENHANCED)
    # =========================
    with st.form("prediction_form"):

        # Row 1
        c1, c2 = st.columns(2)
        sqft = c1.number_input("Square Footage", min_value=100, max_value=10000, value=1200)
        bedrooms = c2.number_input("Bedrooms", min_value=1, max_value=10, value=3)

        # Row 2
        c3, c4 = st.columns(2)
        bathrooms = c3.number_input("Bathrooms", min_value=1, max_value=10, value=2)
        year_built = c4.number_input("Year Built", min_value=1900, max_value=2026, value=2000)

        # Row 3
        c5, c6 = st.columns(2)
        garage = c5.number_input("Garage Spots", min_value=0, max_value=10, value=1)
        lot_size = c6.number_input("Lot Size (sqft)", min_value=100, max_value=50000, value=5000)

        # Row 4
        c7, c8 = st.columns(2)
        floor = c7.number_input("Floor (if apartment)", min_value=0, max_value=50, value=1)
        basement = c8.selectbox("Basement", ["No", "Yes"])

        # Row 5
        c9, c10 = st.columns(2)
        heating = c9.selectbox("Heating Type", ["None", "Gas", "Electric", "Oil", "Other"])
        cooling = c10.selectbox("Cooling Type", ["None", "Central", "Window", "Other"])

        # Row 6
        property_type = st.selectbox("Property Type", ["House", "Condo", "Apartment", "Townhouse", "Other"])

        # ✅ Submit button
        submitted = st.form_submit_button("⚡ Predict Price")
    # =========================
    # PREDICTION LOGIC
    # =========================
    if submitted:
        try:
            # -------------------------
            # Basic validation
            # -------------------------
            if sqft <= 0 or bedrooms <= 0 or bathrooms <= 0:
                st.error("❌ Square footage, bedrooms, and bathrooms must be greater than 0")
                st.stop()

            if year_built < 1900 or year_built > 2026:
                st.error("❌ Year built is out of range")
                st.stop()

            # -------------------------
            # Prepare input dataframe
            # -------------------------
            input_data = pd.DataFrame([{
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft_living": sqft,
                "sqft_lot": lot_size,
                "floors": floor,
                "basement": basement,
                "heating": heating,
                "cooling": cooling,
                "property_type": property_type
            }])

            # Concatenate with training df (without target)
            combined = pd.concat([df.drop("price", axis=1), input_data], ignore_index=True)
            combined = pd.get_dummies(combined)

            # Extract last row (your input) and align with model features
            input_processed = combined.tail(1)
            # ✅ Align columns with trained model safely
            if hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
            else:
                # fallback: use the features you trained the model on
                feature_names = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']

            input_processed = input_processed.reindex(columns=feature_names, fill_value=0)
            # Scale numeric columns
            numeric_cols = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']
            input_processed[numeric_cols] = scaler.transform(input_processed[numeric_cols])

            # Make prediction
            prediction = model.predict(input_processed)[0]

            prediction = prediction * 100000  # convert to lakhs scale

            st.session_state["prediction"] = prediction

            # -------------------------
            # Logging
            # -------------------------
            if "logs" not in st.session_state:
                st.session_state["logs"] = []

            st.session_state["prediction"] = prediction
            st.session_state["input_data"] = input_data

            import datetime
            st.session_state.logs.append(
                f"{datetime.datetime.now().strftime('%H:%M:%S')} → Prediction: ₹ {prediction/100000:.2f} Lakh"
            )

            # -------------------------
            # Categorize prediction
            # -------------------------
            if prediction < 2000000:
                category = "Low Value"
                color_code = "#00c96e"
            elif prediction < 10000000:
                category = "Medium Value"
                color_code = "#f5c518"
            elif prediction < 30000000:
                category = "High Value"
                color_code = "#ff8c00"        
            else:
                category = "Luxury"
                color_code = "#e94560"

            # -------------------------
            # DISPLAY RESULT
            # -------------------------
            res_l, res_r = st.columns(2)

            with res_l:
                st.markdown(f"""
                <div style="
                    background:#252838;
                    border:1px solid {color_code};
                    border-radius:16px;
                    padding:28px;
                    text-align:center;
                    box-shadow: 0 0 25px {color_code}55;
                ">
                    <p style="color:#a0a8c8; font-size:0.8rem;">Predicted Price</p>
                    <p style="font-size:3rem; color:{color_code}; margin:0;">
                        ₹ {prediction:,.0f}
                    </p>
                    <p style="color:{color_code}; font-weight:600;">
                        {category}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with res_r:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    number={
                        'font': {'family': 'Plus Jakarta Sans', 'color': color_code, 'size': 28}
                    },
                    gauge={
                        'axis': {'range': [0, 50000000], 'tickcolor': '#a0a8c8', 'tickfont': {'color': '#a0a8c8'}},
                        'bar': {'color': color_code, 'thickness': 0.25},
                        'bgcolor': 'rgba(37,40,56,0.95)',
                        'bordercolor': 'rgba(54,58,56,1)',
                        'steps': [
                            {'range': [0, 2000000], 'color': '#00c96e'},
                            {'range': [2000000, 10000000], 'color': '#f5c518'},
                            {'range': [10000000, 30000000], 'color': '#ff8c00'},
                            {'range': [30000000, 50000000], 'color': '#e94560'},
                        ],
                        'threshold': {'line': {'color': color_code, 'width': 3}, 'thickness': 0.75, 'value': prediction}
                    }
                ))

                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#a0a8c8"),
                    height=260,
                    margin=dict(t=20, b=10, l=20, r=20)
                )

                st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
            if "logs" not in st.session_state:
                st.session_state["logs"] = []
            import datetime
            st.session_state.logs.append(f"{datetime.datetime.now().strftime('%H:%M:%S')} → Prediction failed: {e}")
# =========================
#  Upload & Explore Data (HOME VERSION)
# =========================
elif page == "Upload & Explore":

    st.markdown("""
    <h1 style="display:flex;align-items:center;gap:10px;">
        <span style="color:#e94560;font-size:2rem;">🏠</span>
        Upload & Explore Data
    </h1>
    """, unsafe_allow_html=True)

    # =========================
    # SESSION STATE
    # =========================
    if "bulk_df" not in st.session_state:
        st.session_state.bulk_df = None

    file = st.file_uploader(
        "Upload File",
        type=["csv", "xlsx", "json"],
        key="bulk_file"
    )

    drive_link = st.text_input("Or paste Google Drive file link")

    # =========================
    # SAMPLE DATA
    # =========================
    if st.button("Load Sample Dataset"):
        try:
            sample_df = df.sample(200)
            st.session_state.bulk_df = sample_df
            st.success("Sample dataset loaded successfully")
        except Exception as e:
            st.error(f"Error: {e}")

    # =========================
    # GOOGLE DRIVE CONVERTER
    # =========================
    def convert_drive_link(link):
        import re
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", link)
        if match:
            file_id = match.group(1)
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        return None

    # =========================
    # FILE LOADING
    # =========================
    if file:
        ext = file.name.split(".")[-1].lower()

        if ext == "csv":
            st.session_state.bulk_df = pd.read_csv(file)

        elif ext == "xlsx":
            st.session_state.bulk_df = pd.read_excel(file)

        elif ext == "json":
            df_json = pd.read_json(file)
            if isinstance(df_json.iloc[0], dict):
                df_json = pd.json_normalize(df_json)
            st.session_state.bulk_df = df_json

        st.success(f"{ext.upper()} file loaded successfully")

    elif drive_link:
        link = convert_drive_link(drive_link)

        if link:
            try:
                import requests
                from io import BytesIO

                response = requests.get(link)
                file_bytes = BytesIO(response.content)

                try:
                    st.session_state.bulk_df = pd.read_csv(file_bytes)
                except:
                    file_bytes.seek(0)
                    try:
                        st.session_state.bulk_df = pd.read_excel(file_bytes)
                    except:
                        file_bytes.seek(0)
                        st.session_state.bulk_df = pd.read_json(file_bytes)

                st.success("File loaded from Google Drive")

            except Exception as e:
                st.error(f"Drive load failed: {e}")
        else:
            st.error("Invalid Google Drive link")

    # =========================
    # MAIN LOGIC
    # =========================
    bulk_df = st.session_state.bulk_df

    if bulk_df is None:
        st.info("Upload file / Load sample / Use Google Drive")

    else:
        bulk_df = bulk_df.replace(r'^\s*$', np.nan, regex=True)
        bulk_df = bulk_df.dropna()

        st.subheader("📊 Uploaded Data")
        st.dataframe(bulk_df.head())

        # =========================
        # FILTERING
        # =========================
        col = st.sidebar.selectbox("Filter Column (Bulk)", bulk_df.columns)

        if pd.api.types.is_numeric_dtype(bulk_df[col]):
            min_val = float(bulk_df[col].min())
            max_val = float(bulk_df[col].max())

            val = st.sidebar.slider("Range", min_val, max_val, (min_val, max_val))
            filtered_df = bulk_df[
                (bulk_df[col] >= val[0]) & (bulk_df[col] <= val[1])
            ]
        else:
            val = st.sidebar.selectbox("Value", bulk_df[col].unique())
            filtered_df = bulk_df[bulk_df[col] == val]

        st.subheader("Filtered Data")
        st.dataframe(filtered_df)

        # =========================
        # BULK PREDICTION (FIXED)
        # =========================
        try:
            required_cols = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors']

            # CHECK MISSING
            missing = [c for c in required_cols if c not in filtered_df.columns]

            if missing:
                st.error(f"❌ Missing columns: {missing}")
                st.stop()

            bulk_processed = filtered_df[required_cols].copy()

            # FILL NA
            bulk_processed = bulk_processed.fillna(0)

            # SCALE
            bulk_scaled = scaler.transform(bulk_processed)

            # PREDICT
            predictions = model.predict(bulk_scaled)

            filtered_df["Predicted Price"] = predictions

            st.success("✅ Predictions generated!")

            st.dataframe(filtered_df)

        except Exception as e:
            st.error(f"Prediction error: {e}")

        # =========================
        # DOWNLOAD
        # =========================
        csv = filtered_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="house_predictions.csv",
            mime="text/csv"
        )

        # =========================
        # VISUALIZATION
        # =========================
        num_cols = filtered_df.select_dtypes(include=np.number).columns

        if len(num_cols) > 0:
            c = st.selectbox("Histogram Column", num_cols)
            fig = px.histogram(filtered_df, x=c)
            st.plotly_chart(fig)

        if len(num_cols) >= 2:
            x = st.selectbox("X Axis", num_cols)
            y = st.selectbox("Y Axis", [i for i in num_cols if i != x])

            fig2 = px.scatter(filtered_df, x=x, y=y)
            st.plotly_chart(fig2)
# =========================
# VISUALIZATION PAGE
# =========================
elif page == "Visualization":

    st.markdown("""
    <h1 style="display:flex;align-items:center;gap:10px;">
        <span class="material-icons" style="color:#e94560;font-size:2rem;"></span>
        Data Insights - Home Prices
    </h1>
    """, unsafe_allow_html=True)

    # =========================
    # DATA OVERVIEW
    # =========================
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())

    st.markdown("---")

    # =========================
    # HISTOGRAM (PRICE DISTRIBUTION)
    # =========================
    st.subheader("🏠 Price Distribution")

    price_col = "price" if "price" in df.columns else df.select_dtypes(include=np.number).columns[-1]

    fig1 = px.histogram(
        df,
        x=price_col,
        nbins=50,
        color_discrete_sequence=["#e94560"]
    )
    st.plotly_chart(fig1, use_container_width=True)

    # =========================
    # SCATTER PLOT
    # =========================
    st.subheader("📉 Price vs Area")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    default_x = "GrLivArea" if "GrLivArea" in numeric_cols else numeric_cols[0]
    default_y = price_col

    col1, col2 = st.columns(2)

    x_axis = col1.selectbox("X-axis", numeric_cols, index=numeric_cols.index(default_x))
    y_axis = col2.selectbox("Y-axis", numeric_cols, index=numeric_cols.index(default_y))

    fig2 = px.scatter(
        df,
        x=x_axis,
        y=y_axis,
        trendline="ols",
        color_discrete_sequence=["#0f3460"]
    )
    st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # HEATMAP
    # =========================
    st.subheader("🔥 Correlation Heatmap")

    corr = df.select_dtypes(include=np.number).corr()

    fig3 = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # =========================
    # BOX PLOT
    # =========================
    st.subheader("📦 Price Distribution by Category")

    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if len(cat_cols) > 0:
        selected_cat = st.selectbox("Select Category", cat_cols)

        fig4 = px.box(
            df,
            x=selected_cat,
            y=price_col,
            color=selected_cat
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No categorical columns available for boxplot")

   # =========================
    # FEATURE IMPORTANCE (FINAL - NO ERROR)
    # =========================
    st.subheader("⭐ Feature Importance")

    try:
        importance = model.feature_importances_

        # =========================
        # GET FEATURE NAMES SAFELY
        # =========================
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)

        elif 'X_train' in globals():
            feature_names = list(X_train.columns)

        else:
            # fallback (drop target column)
            target_col = "SalePrice" if "SalePrice" in df.columns else df.columns[-1]
            feature_names = list(df.drop(target_col, axis=1).columns)

        # =========================
        # FIX LENGTH MISMATCH
        # =========================
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]

        # =========================
        # CREATE DATAFRAME
        # =========================
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        # =========================
        # DISPLAY
        # =========================
        st.dataframe(importance_df)

        fig = px.bar(
            importance_df.head(15),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="viridis",
            title="Top 15 Important Features"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Feature importance error: {e}")

# =========================
# MODEL COMPARISON
# =========================
elif page == "Model Comparison":

    st.markdown("""
    <h1 style="display:flex;align-items:center;gap:10px;">
        <span class="material-icons" style="color:#e94560;font-size:2rem;"></span>
        Model Comparison
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("Compare different ML models based on performance metrics.")

    # =========================
    # 📊 LOAD RESULTS
    # =========================
    results_df = pd.read_csv("model_results.csv")

    st.title("📊 Model Comparison Dashboard")

    # =========================
    # CHECK DATA
    # =========================
    if results_df is None or results_df.empty:
        st.warning("No model comparison data available.")
    else:

        # =========================
        # SORT BY BEST MODEL (LOW RMSE)
        # =========================
        results_df = results_df.sort_values(by="RMSE")

        # =========================
        # RMSE GRAPH
        # =========================
        st.subheader("📉 RMSE Comparison (Lower is Better)")

        fig1 = px.bar(
            results_df,
            x="Model",
            y="RMSE",
            color="Model",
            text="RMSE",
            color_discrete_sequence=px.colors.sequential.Reds
        )
        st.plotly_chart(fig1, use_container_width=True)

        # =========================
        # MAE GRAPH
        # =========================
        st.subheader("📊 MAE Comparison (Lower is Better)")

        fig2 = px.bar(
            results_df,
            x="Model",
            y="MAE",
            color="Model",
            text="MAE",
            color_discrete_sequence=px.colors.sequential.Oranges
        )
        st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # R2 GRAPH
        # =========================
        st.subheader("📈 R² Score Comparison (Higher is Better)")

        fig3 = px.bar(
            results_df,
            x="Model",
            y="R2 Score",
            color="Model",
            text="R2 Score",
            color_discrete_sequence=px.colors.sequential.Greens
        )
        st.plotly_chart(fig3, use_container_width=True)

        # =========================
        # COMBINED GRAPH
        # =========================
        st.subheader("📊 All Metrics Comparison")

        melted = results_df.melt(
            id_vars="Model",
            var_name="Metric",
            value_name="Value"
        )

        fig4 = px.bar(
            melted,
            x="Model",
            y="Value",
            color="Metric",
            barmode="group"
        )
        st.plotly_chart(fig4, use_container_width=True)

        # =========================
        # TABLE
        # =========================
        st.subheader("📋 Detailed Results")
        st.dataframe(results_df)

        # =========================
        # BEST MODEL (DYNAMIC ✅)
        # =========================
        best = results_df.iloc[0]

        st.success(f"""
🏆 Best Model: {best['Model']}

📉 RMSE: {best['RMSE']:.2f}  
📊 MAE: {best['MAE']:.2f}  
📈 R² Score: {best['R2 Score']:.4f}
""")

        # =========================
        # SMART INSIGHT (AUTO)
        # =========================
        st.info(f"""
The **{best['Model']} model** performs best for house price prediction as it achieves the **lowest RMSE and MAE**, 
indicating minimal prediction error, while also maintaining a **high R² score**, meaning it explains most of the variance in house prices.

This makes it the most reliable model for predicting home values in your dataset.
""")

# =========================
# MODEL LOGS PAGE
# =========================
elif page == "Model Logs":

    st.markdown("""<h1 style="display:flex;align-items:center;gap:10px;">
        <span class="material-icons" style="color:#e94560;font-size:2rem;"></span>
        Model Logs</h1>""", unsafe_allow_html=True)

    # =========================
    # INITIALIZE SESSION STATE
    # =========================
    if "logs" not in st.session_state:
        st.session_state.logs = []

    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    if "input_data" not in st.session_state:
        st.session_state.input_data = None

    # =========================
    # MODEL INFO BANNER
    # =========================
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(233,69,96,0.06),rgba(233,69,96,0.04));
                border:1px solid #363a52; border-radius:14px;
                padding:16px 20px; margin-bottom:20px;">
        <span style="font-family:'Plus Jakarta Sans'; color:#e94560; font-size:0.95rem; font-weight:600;">
            Final Model: Gradient Boosting
        </span>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # METRICS
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", "0.5918")
    col2.metric("RMSE", "239332.71")
    col3.metric("MAE", "155148.77")

    st.markdown("---")

    left_l, right_l = st.columns(2, gap="large")

    # =========================
    # COLUMNS LAYOUT
    # =========================
    left_l, right_l = st.columns(2, gap="large")

    # =========================
    # LEFT SIDE
    # =========================
    with left_l:
        st.markdown("#### Features Used")
        features_used = ["total_sqft", "bathrooms", "bedrooms", "balcony", "location", "area_type", "furnishing_status", "availability"]
        st.markdown(" · ".join(features_used))

        st.markdown("#### Pipeline")
        steps = ["Data Cleaning", "Encoding", "Training", "Prediction"]
        st.markdown(" → ".join(
            [f'<span style="color:#e94560;font-family:Plus Jakarta Sans; font-size:0.78rem;font-weight:600;">{s}</span>' for s in steps]
        ), unsafe_allow_html=True)

        st.markdown("#### Dataset Info")
        st.markdown(f"""
        Shape: {df.shape[0]} rows × {df.shape[1]} columns  
        Missing Values: {df.isnull().sum().sum()}
        """)

    # =========================
    # RIGHT SIDE
    # =========================
    with right_l:
        st.markdown("#### Last Prediction")

        if st.session_state.prediction is not None:
            st.markdown(f"""
            <div style="
                background:#252838;
                border:1px solid #e94560;
                border-radius:12px;
                padding:20px;
                text-align:center;
                box-shadow: 0 0 15px rgba(233,69,96,0.4);
            ">
                <p style="color:#a0a8c8; font-size:0.8rem;">Predicted Home Value</p>
                <p style="font-size:2.5rem; color:#e94560; margin:0;">
                    ₹ {st.session_state.prediction:,.0f}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No prediction yet")

        st.markdown("#### Last Input Features")
        if st.session_state.input_data is not None:
            st.dataframe(st.session_state.input_data)
        else:
            st.warning("No input data yet")
        
    # =========================
    # EVENT LOGS
    # =========================
    st.markdown("""<h4 style="display:flex;align-items:center;gap:8px;margin-top:8px;">
        <span class="material-icons" style="color:#5b8dee;font-size:1.2rem;"></span>
        Event Logs
    </h4>""", unsafe_allow_html=True)

    if "logs" not in st.session_state:
        st.session_state.logs = []

    if st.session_state.logs:

        log_html = "".join([
            f'<div style="font-family:DM Mono,monospace; font-size:0.82rem;'
            f'color:#a0a8c8; padding:5px 10px; border-left:2px solid rgba(91,141,238,0.25);'
            f'margin-bottom:4px;">{log}</div>'
            for log in reversed(st.session_state.logs)
        ])

        st.markdown(
            f'<div style="background:rgba(30,33,48,0.95); border:1px solid rgba(54,58,82,1);'
            f'border-radius:10px; padding:12px; max-height:280px; overflow-y:auto;">'
            f'{log_html}</div>',
            unsafe_allow_html=True
        )

        # BUTTONS
        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                "📥 Download Logs",
                data="\n".join(st.session_state.logs),
                file_name="home_value_logs.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with col_b:
            if st.button("🗑️ Clear Logs", use_container_width=True):
                st.session_state.logs = []
                st.rerun()

    else:
        st.info("No logs yet. Run a prediction to see activity here.")