
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "isolation_forest_pipeline.joblib"

st.set_page_config(
    page_title="NetGuard AI | Network Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PREMIUM DASHBOARD THEME
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #06101f;
        --panel: #0b192c;
        --panel2: #0e2038;
        --blue: #38bdf8;
        --cyan: #22d3ee;
        --green: #34d399;
        --red: #fb7185;
        --text: #edf5ff;
        --muted: #8da5c4;
        --border: rgba(103, 166, 232, 0.20);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 75% 0%, #10294a 0%, transparent 38%),
            linear-gradient(135deg, #06101f 0%, #081426 55%, #06101f 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(6, 16, 31, 0.82);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071426, #091b30);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #c4d5ec;
    }

    h1, h2, h3 {
        color: #f1f7ff !important;
        letter-spacing: -0.6px;
    }

    p, label, li {
        color: #c3d4e9;
    }

    .brand {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2.5px;
        color: #38bdf8;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero {
        padding: 30px;
        border: 1px solid rgba(56, 189, 248, 0.27);
        border-radius: 22px;
        background:
            radial-gradient(circle at 90% 10%, rgba(34,211,238,.14), transparent 35%),
            linear-gradient(125deg, rgba(16,42,72,.97), rgba(7,20,39,.98));
        box-shadow: 0 16px 45px rgba(0,0,0,.18);
        margin-bottom: 22px;
    }

    .hero h1 {
        font-size: clamp(27px, 3vw, 40px);
        line-height: 1.15;
        margin: 8px 0 12px 0;
        font-weight: 800;
    }

    .hero p {
        color: #a9bfd9;
        font-size: 14px;
        line-height: 1.8;
        max-width: 720px;
    }

    .eyebrow {
        color: #38bdf8;
        font-size: 11px;
        letter-spacing: 2px;
        font-weight: 800;
        text-transform: uppercase;
    }

    .panel {
        background: linear-gradient(145deg, #0d2037, #09182a);
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .panel-title {
        color: #f0f7ff;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .panel-description {
        color: #91a9c8;
        font-size: 12px;
        line-height: 1.7;
    }

    .metric-card {
        background: linear-gradient(145deg, #102743, #0a192c);
        border: 1px solid rgba(89, 157, 226, .23);
        border-radius: 15px;
        padding: 18px;
        min-height: 112px;
    }

    .metric-label {
        color: #92accb;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #f4f8ff;
        font-size: 27px;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-note {
        color: #38bdf8;
        font-size: 11px;
        margin-top: 8px;
    }

    .status {
        display: inline-block;
        border-radius: 20px;
        padding: 7px 12px;
        font-size: 11px;
        font-weight: 700;
        border: 1px solid;
    }

    .status-ok {
        color: #6ee7b7;
        background: rgba(16,185,129,.10);
        border-color: rgba(52,211,153,.35);
    }

    .status-error {
        color: #fda4af;
        background: rgba(244,63,94,.10);
        border-color: rgba(251,113,133,.35);
    }

    .workflow-step {
        background: #0b1d32;
        border: 1px solid var(--border);
        padding: 14px 7px;
        border-radius: 12px;
        text-align: center;
        height: 100%;
    }

    .workflow-icon {
        font-size: 24px;
        margin-bottom: 8px;
    }

    .workflow-name {
        font-size: 11px;
        font-weight: 700;
        color: #dbeafe;
    }

    .section-kicker {
        color: #38bdf8;
        text-transform: uppercase;
        font-size: 10px;
        letter-spacing: 2px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    div[data-testid="stFileUploader"] {
        border: 1px dashed rgba(56,189,248,.45);
        background: rgba(13,32,55,.65);
        padding: 10px;
        border-radius: 14px;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(56,189,248,.38);
        background: linear-gradient(120deg, #075985, #0369a1);
        color: white;
        font-weight: 700;
        min-height: 42px;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: #67e8f9;
        color: white;
        background: #075985;
    }

    [data-testid="stMetric"] {
        background: #0b1d32;
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #9db4d0;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: var(--border);
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .hero { padding: 20px; }
        .metric-value { font-size: 22px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. REUSABLE COMPONENTS
# =========================================================

def section_heading(title, description=None):
    st.markdown(
        f"""
        <div style="margin: 22px 0 14px 0;">
            <div class="section-kicker">NETGUARD AI / ANALYTICS</div>
            <h2 style="font-size:24px;margin:0;">{title}</h2>
            <div style="color:#91a9c8;font-size:13px;margin-top:6px;">
                {description or ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def styled_chart(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#c4d5ec", size=11),
        margin=dict(l=15, r=15, t=35, b=15),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c4d5ec"),
        ),
        xaxis=dict(
            gridcolor="rgba(117,160,205,.10)",
            zerolinecolor="rgba(117,160,205,.10)",
        ),
        yaxis=dict(
            gridcolor="rgba(117,160,205,.10)",
            zerolinecolor="rgba(117,160,205,.10)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)


def read_csv_safely(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to read this CSV file: {exc}")
        return None


# =========================================================
# 4. MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner="Loading Isolation Forest model...")
def load_model(model_path, modified_time):
    # modified_time invalidates the cache if the model file changes.
    return joblib.load(model_path)


model = None
model_error = None

if MODEL_PATH.is_file():
    try:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as exc:
        model_error = str(exc)
else:
    model_error = f"Model file not found: {MODEL_PATH}"


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 0 22px 0;">
            <div style="font-size:37px;">🛡️</div>
            <div style="font-size:21px;font-weight:800;color:#f1f7ff;">
                NetGuard <span style="color:#38bdf8;">AI</span>
            </div>
            <div style="font-size:11px;color:#8da5c4;margin-top:6px;">
                INTELLIGENT NETWORK SECURITY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("##### WORKSPACE")

    page = st.radio(
        "Navigation",
        ["Overview", "Dataset Analytics", "Anomaly Detection", "About Project"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### SYSTEM STATUS")

    if model is not None:
        st.markdown(
            '<span class="status status-ok">● MODEL OPERATIONAL</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status status-error">● MODEL UNAVAILABLE</span>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="margin-top:24px;padding:15px;background:#0b1d32;
                    border:1px solid rgba(103,166,232,.2);border-radius:12px;">
            <div style="font-size:11px;color:#38bdf8;font-weight:800;">
                PROJECT STACK
            </div>
            <div style="font-size:12px;line-height:2;color:#a9bfd9;margin-top:8px;">
                ◈ Python & Streamlit<br>
                ◈ Pandas & NumPy<br>
                ◈ Isolation Forest<br>
                ◈ NSL-KDD Dataset
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="position:relative;margin-top:30px;color:#6985a7;
                    font-size:10px;line-height:1.8;">
            NETWORK LOG ANOMALY DETECTION<br>
            MACHINE LEARNING PROJECT<br>
            <span style="color:#38bdf8;">BUILT WITH STREAMLIT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 6. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <div class="brand">🛡️ AI-POWERED SECURITY INTELLIGENCE</div>
            <h1>Network Log<br> <span style="color:#38bdf8;">
            Anomaly Detection</span></h1>
            <p>
                Identify unusual network connection patterns with an
                unsupervised machine learning pipeline. Upload network
                traffic records, inspect your dataset, and flag potentially
                anomalous connections using Isolation Forest.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("DATASET", "NSL-KDD", "Network intrusion benchmark")

    with col2:
        metric_card("FEATURES", "41", "Expected network features")

    with col3:
        metric_card("ALGORITHM", "IForest", "Unsupervised learning")

    with col4:
        metric_card(
            "MODEL STATUS",
            "READY" if model is not None else "OFFLINE",
            "Pipeline availability",
        )

    section_heading(
        "Detection Pipeline",
        "From raw network records to interpretable anomaly predictions.",
    )

    workflow = [
        ("📥", "Network Logs"),
        ("🧹", "Preprocessing"),
        ("🧬", "Feature Pipeline"),
        ("🌲", "Isolation Forest"),
        ("🚨", "Anomaly Flags"),
        ("📊", "Results"),
    ]

    columns = st.columns(len(workflow))

    for column, (icon, name) in zip(columns, workflow):
        with column:
            st.markdown(
                f"""
                <div class="workflow-step">
                    <div class="workflow-icon">{icon}</div>
                    <div class="workflow-name">{name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    section_heading("System Overview", "Core capabilities of the application.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📡 Network Intelligence</div>
                <div class="panel-description">
                    Explore network traffic records, feature distributions,
                    missing values, numerical summaries, and categorical
                    attributes before model inference.
                </div>
            </div>
            <div class="panel">
                <div class="panel-title">🌲 Unsupervised Detection</div>
                <div class="panel-description">
                    Apply the saved Isolation Forest pipeline to identify
                    records that differ from the learned patterns of normal
                    network traffic.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📈 Visual Analytics</div>
                <div class="panel-description">
                    Review normal-versus-anomaly counts, percentage
                    distributions, prediction tables, and dataset statistics.
                </div>
            </div>
            <div class="panel">
                <div class="panel-title">📦 Exportable Results</div>
                <div class="panel-description">
                    Download prediction results in CSV format for further
                    analysis, reporting, or integration into your workflow.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_heading("Model Health", "Current saved model status.")

    if model is not None:
        st.success(
            "The saved model loaded successfully and is available for inference."
        )
    else:
        st.error(
            "The model could not be loaded. Check the model path and "
            "dependency versions."
        )
        with st.expander("View model loading details"):
            st.code(model_error or "Unknown loading error")


# =========================================================
# 7. DATASET ANALYTICS PAGE
# =========================================================

elif page == "Dataset Analytics":

    section_heading(
        "Dataset Analytics",
        "Inspect your uploaded network traffic CSV before making predictions.",
    )

    uploaded_file = st.file_uploader(
        "Upload network traffic data",
        type=["csv"],
        key="analysis_upload",
        help="Upload a CSV containing network connection records.",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size:35px;">📂</div>
                <div class="panel-title">Your dataset workspace</div>
                <div class="panel-description">
                    Upload a CSV to unlock dataset previews, data quality
                    checks, feature summaries, and exploratory charts.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        data = read_csv_safely(uploaded_file)

        if data is not None:
            if data.empty:
                st.warning("The uploaded CSV contains no data.")
            else:
                st.success(
                    f"Dataset loaded: {len(data):,} records and "
                    f"{len(data.columns):,} columns."
                )

                missing_total = int(data.isna().sum().sum())
                duplicate_count = int(data.duplicated().sum())

                a, b, c, d = st.columns(4)

                with a:
                    st.metric("Total Records", f"{len(data):,}")

                with b:
                    st.metric("Total Columns", f"{len(data.columns):,}")

                with c:
                    st.metric("Missing Values", f"{missing_total:,}")

                with d:
                    st.metric("Duplicate Rows", f"{duplicate_count:,}")

                section_heading("Data Preview", "First records from your uploaded file.")

                st.dataframe(
                    data.head(20),
                    use_container_width=True,
                    hide_index=True,
                )

                section_heading("Data Quality", "Missing-value distribution by feature.")

                missing_by_column = data.isna().sum()
                missing_by_column = missing_by_column[missing_by_column > 0]

                if len(missing_by_column):
                    missing_df = (
                        missing_by_column.rename("Missing Values")
                        .reset_index()
                        .rename(columns={"index": "Feature"})
                        .sort_values("Missing Values", ascending=False)
                    )

                    fig = px.bar(
                        missing_df,
                        x="Feature",
                        y="Missing Values",
                        color="Missing Values",
                        color_continuous_scale=["#38bdf8", "#fb7185"],
                        title="Missing Values by Feature",
                    )
                    styled_chart(fig, 320)
                else:
                    st.success("No missing values were found in this dataset.")

                section_heading(
                    "Feature Explorer",
                    "Choose a numeric feature to understand its distribution.",
                )

                numeric_columns = data.select_dtypes(
                    include=np.number
                ).columns.tolist()

                if numeric_columns:
                    selected_feature = st.selectbox(
                        "Select numeric feature",
                        numeric_columns,
                    )

                    feature_values = data[selected_feature].dropna()

                    if len(feature_values):
                        fig = px.histogram(
                            x=feature_values,
                            nbins=35,
                            title=f"Distribution of {selected_feature}",
                            labels={"x": selected_feature, "y": "Record Count"},
                            color_discrete_sequence=["#38bdf8"],
                        )
                        styled_chart(fig, 330)
                    else:
                        st.info("This feature has no numeric values to display.")
                else:
                    st.info("No numeric features were found in the uploaded CSV.")

                section_heading(
                    "Statistical Summary",
                    "Descriptive statistics for numeric features.",
                )

                summary = data.describe(include="all").transpose()
                st.dataframe(summary, use_container_width=True)

                section_heading(
                    "Column Information",
                    "Data types and missing values for each column.",
                )

                column_info = pd.DataFrame({
                    "Feature": data.columns.astype(str),
                    "Data Type": data.dtypes.astype(str).values,
                    "Non-Null Values": data.notna().sum().values,
                    "Missing Values": data.isna().sum().values,
                    "Unique Values": data.nunique(dropna=True).values,
                })

                st.dataframe(
                    column_info,
                    use_container_width=True,
                    hide_index=True,
                )


# =========================================================
# 8. ANOMALY DETECTION PAGE
# =========================================================

elif page == "Anomaly Detection":

    section_heading(
        "Anomaly Detection Engine",
        "Score uploaded network records using your saved Isolation Forest pipeline.",
    )

    if model is None:
        st.error("The machine learning model is unavailable.")

        st.markdown(
            "Expected model location: "
            f"`{MODEL_PATH}`"
        )

        st.markdown(
            "Train or restore your model, then restart or refresh the application."
        )

        with st.expander("Technical details"):
            st.code(model_error or "Unknown model loading error")

    else:
        st.markdown(
            '<span class="status status-ok">● ISOLATION FOREST LOADED</span>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        uploaded_file = st.file_uploader(
            "Upload a network traffic CSV file",
            type=["csv"],
            key="prediction_upload",
            help=(
                "Use the same feature schema expected by the saved "
                "training pipeline."
            ),
        )

        if uploaded_file is None:
            st.markdown(
                """
                <div class="panel">
                    <div style="font-size:35px;">🔍</div>
                    <div class="panel-title">Ready to analyze network traffic</div>
                    <div class="panel-description">
                        Upload a compatible CSV file to generate predictions,
                        review suspicious records, and export the results.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            data = read_csv_safely(uploaded_file)

            if data is not None:
                if data.empty:
                    st.warning("The uploaded CSV contains no records.")
                else:
                    st.markdown("##### Uploaded Data Preview")

                    st.dataframe(
                        data.head(10),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Target/metadata columns are not model input features.
                    excluded_columns = [
                        "label",
                        "difficulty",
                        "is_anomaly",
                        "Prediction",
                    ]

                    prediction_data = data.drop(
                        columns=[
                            col for col in excluded_columns
                            if col in data.columns
                        ],
                        errors="ignore",
                    )

                    if prediction_data.shape[1] == 0:
                        st.error("No model input features remain after removing metadata.")
                    else:
                        if st.button(
                            "🚀 Analyze Network Traffic",
                            use_container_width=True,
                        ):
                            with st.spinner(
                                "Analyzing network records..."
                            ):
                                try:
                                    predictions = np.asarray(
                                        model.predict(prediction_data)
                                    )

                                    # Scikit-learn IsolationForest convention:
                                    # +1 = inlier (normal), -1 = outlier (anomaly).
                                    # Validate instead of silently mislabeling
                                    # a model with a different output convention.
                                    unique_predictions = set(
                                        np.unique(predictions).tolist()
                                    )

                                    if not unique_predictions.issubset({-1, 1}):
                                        raise ValueError(
                                            "Unexpected prediction values "
                                            f"{unique_predictions}. This app "
                                            "expects scikit-learn Isolation "
                                            "Forest outputs of -1 and +1."
                                        )

                                    if len(predictions) != len(data):
                                        raise ValueError(
                                            "The number of predictions does "
                                            "not match the number of records."
                                        )

                                    statuses = np.where(
                                        predictions == 1,
                                        "Normal",
                                        "Anomaly",
                                    )

                                    result = data.copy()
                                    result["Prediction"] = statuses

                                    normal_count = int(
                                        np.sum(statuses == "Normal")
                                    )
                                    anomaly_count = int(
                                        np.sum(statuses == "Anomaly")
                                    )
                                    total_count = len(statuses)

                                    anomaly_rate = (
                                        anomaly_count / total_count * 100
                                        if total_count
                                        else 0
                                    )

                                    st.session_state[
                                        "ng_results"
                                    ] = result

                                    st.session_state[
                                        "ng_summary"
                                    ] = {
                                        "total": total_count,
                                        "normal": normal_count,
                                        "anomaly": anomaly_count,
                                        "rate": anomaly_rate,
                                    }

                                except Exception as exc:
                                    st.error(
                                        f"Prediction failed: {exc}"
                                    )

                                    st.warning(
                                        "Check that the CSV has the same "
                                        "feature names, data types, and "
                                        "feature schema used to train the "
                                        "saved pipeline. Do not upload "
                                        "target labels as input features."
                                    )

                # Display the most recent successful analysis.
                # Results remain visible during normal Streamlit reruns.
                if "ng_results" in st.session_state:
                    result = st.session_state["ng_results"]
                    summary = st.session_state["ng_summary"]

                    st.markdown("---")

                    section_heading(
                        "Detection Summary",
                        "Prediction totals for the most recent successful analysis.",
                    )

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.metric(
                            "Total Records",
                            f"{summary['total']:,}",
                        )

                    with c2:
                        st.metric(
                            "Normal",
                            f"{summary['normal']:,}",
                        )

                    with c3:
                        st.metric(
                            "Anomalies",
                            f"{summary['anomaly']:,}",
                        )

                    with c4:
                        st.metric(
                            "Anomaly Rate",
                            f"{summary['rate']:.2f}%",
                        )

                    left, right = st.columns([1.15, 1])

                    with left:
                        section_heading(
                            "Prediction Distribution",
                            "Number of records in each predicted class.",
                        )

                        counts = result["Prediction"].value_counts()

                        distribution = pd.DataFrame({
                            "Status": ["Normal", "Anomaly"],
                            "Records": [
                                int(counts.get("Normal", 0)),
                                int(counts.get("Anomaly", 0)),
                            ],
                        })

                        fig = px.bar(
                            distribution,
                            x="Status",
                            y="Records",
                            color="Status",
                            color_discrete_map={
                                "Normal": "#34d399",
                                "Anomaly": "#fb7185",
                            },
                            text="Records",
                        )

                        fig.update_traces(
                            textposition="outside",
                            cliponaxis=False,
                        )

                        styled_chart(fig, 320)

                    with right:
                        section_heading(
                            "Class Composition",
                            "Share of normal and anomalous records.",
                        )

                        pie_data = pd.DataFrame({
                            "Status": ["Normal", "Anomaly"],
                            "Count": [
                                summary["normal"],
                                summary["anomaly"],
                            ],
                        })

                        fig = go.Figure(
                            data=[
                                go.Pie(
                                    labels=pie_data["Status"],
                                    values=pie_data["Count"],
                                    hole=0.68,
                                    marker=dict(
                                        colors=["#34d399", "#fb7185"],
                                        line=dict(
                                            color="#081426",
                                            width=4,
                                        ),
                                    ),
                                    textinfo="label+percent",
                                    textfont=dict(size=12),
                                )
                            ]
                        )

                        fig.update_layout(
                            annotations=[
                                dict(
                                    text=f"{summary['rate']:.1f}%<br>ANOMALIES",
                                    x=0.5,
                                    y=0.5,
                                    showarrow=False,
                                    font=dict(
                                        size=15,
                                        color="#f1f7ff",
                                    ),
                                )
                            ]
                        )

                        styled_chart(fig, 320)

                    section_heading(
                        "Detection Results",
                        "Inspect the predicted status for each network connection.",
                    )

                    filter_option = st.selectbox(
                        "Filter records",
                        [
                            "All Records",
                            "Anomalies Only",
                            "Normal Only",
                        ],
                    )

                    if filter_option == "Anomalies Only":
                        filtered_result = result[
                            result["Prediction"] == "Anomaly"
                        ]
                    elif filter_option == "Normal Only":
                        filtered_result = result[
                            result["Prediction"] == "Normal"
                        ]
                    else:
                        filtered_result = result

                    st.dataframe(
                        filtered_result,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        f"Showing {len(filtered_result):,} of "
                        f"{len(result):,} records."
                    )

                    csv_bytes = result.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Complete Detection Results",
                        data=csv_bytes,
                        file_name="network_anomaly_detection_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.info(
                        "Interpretation: Isolation Forest identifies "
                        "outliers relative to the learned data patterns. "
                        "An anomaly prediction is not, by itself, proof "
                        "of a cyberattack."
                    )


# =========================================================
# 9. ABOUT PROJECT PAGE
# =========================================================

elif page == "About Project":

    section_heading(
        "About NetGuard AI",
        "Machine learning for network traffic anomaly analysis.",
    )

    st.markdown(
        """
        <div class="hero">
            <div class="brand">PROJECT PROFILE</div>
            <h1>Intelligent Network<br>
            <span style="color:#38bdf8;">Threat Analytics</span></h1>
            <p>
                This project uses an Isolation Forest pipeline to flag
                unusual network connection records. The application
                combines dataset exploration, model inference, and
                interactive result analysis in a single interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_heading("Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">🐍 Python</div>
                <div class="panel-description">
                    Core programming language for preprocessing and inference.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tech2:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">🌲 Isolation Forest</div>
                <div class="panel-description">
                    Unsupervised outlier detection for network connection records.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tech3:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">📊 Streamlit & Plotly</div>
                <div class="panel-description">
                    Interactive dashboard components and data visualization.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    section_heading("How Detection Works")

    st.markdown(
        """
        1. **Upload:** Load network connection records from a CSV file.
        2. **Prepare:** Exclude target or metadata columns and pass the
           remaining features to the saved pipeline.
        3. **Predict:** The trained Isolation Forest pipeline returns
           an inlier or outlier prediction for each record.
        4. **Interpret:** The application displays the predicted classes,
           counts, percentages, and distribution.
        5. **Export:** Download the records with their predicted labels.
        """
    )

    st.warning(
        "Model predictions depend on the training data, preprocessing "
        "pipeline, and input feature schema. Validate the model on "
        "labeled test data before drawing conclusions about real attacks."
    )


# =========================================================
# 10. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;padding:8px 0 18px 0;">
        <div style="color:#38bdf8;font-size:11px;font-weight:800;
                    letter-spacing:2px;">
            🛡️ NETGUARD AI
        </div>
        <div style="color:#7892b2;font-size:11px;margin-top:7px;">
            Network Log Anomaly Detection · Powered by Machine Learning
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
