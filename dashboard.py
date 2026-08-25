import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

st.set_page_config(
    page_title="E-commerce Customer Segmentation",
    page_icon="🛒",
    layout="wide",
)

# Load the already-trained models. No training happens during deployment.
scaler = joblib.load(MODELS / "rfm_scaler.joblib")
kmeans = joblib.load(MODELS / "kmeans_model.joblib")
classifier = joblib.load(MODELS / "customer_segment_classifier.joblib")

# Compatibility fix for LogisticRegression models serialized with an older
# scikit-learn version. Newer versions may expect this attribute to exist.
if not hasattr(classifier, "multi_class"):
    classifier.multi_class = "auto"

FEATURES = ["Recency", "Frequency", "Monetary"]
SEGMENT_NAMES = {
    0: "Recent / Occasional",
    1: "At-Risk",
    2: "Regular / Growing",
    3: "VIP / Loyal",
}
SEGMENT_ACTIONS = {
    0: "Encourage repeat purchases with personalized recommendations, bundles, and loyalty incentives.",
    1: "Use win-back campaigns, reminders, limited-time offers, and personalized discounts.",
    2: "Focus on cross-selling, upselling, loyalty benefits, and increasing purchase frequency.",
    3: "Protect and reward these high-value customers with VIP benefits, exclusive offers, and early access.",
}

st.title("🛒 E-commerce Customer Segmentation & Prediction")
st.caption("BIA Capstone Project • RFM-based segmentation with K-Means and a trained classification model")

# Overview
c1, c2, c3, c4 = st.columns(4)
c1.metric("Segments", "4")
c2.metric("RFM Features", "3")
c3.metric("Clustering", "K-Means")
c4.metric("Prediction Model", "Logistic Regression")

st.divider()

st.subheader("Customer Segment Prediction")
st.write("Enter a customer's RFM values. The app applies the same log transformation and scaler used during model development, then predicts the customer segment.")

col1, col2, col3 = st.columns(3)
with col1:
    recency = st.number_input(
        "Recency (days)",
        min_value=0.0,
        value=30.0,
        step=1.0,
        help="Days since the customer's most recent purchase. Lower is generally better."
    )
with col2:
    frequency = st.number_input(
        "Frequency (orders)",
        min_value=1.0,
        value=5.0,
        step=1.0,
        help="Number of unique invoices/orders made by the customer. Higher is generally better."
    )
with col3:
    monetary = st.number_input(
        "Monetary Value (revenue)",
        min_value=0.0,
        value=2000.0,
        step=100.0,
        help="Total customer revenue. Higher indicates greater monetary value."
    )

if st.button("Predict Customer Segment", type="primary", use_container_width=True):
    raw = pd.DataFrame([[recency, frequency, monetary]], columns=FEATURES)
    log_values = np.log1p(raw)
    scaled = scaler.transform(log_values)

    cluster = int(kmeans.predict(scaled)[0])
    classifier_prediction = int(classifier.predict(log_values)[0])
    probability = float(np.max(classifier.predict_proba(log_values)[0])) if hasattr(classifier, "predict_proba") else None

    # K-Means is the primary segmentation definition used in the project.
    st.success(f"Predicted Segment: **{SEGMENT_NAMES.get(cluster, f'Cluster {cluster}')}**")

    r1, r2 = st.columns(2)
    with r1:
        st.metric("K-Means Cluster", cluster)
    with r2:
        st.metric("Classifier Agreement", "Yes" if classifier_prediction == cluster else "No")

    st.info(SEGMENT_ACTIONS.get(cluster, "Use targeted marketing based on this customer's purchasing behavior."))

    if probability is not None:
        st.caption(f"Classifier confidence for predicted class: {probability:.1%}")

st.divider()

st.subheader("How the segments are interpreted")
segment_table = pd.DataFrame({
    "Cluster": [0, 1, 2, 3],
    "Segment": [SEGMENT_NAMES[i] for i in range(4)],
    "Business meaning": [
        "Recent customers with relatively low repeat frequency/spend",
        "Customers with long recency and low purchase activity",
        "Middle-value customers with moderate engagement",
        "Highly engaged, high-value customers"
    ]
})
st.dataframe(segment_table, use_container_width=True, hide_index=True)

st.subheader("Project Methodology")
st.markdown(
    """
    **Data → Cleaning → Customer-level 80/20 split → RFM → Log transform → StandardScaler → """
    "K-Means / Hierarchical / DBSCAN comparison → K-Means (K=4) → Classification → Final holdout evaluation"
)

st.caption("Note: clustering is unsupervised; the final 20% holdout is used for held-out clustering quality and classifier agreement, not a conventional human-labelled accuracy score.")
