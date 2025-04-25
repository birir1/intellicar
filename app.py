import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from models.model_trainer import train_predictive_model  # Your model function

# Page setup
st.set_page_config(
    page_title="Intellicar Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🚗 Intellicar: CAN Data Threat Detection Dashboard")

# Sidebar
st.sidebar.header("📂 Upload Files")
can_file = st.sidebar.file_uploader("Upload CAN Data (.csv)", type=["csv"])
threat_file = st.sidebar.file_uploader("Upload Threat Intel (.json)", type=["json"])

if can_file and threat_file:
    can_data = pd.read_csv(can_file)
    nlp_features = pd.read_json(threat_file)

    st.success("✅ Data successfully uploaded!")

    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Model Performance", "📁 Raw Data"])

    # Train model
    predictions, model, y_test, y_pred = train_predictive_model(can_data, nlp_features)
    accuracy = (y_test == y_pred).mean()

    with tab1:
        st.subheader("🔧 CAN Data Overview")
        st.write(f"Shape: {can_data.shape}")
        st.dataframe(predictions[['speed', 'rpm', 'temp', 'brake', 'predicted_threat']].head())

        # Metric cards
        col1, col2, col3 = st.columns(3)
        col1.metric("Records", len(can_data))
        col2.metric("Detected Threats", predictions["predicted_threat"].sum())
        col3.metric("Model Accuracy", f"{accuracy:.2%}")

        # Chart: Speed vs RPM
        fig1 = px.scatter(predictions, x="speed", y="rpm",
                          color=predictions["predicted_threat"].astype(str),
                          title="Speed vs RPM with Threat Classification",
                          labels={"color": "Threat Predicted"})
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.subheader("🧠 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig2)

        st.markdown(f"**Model Used**: `{model.__class__.__name__}`")
        st.markdown("You can try improving performance by uploading more detailed CAN datasets and rich threat intelligence.")

    with tab3:
        st.subheader("📁 Raw Files")
        st.write("**CAN Data**")
        st.dataframe(can_data)

        st.write("**Threat Intelligence**")
        st.json(nlp_features.to_dict())
else:
    st.info("👈 Please upload both CAN data (.csv) and Threat Intel (.json) to get started.")
