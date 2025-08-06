import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app settings
st.set_page_config(page_title="3D Printer Predictive Maintenance", layout="wide")
st.title("🔧 3D Printer Predictive Maintenance")
st.markdown("""
This app uses a machine learning model to **predict maintenance needs** for 3D printers based on sensor data.
Upload your `.csv` file with the required input features to get predictions.
""")

# Load trained model (cached for performance)
@st.cache_resource
def load_model():
    try:
        model = joblib.load("printer_predictive_model.pkl")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# Upload input data
st.sidebar.header("📤 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Preview of Uploaded Data")
        st.dataframe(df)

        if model:
            if st.button("🚀 Predict Maintenance"):
                try:
                    # ✅ Drop Timestamp column if it exists
                    if 'Timestamp' in df.columns:
                        df = df.drop('Timestamp', axis=1)

                    # ✅ Make predictions
                    predictions = model.predict(df)
                    st.success("✅ Prediction completed!")

                    # ✅ Show Prediction Table
                    st.subheader("📈 Prediction Results")
                    results = df.copy()
                    results['Predicted_Status'] = predictions
                    st.dataframe(results)

                    # ✅ Fault Type Distribution Chart
                    st.subheader("📊 Fault Type Distribution")
                    fault_counts = results['Predicted_Status'].value_counts()

                    fig, ax = plt.subplots()
                    sns.barplot(x=fault_counts.index, y=fault_counts.values, ax=ax)
                    ax.set_xlabel("Fault Type")
                    ax.set_ylabel("Number of Occurrences")
                    ax.set_title("Predicted Fault Distribution")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.warning("Model is not loaded. Please check your `printer_predictive_model.pkl` file.")
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
else:
    st.info("Upload a CSV file to start predictions.")
