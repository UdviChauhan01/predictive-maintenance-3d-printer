import streamlit as st
import pandas as pd
import joblib

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
                    # Clean up data before prediction
                    df_clean = df.copy()

                    # Drop known unwanted columns
                    if 'Unnamed: 0' in df_clean.columns:
                        df_clean.drop('Unnamed: 0', axis=1, inplace=True)
                    if 'Status' in df_clean.columns:
                        df_clean.drop('Status', axis=1, inplace=True)

                    # Keep only the features used during training
                    if hasattr(model, 'feature_names_in_'):
                        df_clean = df_clean[model.feature_names_in_]

                    predictions = model.predict(df_clean)
                    st.success("Prediction completed!")
                    st.subheader("Prediction Results")
                    st.dataframe(pd.DataFrame({'Prediction': predictions}))
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        else:
            st.warning("Model is not loaded. Please check your `printer_predictive_model.pkl` file.")
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
else:
    st.info("Upload a CSV file to start predictions.")
