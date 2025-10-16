# AI-Based Predictive Maintenance for 3D Printers

This project predicts potential failures in 3D printers using synthetic sensor data and machine learning models.
It was developed under the Intel® AI for Manufacturing Certification Program in collaboration with Gujarat Technological University.

##🌐 Live App  

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predictive-maintenance-3d-printer-acplqrybdln8s46oqxrzze.streamlit.app/)

Click the badge above to try the interactive Streamlit app, where you can upload sensor data and view predictive insights.

## Project Objective

To develop a predictive maintenance system that can forecast machine faults before they occur — helping reduce downtime, save costs, and improve productivity in additive manufacturing.

## ⚙️ How It Works

- Data Simulation – Synthetic sensor data (temperature, vibration, print speed, etc.) is generated to mimic real 3D printer behavior.
- Feature Engineering – Data is cleaned, normalized, and key metrics are extracted.
- Model Training – Multiple ML models are trained and compared (Random Forest, XGBoost, LightGBM).
- Deployment – The best-performing model is deployed using Streamlit for easy accessibility.

## 🧱 Project Structure
| File                           | Description                         |
| ------------------------------ | ----------------------------------- |
| `data_collector.py`            | Generates synthetic 3D printer data |
| `data_preprocessor.py`         | Cleans and engineers features       |
| `model_trainer.py`             | Trains and evaluates ML models      |
| `printer_predictive_model.pkl` | Saved trained model                 |
| `streamlit_app.py`             | Streamlit-based web dashboard       |
| `requirements.txt`             | List of dependencies                |
| `README.md`                    | Project documentation               |
| `3d_printer_data_with_time.csv`| Project documentation               |


## 🚀 Machine Learning Models Used
| Model             | Description                             | Key Advantage                   |
| ----------------- | --------------------------------------- | ------------------------------- |
| **Random Forest** | Ensemble of decision trees              | High accuracy & robust to noise |
| **XGBoost**       | Gradient boosting framework             | Excellent performance & speed   |
| **LightGBM**      | Gradient boosting with leaf-wise growth | Efficient on large datasets     |

🏆 LightGBM achieved the best performance and was selected for deployment

## 🖥️ Tech Stack

- Languages: Python
- Libraries: Scikit-learn, LightGBM, XGBoost, Pandas, NumPy, Matplotlib, Seaborn
- Deployment: Streamlit

## 🧑‍💻 Contributors

- Udvi Chauhan – Developer
- Intel® AI for Manufacturing Certification Course – Mentorship & Certification
- Gujarat Technological University - University collaboration
- AMpire 3D Solutions Pvt. Ltd. – Industrial collaboration

## 🌟 Future Enhancements

- Integrate real-time sensor input using Arduino or Raspberry Pi, so predictions happen instantly during printing.
- Explore advanced AI models like LSTM to continuously track sensor data and predict upcoming faults more accurately.
- Implement automatic model retraining with incoming operational data.

## 🔗 Connect
[![LinkedIn](https://www.linkedin.com/in/udvi-chauhan/)
