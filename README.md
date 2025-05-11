# california-house-price-predictor
A machine learning project to predict California housing prices using demographic and geographic data. Includes EDA, outlier handling, and training with Linear Regression, Random Forest, and XGBoost. Evaluated using RMSE, MAE, and R² metrics. Deployed as an interactive Streamlit app for real-time predictions.
🏠 California House Price Prediction
This project predicts median house values in California districts using the California Housing dataset. It includes a complete data science workflow—from data cleaning and EDA to model training, evaluation, and deployment with an interactive Streamlit web app.

📌 Project Overview
Objective: Build a regression model to predict median_house_value based on features like income, location, and housing characteristics.

Dataset: California Housing Dataset (via scikit-learn or similar CSV format)

Tech Stack: Python, pandas, seaborn, matplotlib, scikit-learn, Streamlit

📊 Workflow Summary
1. Exploratory Data Analysis (EDA)
Handled missing values in total_bedrooms

Detected and removed outliers using boxplots and IQR method

Visualized distributions and feature-target relationships

Examined feature correlations with a heatmap

2. Data Preprocessing
One-hot encoded the ocean_proximity categorical feature

Normalized input formats

Performed train-test split (80/20)

3. Model Building & Evaluation
Trained a Linear Regression model

Evaluated using RMSE, R² score, and MAE

Planned to extend with Random Forest and XGBoost (optional)

4. Deployment
Developed a Streamlit app

Allows users to input values and get real-time price predictions

🚀 How to Run
Option 1: Locally
bash
Copy
Edit
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
streamlit run app.py  # or the .py file you've saved
Option 2: Streamlit Cloud
Upload your repo to GitHub

Go to Streamlit Cloud

Deploy by linking your GitHub and selecting the main file (e.g., app.py)

📈 Future Improvements
Add support for multiple model selection (Random Forest, XGBoost)

Improve UI/UX with sliders and layout

Add actual vs predicted plot, residuals, feature importance

Save and load model with joblib or pickle


