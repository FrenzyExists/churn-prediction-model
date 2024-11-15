# Churn Prediction Model

A Machine Learning project as part of the Headstarter 

## Overview

This project predicts bank customer churn using machine learning algorithms. Users can input customer details to assess the likelihood of churn and view visualizations of the predictions.

## Features

- Machine Learning Models: Utilizes XGBoost, Random Forest, K Nearest Neightbors and Decision Trees 
- Interactive UI: Built with Streamlit for intuitive UI and Plotly for interative customer data input and prediction visualization.
- Personalized Insights: Uses Llama LLMs to generates explanations for predictions as well as customized emails for customers.
- Includes a backend that accepts raw json data as input and provides probability of customer to churn

## Technologies Used

Machine Learning: XGBoost, Scikit Learn
UI: Streamlit
Data Visualization: Plotly
Data Processing: Pandas, NumPy, Matplotlib, Python, Seaborn
Backend: Flask
Getting Started

Clone the repository:
```bash
git clone https://github.com/FrenzyExists/churn-prediction-model
```

Install required packages:
```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run webapp/app.py
```

Run the backend:

```bash
python webapp/backend/server.py
```