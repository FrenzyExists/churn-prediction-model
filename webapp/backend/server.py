from flask import Flask, request
import pandas as pd
import numpy as np
import sys, os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils as ut  # Now you can import utils.py

def prepare_inputs(
    credit_score,
    location,
    gender,
    age,
    tenure,
    balance,
    num_products,
    has_credit_card,
    is_active_member,
    estimated_salary,
):
    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Gender_Female": 1 if gender == "Female" else 0,
    }
    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

app = Flask(__name__)


@app.route("/")
def home_endpoint():
    return "Welcome to the Churn Predictor API!"

@app.route("/predict", methods=["POST"])
def get_prediction():
    if request.method == "POST":
        req = request.get_data(as_text=True)
        
        print("Requested")
        print("TYPE", type(req))
        # Get data posted as JSON
        data = json.loads(req)
        
        print("DATA",data)
        if not data:
            return {"error": "Invalid or empty JSON body"}, 400  # Invalid JSON

        print("Received Data:", data)

        # Assuming data is a dictionary, convert it into a DataFrame
        df = pd.DataFrame([data])
        
        has_cr_card_value = df['HasCrCard'].iloc[0]  # Access the first row value
        has_cr_card_bool = bool(has_cr_card_value)

        IsActiveMember_value = df['IsActiveMember'].iloc[0]  # Access the first row value
        IsActiveMember_bool = bool(IsActiveMember_value)
                
        print("DataFrame:", df)
        print("DDSSDDS", df['HasCrCard'])
        input_df, input_dict  = prepare_inputs(
            df['CreditScore'].loc[0],
            df['Geography'].loc[0],
            df['Gender'].loc[0],
            int(df['Age'].loc[0]),
            int(df['Tenure'].loc[0]),
            float(df['Balance'].loc[0]),
            int(df['NumOfProducts'].loc[0]),
            has_cr_card_bool,
            IsActiveMember_bool,
            float(df['EstimatedSalary'].loc[0])
        )
        # Assuming the model is already loaded
        prediction = model.predict_proba(input_df)
        # avg_probability = np.mean(list(probabilities.values()))
        try:
            print("Prediction:", prediction)
            p = prediction[0][1]
            av = prediction[0][0]
            
            # Convert prediction to a Python float to avoid issues with serialization
            p = float(p)
            av = float(av)
            print(p, av)
            print("WTC")
            dddd = {"avg_probability": av, "prediction": p}, 200
            print(dddd)
            # Return the probability and prediction
            return dddd
        except Exception as e:
            return {"error": f"Error processing request: {str(e)}"}, 400  # Catch any exception and return error
    else:
        # Method Not Allowed error for any method that's not POST
        return {"error": "Method Not Allowed. Use POST."}, 405


if __name__ == '__main__':
    model = ut.load_model("models/xgb_model.pkl")  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80, debug=True)