import streamlit as st
import pandas as pd
import pickle
import os
import typing as ty
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1", api_key=os.environ.get("GROQ_API_KEY")
)


class Customer(ty.TypedDict):
    Age: int
    Balance: float
    Gender: ty.Dict[str, int]
    Geography: ty.Dict[str, int]  # Keys are countries, values are either 0 or 1
    EstimatedSalary: float
    Tenure: int
    HasCreditCard: int
    IsActiveMember: int
    NumberOfProducts: int


def load_model(filename):
    with open(filename, "rb") as fb:
        return pickle.load(fb)


def load_all_models():
    models = os.listdir("models")
    model_dict = {}
    for m in models:
        model_dict[os.path.splitext(m)[0]] = load_model("models/{}".format(m))
    return model_dict


# Change this so its more compatible to what one-hot encoding throws
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

def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at Chase Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.
    
    You noticed a customer named {surname} has a {round(probability) * 100, 1}% of churning.
    
    Here is the customer's information:
    {input_dict}
    
    Here is some explanation as to why the customer might be at risk of churning:
    {explanation}
    
    Generate an email to the customer based on their information asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank. 
    
    Make sure to list out a set of incentives to stay based on their information, in bullet points format. Don't ever mention the probability of churning, or the machine learning model to the customer
    """
    
    print("email ---->", prompt)

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant", messages=[{"role": "user", "content": prompt}]
    )
    print(raw_response.choices)
    return raw_response.choices[0].message.content

models = load_all_models()


def make_predictions(input_df, input_dict):
    # xgboost = load_model("models/xgboost-voting_clf.pkl")
    print("MODESL")
    print(models.keys())
    probabilities = {
        "XGBoost": models["xgb_model"].predict_proba(input_df)[0][1],
        "Random Forest": models["rf_model"].predict_proba(input_df)[0][1],
        "K-Nearest Neightbors": models["knn_model"].predict_proba(input_df)[0][1],
        "Decision Trees": models["dt_model"].predict_proba(input_df)[0][1],
        # 'SVM': models['svm_model'].predict_prob(input_df)[0][1]
    }

    # display on frontend
    avg_probability = np.mean(list(probabilities.values()))
    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model} {prob}")
    st.write(f"Average Probability: {avg_probability}")
    return avg_probability


st.title("Customer Churn Prediction App")
st.write("Hello world")

df = pd.read_csv("sample-data/churn.csv")


def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist working at a bank, where you specialize in interpreting and explaining predictions of machine learning models in detail for the bank to understand.
    
    Your machine learning model has predicted that a customer named {surname} has a {round(probability * 100, 1)}% probability of churning, based on the information provided bellow.
    
    Here is the customer's information:
    {input_dict}
    
    Here are the machine learning model's top 10 most important features for predicting churn:
                  feature  importance
    4       NumOfProducts    0.323888
    6      IsActiveMember    0.164146
    1                 Age    0.109550
    9   Geography_Germany    0.091373
    3             Balance    0.052786
    8    Geography_France    0.046463
    11      Gender_Female    0.045283
    10    Geography_Spain    0.036855
    0         CreditScore    0.035005
    7     EstimatedSalary    0.032655
    5           HasCrCard    0.031940
    2              Tenure    0.030054
    12        Gender_Male    0.000000
    
    {pd.set_option('display.max_columns', None)}
    
    Here is a summary of statistics for churned customers:
    {df[df['Exited'] == 1].describe()}
    
    Do not mention the probability of churning or the machine learning model, or say anything like "Based on the machine learning model's prediction and the top 10 most important features". Don't say anything like "Given the customer's information, here's the explanation". Only explain the prediction.
    
    - If the customer has less than 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning
    - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning
    - Your explanation should be based on the customer's information, the summary statistics of churned and non-churned customers, and the feature importances provided.
    
    Generate your interpretation at 3 to 4 sentences long, from 2 to 3 paragraph long. Bellow is an example of how to structure the explanation, do not modify this structure. 
    
    **Customer Risk Assessment:** {surname}
    
    
    **Assessment:** 23% risk of churning.
    
    
    Hargrave's low credit score and relatively high age compared to other churned customers suggest that he is at a moderate risk of churning. However, his tenure and balance are relatively low, which could indicate that he is not yet deeply invested in the bank's services. Overall, it appears that Hargrave is at risk of churning due to a combination of factors, including his demographics and financial situation.

    Despite having a low tenure, Hargrave is still using at least one product from the bank, which suggests that he has some interest in maintaining the relationship. Furthermore, his geography and demographics are relatively typical of churned customers, with a higher propensity to move to other countries and a lower likelihood of being a female customer. However, his low balance and high age may indicate that he is not actively contributing to the bank's services.

    The customer's low balance and high age, combined with a relatively young age compared to churned customers, suggest that Hargrave is relatively new to the bank and may be in the process of assessing the services provided. However, his relatively high age and low credit score may indicate that he is approaching a stage in life where he may be more likely to switch banks or stop using services altogether. Overall, while Hargrave does not present an extremely high risk of churning, there are several factors that suggest he may be at risk of churning in the near future.
    
    
    
    
    Follow this formatting. Do not change it
    
    """
    print("Explanation prompt ---->", prompt)

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview", messages=[{"role": "user", "content": prompt}]
    )
    print(raw_response.choices)
    return raw_response.choices[0].message.content

# - Start your explanation with something like this: "(surnamme) may (not be at/be at) risk of churning because (explanation). Don't include the parenthesis. Don't include both explanations, only one"
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id, selected_customer_surname = selected_customer_option.split(
        " - "
    )
    selected_customer_id = int(selected_customer_id)

    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]
    print(selected_customer)

    col1, col2 = st.columns(2)

    with col1:  # use the with?
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer["CreditScore"]),
        )
        loc = df["Geography"].unique().tolist()
        location = st.selectbox(
            "Location", loc, index=loc.index(selected_customer["Geography"])
        )
        g = df["Gender"].unique().tolist()
        gender = st.radio("Gender", g, index=g.index(selected_customer["Gender"]))
        age = st.number_input(
            "Age", min_value=18, max_value=120, value=int(selected_customer["Age"])
        )
        tenure = st.number_input(
            "Tenure in years",
            min_value=0,
            max_value=50,
            value=int(selected_customer["Tenure"]),
        )

        with col2:
            balance = st.number_input(
                "Balance", min_value=0.0, value=float(selected_customer["Balance"])
            )
            num_products = st.number_input(
                "Number of Products",
                min_value=1,
                max_value=10,
                value=int(selected_customer["NumOfProducts"]),
            )
            has_credit = st.checkbox(
                "Have a Credit Card?", value=bool(selected_customer["HasCrCard"])
            )
            is_active_member = st.checkbox(
                "Is it an active member?",
                value=bool(selected_customer["IsActiveMember"]),
            )
            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=float(selected_customer["EstimatedSalary"]),
            )
    input_df, input_dict = prepare_inputs(
        credit_score,
        location,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_credit,
        is_active_member,
        estimated_salary,
    )
    avg_probability = make_predictions(input_df, input_dict)
    
    # openai
    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])
    email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
    
    st.markdown('---')
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)
    st.markdown('---')
    st.subheader("Personalized Email")
    st.markdown(email)
    
    