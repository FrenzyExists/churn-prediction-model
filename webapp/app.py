import streamlit as st
import pandas as pd


st.title("Customer Churn Prediction App")
st.write("Hello world")

df = pd.read_csv("sample-data/churn.csv")

customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id, selected_customer_surname = selected_customer_option.split(" - ")
    selected_customer_id = int(selected_customer_id)
    
    selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]
    print(selected_customer)
    
    col1, col2 = st.columns(2)
    
    with col1: # use the with?
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer['CreditScore'])
        )
        loc = df['Geography'].unique().tolist()
        location = st.selectbox(
            "Location",
            loc,
            index=loc.index(selected_customer['Geography'])
        )
        g = df['Gender'].unique().tolist()
        gender = st.radio(
            'Gender', 
            g,
            index=g.index(selected_customer['Gender'])
        )
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=120,
            value=int(selected_customer['Age'])
        )
        tenure = st.number_input(
            "Tenure in years",
            min_value=0,
            max_value=50,
            value=int(selected_customer['Tenure'])
        )
        
        with col2:
            balance = st.number_input(
                "Balance",
                min_value=0.0,
                value=float(selected_customer['Balance'])
            )
            num_products = st.number_input(
                "Number of Products",
                min_value=1,
                max_value=10,
                value=int(selected_customer['NumOfProducts'])
            )
            has_credit = st.checkbox(
                "Have a Credit Card?",
                value=bool(selected_customer['HasCrCard'])
            )
            is_active_member = st.checkbox(
                "Is it an active member?",
                value=bool(selected_customer['IsActiveMember'])
            )
            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=float(selected_customer['EstimatedSalary'])
            )
    

    