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
    

    

    