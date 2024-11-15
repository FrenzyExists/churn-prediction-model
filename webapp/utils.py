import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def create_gauge_chart(probability):
    # Determine color based on churn probability
    if probability < 0.3:
        color = "rgb(0, 128, 0)"  # Intense green
    elif probability < 0.6:
        color = "rgb(255, 165, 0)"  # Intense yellow
    else:
        color = "rgb(255, 0, 0)"  # Intense red

    # Create the gauge chart
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Churn Probability", 'font': {'size': 24, 'color': 'black'}},
            number={'font': {'size': 40, 'color': 'black'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "black",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(0, 128, 0, 0.7)"},
                    {'range': [30, 60], 'color': "rgba(255, 165, 0, 0.7)"},
                    {'range': [60, 100], 'color': "rgba(255, 0, 0, 0.7)"}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 100}
            }
        )
    )

    # Update chart layout
    fig.update_layout(
        plot_bgcolor="rgba(255,255,255,1)",
        paper_bgcolor="rgba(255,255,255,1)",
        font={'color': "black"},
        width=400,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig



def create_model_probability_chart(probabilities):
    models = list(probabilities.keys())
    probs = list(probabilities.values())
    
    fig = go.Figure(data=[
        go.Bar(
            y=models,
            x=probs,
            orientation='h',
            text=[f'{p:.2%}' for p in probs],
            textposition='auto',
        )
    ])

    # Center-align title in the layout
    fig.update_layout(
        title={
            "text": "Churn Probability by Model",
            "x": 0.5,  # Centers the title
            "xanchor": "center"
        },
        xaxis_title="Probability",
        yaxis_title="Models",
        xaxis=dict(
            tickformat=".0%", 
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1],  # Set custom tick values
            ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],  # Custom tick labels
            range=[0, 1]
        ),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)  # Increased top margin to fit title
    )
    return fig


def customer_percentile(df, surname):

    selected_row = df[df['Surname'] == surname].iloc[0]

    # Metrics to consider
    metrics = ['NumOfProducts', 'Balance', 'EstimatedSalary', 'Tenure', 'CreditScore']

    def calculate_percentile(df, metric, value):
        rank = np.sum(df[metric] <= value)  # Count the number of values <= the customer's value
        percentile = (rank / len(df)) * 100  # Scale to percentile between 0 and 100
        return percentile

    # Calculate percentiles for the selected customer
    percentiles = {}
    for metric in metrics:
        value = selected_row[metric]
        percentile = calculate_percentile(df, metric, value)
        percentiles[metric] = percentile

    # Create a horizontal bar chart
    fig = go.Figure()

    # Add a single bar trace for the selected customer
    fig.add_trace(go.Bar(
        x=list(percentiles.values()),  # Percentiles
        y=list(percentiles.keys()),  # Metrics
        orientation='h',  # Horizontal bars
        text=[f"{v:.2f}%" for v in percentiles.values()],  # Show percentage as text
        textposition='auto',
        marker=dict(color='royalblue')  # Color of the bars
    ))

    # Update layout for better visualization
    fig.update_layout(
        title=f"Percentiles for Customer: {surname}",
        xaxis_title="Percentile (%)",
        yaxis_title="Metric",
        height=400,
        margin=dict(l=40, r=40, t=40, b=100),
        showlegend=False  # Hide legend since we have one customer
    )


    return fig
    
