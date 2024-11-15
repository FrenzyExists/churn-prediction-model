import plotly.graph_objects as go

import plotly.graph_objects as go

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
            textposition='auto'
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
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)  # Increased top margin to fit title
    )
    return fig
