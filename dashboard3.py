import dash
from dash import dcc, html
import plotly.graph_objects as go
from prediction import dashboard_prediction

# Initialize Dash app
app = dash.Dash(__name__)

# Perform prediction
df_original, y_pred = dashboard_prediction()

# Prepare data for visualization
real_volatility = df_original['SP500_Volatility_20d'].iloc[:-len(y_pred)]
real_dates = df_original.index[:-len(y_pred)]
pred_dates = df_original.index[-len(y_pred):]

# Create the figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=real_dates, y=real_volatility, mode='lines', name='Volatilité réelle'))
fig.add_trace(go.Scatter(x=pred_dates, y=y_pred[:, 0], mode='lines', name='Volatilité prédite'))
fig.update_layout(title='Prédiction de la volatilité', xaxis_title='Date', yaxis_title='Volatilité')

# Define layout
app.layout = html.Div([
    html.H1("Prédiction de la volatilité"),
    dcc.Graph(id='volatility-chart', figure=fig)
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
