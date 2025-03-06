import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from features_selection import dashboard_features_selection

# Initialize Dash app
app = dash.Dash(__name__)

selected_features, corr_matrix = dashboard_features_selection()

# Create heatmap
fig = px.imshow(corr_matrix.round(2),
                text_auto=True,
                aspect="auto",
                title="Matrice de corrélation",
                color_continuous_scale="RdYlBu_r")

# Define layout
app.layout = html.Div([
    html.H1("Analyse des variables"),
    html.H2("Variables sélectionnées:"),
    html.Ul([html.Li(feature) for feature in selected_features]),
    dcc.Graph(id='heatmap', figure=fig)
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
