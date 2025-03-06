import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from features_selection import dashboard_features_selection
from sentiment_analysis import dashboard_sentiment_analysis
from prediction import dashboard_prediction

# Initialize Dash app
app = dash.Dash(__name__)

# Load data and perform computations for each page
selected_features, corr_matrix = dashboard_features_selection()
sentiment_counts, sentiment_scores, wordcloud = dashboard_sentiment_analysis()
df_original, y_pred = dashboard_prediction()

# Create heatmap for features selection
heatmap_fig = px.imshow(corr_matrix.round(2),
                        text_auto=True,
                        aspect="auto",
                        title="Matrice de corrélation",
                        color_continuous_scale="RdYlBu_r")

# Create charts for sentiment analysis
pie_chart = px.pie(values=sentiment_counts, names=sentiment_counts.index, title='Distribution des sentiments')
scatter_chart = px.scatter(sentiment_scores, x='Date', y='sentiment_score', title='Evolution du sentiment')
wordcloud_chart = px.imshow(wordcloud, title='Wordcloud des articles')

# Prepare data for volatility prediction visualization
real_volatility = df_original['SP500_Volatility_20d'].iloc[:-len(y_pred)]
real_dates = df_original.index[:-len(y_pred)]
pred_dates = df_original.index[-len(y_pred):]

# Create the figure for volatility prediction
volatility_fig = go.Figure()
volatility_fig.add_trace(go.Scatter(x=real_dates, y=real_volatility, mode='lines', name='Volatilité réelle'))
volatility_fig.add_trace(go.Scatter(x=pred_dates, y=y_pred[:, 0], mode='lines', name='Volatilité prédite'))
volatility_fig.update_layout(title='Prédiction de la volatilité', xaxis_title='Date', yaxis_title='Volatilité')

# Define layout with tabs
app.layout = html.Div([
    html.H1("Dashboard Multi-Pages"),
    dcc.Tabs(id="tabs", value='features-selection', children=[
        dcc.Tab(label='Analyse des variables', value='features-selection', children=[
            html.H2("Variables sélectionnées:"),
            html.Ul([html.Li(feature) for feature in selected_features]),
            dcc.Graph(id='heatmap', figure=heatmap_fig)
        ]),
        dcc.Tab(label='Analyse des sentiments', value='sentiment-analysis', children=[
            dcc.Graph(id='pie-chart', figure=pie_chart),
            dcc.Graph(id='scatter-chart', figure=scatter_chart),
            dcc.Graph(id='wordcloud-chart', figure=wordcloud_chart)
        ]),
        dcc.Tab(label='Prédiction de la volatilité', value='volatility-prediction', children=[
            dcc.Graph(id='volatility-chart', figure=volatility_fig)
        ])
    ])
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
