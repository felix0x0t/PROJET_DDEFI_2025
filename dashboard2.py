import dash
from dash import dcc, html
import plotly.express as px
from sentiment_analysis import dashboard_sentiment_analysis

# Initialize Dash app
app = dash.Dash(__name__)

sentiment_counts, sentiment_scores, wordcloud = dashboard_sentiment_analysis()

# Create charts
pie_chart = px.pie(values=sentiment_counts, names=sentiment_counts.index, title='Distribution des sentiments')
scatter_chart = px.scatter(sentiment_scores, x='Date', y='sentiment_score', title='Evolution du sentiment')
wordcloud_chart = px.imshow(wordcloud, title='Wordcloud des articles')

# Define layout for the new page
app.layout = html.Div([
    html.H1("Analyse des sentiments"),
    dcc.Graph(id='pie-chart', figure=pie_chart),
    dcc.Graph(id='scatter-chart', figure=scatter_chart),
    dcc.Graph(id='wordcloud-chart', figure=wordcloud_chart)
])

# Run app
if __name__ == '__main__':
    app.run_server(debug=True)
