import pandas as pd
import datetime
import numpy as np
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from wordcloud import WordCloud

def scrap_google_news(ticker, nb_news):
    url = f'https://news.google.com/rss/search?q={ticker}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'xml')
    items = soup.findAll('item')

    news = []
    for item in items[:nb_news]:
        news.append({
            'Source': "Google News",
            'Ticker': ticker,
            'Title': item.title.text,
            #'link': item.link.text,
            'Date': item.pubDate.text
        })
    data = pd.DataFrame(news)
    data['Date'] = pd.to_datetime(data['Date'], format="%a, %d %b %Y %H:%M:%S %Z").dt.date
    return data

def scrap_finviz_news(ticker, nb_news=5):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}  # Set User-Agent to avoid getting blocked
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Failed to retrieve data for {ticker}: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the news table
    news_table = soup.find(id='news-table')
    
    news_titles = []
    if news_table:
        rows = news_table.find_all('tr')
        for row in rows[:nb_news]:
            # The news title is usually in the second column
            title_column = row.find_all('td')[1] if len(row.find_all('td')) > 1 else None
            if title_column:
                title = title_column.get_text(strip=True)
                #link = title_column.find('a')['href'] if title_column.find('a') else None
                date = row.find_all('td')[0].get_text(strip=True)
                news_titles.append({'Source': "Finviz",
                                    'Ticker': ticker,
                                    'Title': title,
                                    #'link': link,
                                    'Date': date})
    data = pd.DataFrame(news_titles)
    #Change the date format to have a datetime object
    for i in range(len(data)):
        if("Today" in data.loc[i, 'Date']):
            data.loc[i, 'Date'] = data.loc[i, 'Date'].replace("Today", datetime.datetime.datetime.today().strftime('%b-%d-%y'))
        elif(data.loc[i, 'Date'][0] in '0123456789'):
            data.loc[i, 'Date'] = str(data.loc[i-1, 'Date'])[:10]+" "+data.loc[i, 'Date']
    data['Date'] = pd.to_datetime(data['Date'], format="%b-%d-%y %I:%M%p").dt.date
    return data


def dashboard_sentiment_analysis():
    nb_news = 1000
    df_news = scrap_google_news('SP500', nb_news)
    df_news = pd.concat([df_news, scrap_google_news('SPX', nb_news)], ignore_index=True)
    df_news = pd.concat([df_news, scrap_finviz_news('SPY', nb_news)], ignore_index=True)
    
    # Load sentiment analysis pipeline
    # model from https://huggingface.co/ProsusAI/finbert
    sentiment_analysis = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    
    # Function to get sentiment using BERT
    def get_sentiment_bert(text):
        result = sentiment_analysis(text)
        dico = {"positive":1, "neutral":0, "negative":-1}
        return dico[result[0]['label']], result[0]['score']
    
    # Apply sentiment analysis
    df_news['sentiment_score'], df_news['sentiment_confidence'] = zip(*df_news['Title'].apply(get_sentiment_bert))

    sentiment_counts = df_news['sentiment_score'].replace({1:'Positif', 0:'Neutre', -1:'Négatif'}).value_counts()

    #sentiment_scores = df_news.groupby(['Date'])['sentiment_score'].mean().reset_index() # we don't have enough data so for the presentation we generate data
    sentiment_scores = pd.DataFrame({'Date': pd.date_range(start='2022-01-01', periods=100), 'sentiment_score': np.random.uniform(-1, 1, 100)})

    # Generate a word cloud image
    text = ' '.join(df_news['Title'].values)
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)

    return sentiment_counts, sentiment_scores, wordcloud