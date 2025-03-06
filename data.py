import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import datetime
import numpy as np

def create_df():
    # Extract

    # Yahoo finance
    # List of tickers for the indices
    tickers = {
        'SP500': '^GSPC',
        'NASDAQ': '^IXIC',
        'DJ': '^DJI',
        'Nikkei': '^N225',
        'Stoxx': '^STOXX50E',
        'FTSE': '^FTSE',
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Oil': 'CL=F',
        'Gas': 'NG=F',
        'EUR/USD': 'EURUSD=X',
        'USD/JPY': 'JPY=X',
        'GBP/USD': 'GBPUSD=X',
        'US_10Y': '^TNX',
        'US_2Y': '^IRX',
        'US Corporate Bonds': 'LQD',  # iShares iBoxx $ Investment Grade Corporate Bond ETF
        'US HY Bonds': 'HYG',  # iShares iBoxx $ High Yield Corporate Bond ETF
    }

    # Dictionary to store the data
    data = {}

    for name, ticker in tickers.items():
        index_data = yf.Ticker(ticker).history(period="5y", interval="1d")[["Close", "Volume"]]
        index_data.index = index_data.index.strftime('%Y-%m-%d')
        data[name+'_Close'] = index_data['Close']
        if name not in ['EUR/USD', 'USD/JPY', 'GBP/USD', 'US_10Y', 'US_2Y', 'VIX']:
            data[name+'_Volume'] = index_data['Volume']
        data[name+'_Returns'] = index_data['Close'].pct_change()   #très corrélé avec les log returns
        #data[name+'_Log_Returns'] = np.log(data[name+'_Close']/data[name+'_Close'].shift(1))
        data[name+'_Volatility_20d'] = data[name+'_Returns'].rolling(window=20).std() * np.sqrt(252)


    # Convert to DataFrame
    df = pd.concat(data.values(), keys=data.keys(), axis=1)
    df.index = pd.to_datetime(df.index)


    # ## US interest rates (FRED)
    # get interest rates from FRED for the last 5 years
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=5*365)

    dict_maturities={}
    #maturities = [1/12, 0.25, 0.5,1,2,3,5,7,10,20,30]
    maturities = [1/12] # très grande corrélation entre les taux donc on en garde qu'un
    for i in maturities:
        if(i<1):
            dict_maturities[str(int(i*12))+'M']='DGS'+str(int(i*12))+'MO'
        else:
            dict_maturities[str(i)+'Y']='DGS'+str(i)

    data_ir = pd.DataFrame()
    for key, series_id in dict_maturities.items():
        data_ir[key] = pdr.get_data_fred(series_id, start, end)
    data_ir = data_ir.dropna()/100    #we delete the dates with missing values and convert to percentage

    # add CPI data
    data_cpi = pdr.get_data_fred('CPIAUCSL', start, end)
    data_ir['CPI'] = data_cpi

    data_ir.index = pd.to_datetime(data_ir.index)

    # Geopolitical events
    geopol_events = pd.read_excel("data_gpr_daily_recent.xls", sheet_name="Sheet1", usecols="A:I")
    geopol_events = geopol_events.drop(columns=["DAY"])
    geopol_events.columns = ["N10D", "GPRD", "GPRD_ACT", "GPRD_THREAT", "DATE", "GPRD_MA30", "GPRD_MA7", "EVENT"]
    geopol_events.index = pd.to_datetime(geopol_events["DATE"])
    geopol_events = geopol_events.drop(columns=["DATE"])


    # ## Options volume
    # Source : https://www.cboe.com/us/options/market_statistics/historical_data/
    df_vol_options = pd.read_csv("daily_volume_SPX.csv", index_col=0)
    df_vol_options.index = pd.to_datetime(df_vol_options.index)
    df_vol_options = df_vol_options["Volume"]
    df_vol_options.name = "Volume_Options_SPX"


    # ## Merging datasets
    df = df.merge(data_ir, left_index=True, right_index=True, how="left")
    df = df.merge(geopol_events, left_index=True, right_index=True, how="left")
    df = df.merge(df_vol_options, left_index=True, right_index=True, how="left")


    # # Transform
    df.sort_index(inplace=True)

    df["CPI"] = df["CPI"].ffill() # CPI is monthly so we keep constant value for the month
    df["1M"] = df["1M"].ffill() # if there is no value for the day we keep the last value
    #df["EVENT"] = df["EVENT"].fillna("None") # if there is no event we put None
    df.loc[~df["EVENT"].isna(), "EVENT"] = 1    # replace the NaN values in the geopolitical events by 0 and the others by 1
    df["EVENT"] = df["EVENT"].fillna(0)
    df[["Nikkei_Close", "Nikkei_Volume", "Nikkei_Returns", "Nikkei_Volatility_20d", "Stoxx_Close", "Stoxx_Volume", "Stoxx_Returns", "Stoxx_Volatility_20d", "FTSE_Close", "FTSE_Volume", "FTSE_Returns", "FTSE_Volatility_20d"]] = df[["Nikkei_Close", "Nikkei_Volume", "Nikkei_Returns", "Nikkei_Volatility_20d", "Stoxx_Close", "Stoxx_Volume", "Stoxx_Returns", "Stoxx_Volatility_20d", "FTSE_Close", "FTSE_Volume", "FTSE_Returns", "FTSE_Volatility_20d"]].ffill() # if foreign markets are closed, we keep the last value

    df = df.loc[~df["SP500_Close"].isna()] # we don't try to predict volatility when the market is closed

    df = df[22:-2]  # we remove the first days because we need 20 days to calculate the volatility and the last days because data may be not available because it is too recent


    # On supprime les NaN pour ne pas avoir de problèmes ensuite.
    df = df.dropna()


    # ## Feature engineering
    # Relative strength index
    def rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    # Prepare a dictionary to hold the new columns
    new_columns = {}

    for name, ticker in tickers.items():
        ema_12 = df[name+'_Returns'].ewm(span=12, adjust=False).mean()
        new_columns[name+'_EMA26'] = df[name+'_Returns'].ewm(span=26, adjust=False).mean()

        # Calculate MACD and Signal
        new_columns[name+'_MACD'] = ema_12 - new_columns[name+'_EMA26']
        new_columns[name+'_Signal'] = new_columns[name+'_MACD'].ewm(span=9, adjust=False).mean()

        # Calculate RSI for two different windows
        new_columns[name+'_RSI10'] = rsi(df[name+'_Returns'], 10)
        new_columns[name+'_RSI22'] = rsi(df[name+'_Returns'], 22)

    # Once all new columns are ready, concatenate them to the original DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Drop the NaN values
    df = df.dropna()

    # Drop columns based on correlation analysis (see notebook)
    # Drop the 1M column because the interest rate information is already in the bond prices
    df = df.drop(columns=["1M"])

    # Drop the NASDAQ and Dow Jones close prices, returns, volatility, EMA26, MACD, Signal, RSI10 and RSI22 because they are highly correlated with the S&P 500
    df = df.drop(columns=["NASDAQ_Close", "NASDAQ_Returns", "NASDAQ_Volatility_20d", "NASDAQ_EMA26", "NASDAQ_MACD", "NASDAQ_Signal", "NASDAQ_RSI10", "NASDAQ_RSI22", "DJ_Close", "DJ_Returns", "DJ_Volatility_20d", "DJ_EMA26", "DJ_MACD", "DJ_Signal", "DJ_RSI10", "DJ_RSI22"])

    # USD/JPY_Close, US_10Y_Close and CPI are highly correlated so we decide drop two of them: USD/JPY_Close and US_10Y_Close
    df = df.drop(columns=["USD/JPY_Close", "US_10Y_Close"])

    # Drop US HY Bonds_EMA26 and US HY Bonds_Signal because we think it is redundant information with the US Corporate Bonds
    df = df.drop(columns=["US HY Bonds_EMA26", "US HY Bonds_Signal"])

    return df