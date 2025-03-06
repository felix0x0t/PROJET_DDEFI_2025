import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from data import create_df

def load_data():
    #return create_df()
    # To avoid to many API calls, we will load the data from a CSV file
    df = pd.read_csv("data.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def feature_selection(df_original):
    # Define features and target variable
    X = df_original.drop(columns=["SP500_Volatility_20d"])  # Keep only exogenous variables
    y = df_original["SP500_Volatility_20d"]  # Target variable

    # Standardize features (Elastic Net is sensitive to feature scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Elastic Net
    elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.9, max_iter=5000, random_state=42)
    elastic_net.fit(X_scaled, y)

    # Get feature importance (non-zero coefficients)
    selected_features = X.columns[elastic_net.coef_ != 0]

    df_reduced = df_original[selected_features]
    df_reduced.loc[:, "SP500_Volatility_20d"] = df_original.loc[:, "SP500_Volatility_20d"]

    return df_reduced


def dashboard_features_selection():
    df_original = load_data()
    df_reduced = feature_selection(df_original)
    feature_name_mapping = {
    "Nikkei_Volatility_20d": "Nikkei 20-Day Volatility",
    "Stoxx_Close": "Stoxx Closing Price",
    "Stoxx_Volatility_20d": "Stoxx 20-Day Volatility",
    "Gas_Close": "Natural Gas Closing Price",
    "Gas_Volatility_20d": "Natural Gas 20-Day Volatility",
    "GBP/USD_Volatility_20d": "GBP/USD 20-Day Volatility",
    "US HY Bonds_Volatility_20d": "US High Yield Bonds 20-Day Volatility",
    "SP500_EMA26": "S&P 500 26-Day Exponential Moving Average",
    "SP500_Volatility_20d": "S&P 500 20-Day Volatility"
    }
    df_reduced = df_reduced.rename(columns=feature_name_mapping)
    return df_reduced.columns, df_reduced.corr()