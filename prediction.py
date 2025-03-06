import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.decomposition import PCA

# Load data
def load_data():
    #return create_df()
    # To avoid to many API calls, we will load the data from a CSV file
    df = pd.read_csv("data.csv", index_col=0)
    df.index = pd.to_datetime(df.index)
    return df

# Features selection
def features_selection_pca(df):
    X = df.drop(columns=["SP500_Volatility_20d"])

    # Standardize features (important for PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=0.80)  # Retains 80% of variance
    X_pca = pca.fit_transform(X_scaled)

    # Create a DataFrame with PCA features
    pca_columns = [f"PC{i+1}" for i in range(pca.n_components_)]
    X_pca_df = pd.DataFrame(X_pca, columns=pca_columns)

    X_pca_df["SP500_Volatility_20d"] = df["SP500_Volatility_20d"].values
    X_pca_df["SP500_Returns"] = df["SP500_Returns"].values
    X_pca_df.index = pd.to_datetime(df.index)
    
    return X_pca_df

# Prediction
def create_sequences(data, seq_length, horizon):
    sequences = []
    for i in range(len(data) - seq_length - horizon + 1):
        seq = data.iloc[i:i + seq_length]
        label = data['SP500_Volatility_20d'].iloc[i + seq_length:i + seq_length + horizon].values
        sequences.append((seq, label))
    return sequences

def lstm_data_processing(df):
    # Normalize the data
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    # Separate features and target for scaling
    features = df.drop(columns=['SP500_Volatility_20d'])
    target = df[['SP500_Volatility_20d']]

    scaled_features = scaler_features.fit_transform(features)
    scaled_target = scaler_target.fit_transform(target)

    # Combine scaled features and target into a DataFrame
    scaled_data = pd.DataFrame(scaled_features, columns=features.columns)
    scaled_data['SP500_Volatility_20d'] = scaled_target

    return scaled_data, scaler_target

def lstm_prediction(df, scaled_data, scaler_target):
    # Define sequence length and prediction horizon
    seq_length = 10
    horizon = 20  # Number of future time steps to predict

    # Create sequences
    sequences = create_sequences(scaled_data, seq_length, horizon)

    # Split into features and labels
    X, y = zip(*sequences)
    X, y = np.array(X), np.array(y)

    # Split into training and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, X_train.shape[2])))
    model.add(Dense(horizon))  # Output layer with 'horizon' number of neurons
    model.compile(optimizer='adam', loss='mse')

    # Fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # Make predictions
    y_pred = model.predict(X_test)

    # Inverse the scaling for the target variable
    y_pred_inv = scaler_target.inverse_transform(y_pred)

    return y_pred_inv

def dashboard_prediction():
    df_original = load_data()
    df_pca = features_selection_pca(df_original)
    scaled_data, scaler_target = lstm_data_processing(df_pca)
    y_pred = lstm_prediction(df_pca, scaled_data, scaler_target)
    return df_original, y_pred
