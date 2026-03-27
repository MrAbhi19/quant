import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def fetch_and_engineer(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError("Empty dataframe returned from yfinance.")
    
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['volatility'] = df['log_returns'].rolling(window=20).std()
    df['vol_ratio'] = df['log_returns'].rolling(window=5).std() / df['volatility']
    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()
    
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    rsi_norm = 0.1 * (rsi - 50)
    df['rsi_fisher'] = (np.exp(2 * rsi_norm) - 1) / (np.exp(2 * rsi_norm) + 1)
    
    df['lag_1'] = df['log_returns'].shift(1)
    df['lag_2'] = df['log_returns'].shift(2)
    
    df['target'] = (df['log_returns'].shift(-1).abs() > 0.01).astype(int)
    
    return df.dropna()

def create_sequences(data, target, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

def prepare_data(df, feature_cols, seq_length):
    raw_data = df[feature_cols].values
    targets = df['target'].values
    
    split_idx = int(len(raw_data) * 0.8)
    
    scaler = StandardScaler()
    scaler.fit(raw_data[:split_idx])
    scaled_data = scaler.transform(raw_data)
    
    X, y = create_sequences(scaled_data, targets, seq_length)
    
    seq_split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:seq_split_idx], X[seq_split_idx:]
    y_train, y_test = y[:seq_split_idx], y[seq_split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler

def build_lstm(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def get_prediction_and_metrics(ticker, start_date, end_date, seq_length=10, epochs=50):
    features = ['log_returns', 'volatility', 'vol_ratio', 'ma_5', 'ma_10', 'ma_20', 'rsi_fisher', 'lag_1', 'lag_2']
    
    df = fetch_and_engineer(ticker, start_date, end_date)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df, features, seq_length)
    
    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=0)
    
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred_class = (y_pred_prob > 0.5).astype(int)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_class),
        'Precision': precision_score(y_test, y_pred_class, zero_division=0),
        'Recall': recall_score(y_test, y_pred_class, zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, y_pred_prob)
    }
    
    recent_data = df[features].iloc[-seq_length:].values
    recent_scaled = scaler.transform(recent_data)
    X_new = np.array([recent_scaled])
    
    prob = model.predict(X_new, verbose=0)[0][0]
    return prob, metrics

if __name__ == "__main__":
    ticker = 'YESBANK.NS'
    start_date = '2024-01-01'
    end_date = '2026-03-27'
    
    probability, metrics = get_prediction_and_metrics(ticker, start_date, end_date)
    
    print(f"Probability of an absolute move > 1% tomorrow for {ticker}: {probability:.2%}\n")
    print("Model Test Set Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
