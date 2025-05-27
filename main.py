import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import ta
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def fetch_data(ticker='BTC-USD', period='500d', interval='1h'):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False)
    df.reset_index(inplace=True)
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Datetime': 'timestamp'
    }, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def add_indicators(df):
    close = df['close'].squeeze()

    # Розрахунок індикаторів
    df['rsi'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd = ta.trend.MACD(close=close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close=close)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['ema_10'] = ta.trend.EMAIndicator(close=close, window=10).ema_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df.dropna()


def create_sequences(data, seq_length=24):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length][3]  # Індекс 'close'
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def main():
    print("📊 Fetching data...")
    try:
        df = fetch_data()
        print("✅ Data columns:", df.columns.tolist())
        df = add_indicators(df)
    except Exception as e:
        print(f"🔴 Error: {e}")
        return

    # Підготовка даних
    features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'macd_signal',
                'bb_high', 'bb_low', 'ema_10', 'ema_20', 'hour', 'dayofweek']
    data = df[features].values

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Створення послідовностей
    X, y = create_sequences(data_scaled)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Навчання моделі
    model = LSTMModel(input_size=X.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🔁 Training model...")
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/1000, Loss: {loss.item():.6f}")


    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor)
        mse = criterion(y_pred, y_tensor)
        rmse = torch.sqrt(mse)
        print(f"\n📉 Model RMSE: {rmse.item():.4f}")

    # Прогноз
    last_seq = torch.tensor(data_scaled[-24:], dtype=torch.float32).unsqueeze(0)
    pred_scaled = model(last_seq).item()
    predicted_price = pred_scaled * scaler.scale_[3] + scaler.mean_[3]
    print(f"\n🔮 Predicted Price: ${predicted_price:.2f}")

    # Візуалізація
    plt.figure(figsize=(15, 6))

    # Останні 11 годин
    last_11 = df.tail(11)
    plt.plot(last_11['timestamp'], last_11['close'], 'b-o', label='Actual Price')

    # Прогноз
    prediction_time = last_11['timestamp'].iloc[-1] + pd.Timedelta(hours=1)
    plt.axvline(x=prediction_time, color='r', linestyle='--', label='Prediction')
    plt.plot(prediction_time, predicted_price, 'ro', markersize=10)

    plt.title('BTC Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
