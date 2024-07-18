import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from streamlit_lightweight_charts import renderLightweightCharts
from datetime import datetime
import random

# Function to set random seed
def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# Function to fetch real-time data
def fetch_crypto_data(symbol, timeframe, limit):
    data = yf.download(symbol, period="10y", interval=timeframe)  # Fetch 10 years of data
    return data

# LSTM model functions
def prepare_data_lstm(data, look_back=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    return np.array(X), np.array(y), scaler


def create_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Function to make predictions using the selected model
def make_predictions(model_type, data, num_candles):
    if model_type == 'LSTM':
        set_random_seed()
        look_back = 90
        X, y, scaler = prepare_data_lstm(data['Close'].values, look_back)  # Use 'Close' column for prediction
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        model = create_lstm_model(look_back)
        model.fit(X, y, epochs=20, batch_size=64, verbose=0)
        last_60_days = data['Close'].values[-90:]  # Use 'Close' column for prediction
        last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        pred_prices = []
        for _ in range(num_candles):
            pred_price = model.predict(X_test)
            pred_prices.append(float(scaler.inverse_transform(pred_price)[0][0]))
            X_test = np.append(X_test[:, 1:, :], pred_price.reshape(1, 1, 1), axis=1)
        return pred_prices
    elif model_type == 'ARIMA':
        model = ARIMA(data['Close'], order=(1, 1, 1))  # Use 'Close' column for prediction
        results = model.fit()
        forecast = results.forecast(steps=num_candles)
        return forecast.tolist()
    elif model_type == 'SARIMA':
        model = SARIMAX(data['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Use 'Close' column for prediction
        results = model.fit()
        forecast = results.forecast(steps=num_candles)
        return forecast.tolist()
    elif model_type == 'Linear Regression':
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['Close'].values  # Use 'Close' column for prediction
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(data), len(data) + num_candles).reshape(-1, 1)
        predictions = model.predict(future_X)
        return predictions.tolist()


# Streamlit app
st.set_page_config(layout="wide")
st.title('Crypto Price Prediction App')

# Sidebar for user inputs
with st.sidebar:
    st.header("Input Parameters")
    crypto_name = st.text_input('Enter Crypto Name (e.g., BTC-USD):', 'BTC-USD')
    timeframe = st.selectbox('Select Timeframe:', ['1d', '1m', '5m', '15m', '30m'])
    num_candles = st.number_input('Number of candles to predict:', min_value=1, max_value=100, value=10)
    model_type = st.selectbox('Select Model:', ['LSTM', 'ARIMA', 'SARIMA', 'Linear Regression'])
    predict_button = st.button('Predict')

# Main content area
if predict_button:
    # Show a spinner while processing
    with st.spinner('Fetching data and making predictions...'):
        # Fetch data
        data = fetch_crypto_data(crypto_name, timeframe, 1000)
        data = data.reset_index()

        # Make predictions and time the process
        start_time = time.time()
        pred_prices = make_predictions(model_type, data, num_candles)
        end_time = time.time()
        prediction_time = end_time - start_time

        # Prepare data for plotting
        historical_dates = data['Date'].values
        future_dates = pd.date_range(start=historical_dates[-1], periods=num_candles + 1, freq=timeframe).values[1:]

        # Prepare data for Lightweight Charts
        candlestick_data = data.apply(lambda x: {
            'time': x['Date'].timestamp(),
            'open': float(x['Open']),
            'high': float(x['High']),
            'low': float(x['Low']),
            'close': float(x['Close'])
        }, axis=1).tolist()

        future_dates_py = [pd.Timestamp(date).to_pydatetime() for date in future_dates]
        line_data = [{'time': date.timestamp(), 'value': price} for date, price in zip(future_dates_py, pred_prices)]

        chartOptions = {
            "height": 500,
            "rightPriceScale": {
                "scaleMargins": {
                    "top": 0.2,
                    "bottom": 0.2,
                },
                "borderVisible": False,
            },
            "timeScale": {
                "borderVisible": False,
            },
            "crosshair": {
                "horzLine": {
                    "visible": False,
                    "labelVisible": False
                },
                "vertLine": {
                    "visible": True,
                    "style": 0,
                    "width": 2,
                    "color": "rgba(224, 227, 235, 0.1)",
                    "labelVisible": False,
                }
            },
            "grid": {
                "vertLines": {
                    "visible": False,
                },
                "horzLines": {
                    "visible": False,
                }
            },
        }

        seriesCandlestickChart = [{
            "type": 'Candlestick',
            "data": candlestick_data,
            "options": {
                "upColor": "#26a69a",
                "downColor": "#ef5350",
                "borderVisible": False,
                "wickUpColor": "#26a69a",
                "wickDownColor": "#ef5350"
            }
        }, {
            "type": 'Line',
            "data": line_data,
            "options": {
                "color": "#2962FF",
                "lineWidth": 2,
                "crosshairMarkerVisible": True,
                "crosshairMarkerRadius": 6,
                "lineType": 1,
            }
        }]

    # Render the chart
    st.subheader(f"{crypto_name} Price Prediction using {model_type}")
    renderLightweightCharts([{
        "chart": chartOptions,
        "series": seriesCandlestickChart
    }], 'crypto_chart')

    # Display prediction time
    st.write(f"Prediction Time: {prediction_time:.2f} seconds")

    # Display predicted prices in a full-width table
    st.subheader("Predicted Prices")
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': pred_prices
    })
    st.dataframe(pred_df, use_container_width=True)
