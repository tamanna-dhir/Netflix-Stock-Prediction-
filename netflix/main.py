import os
import datetime
import numpy as np
import speech_recognition as sr
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from ibm_watson import AssistantV1, SpeechToTextV1, TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import webbrowser
import pandas as pd
import time
import requests
import logging
import streamlit as st

# === Alpha Vantage API Key ===
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '15CD847ESS6XON9W')

# === IBM Watson Credentials ===
STT_API_KEY = os.getenv('STT_API_KEY', 'n5Lb70ZzdFPBMxZ4znjy5jyVcmxnk29GGTTtpFmVLmDE')
STT_URL = 'https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/631ed6b3-62e4-4be5-9913-d56a8d91040f'

TTS_API_KEY = os.getenv('TTS_API_KEY', 'T3c9yBrv35VtTPIt2KslXdCbhuAxn4co6nW3cB0PSFdv')
TTS_URL = 'https://api.us-south.text-to-speech.watson.cloud.ibm.com/instances/592a39df-4d2d-4778-8e72-a8b0447d1dd0'

ASSISTANT_API_KEY = os.getenv('ASSISTANT_API_KEY', 'E9SsLc77focYBnqboMZZ5kAlVDZ4pCN8UPbOWy9hhcAm')
ASSISTANT_URL = 'https://api.us-south.assistant.watson.cloud.ibm.com'
WORKSPACE_ID = os.getenv('WORKSPACE_ID', 'dd9cb03e-e379-46b6-9d75-640ab2332c68')

# === IBM Service Auth ===
stt = SpeechToTextV1(authenticator=IAMAuthenticator(STT_API_KEY))
stt.set_service_url(STT_URL)

tts = TextToSpeechV1(authenticator=IAMAuthenticator(TTS_API_KEY))
tts.set_service_url(TTS_URL)

assistant = AssistantV1(version='2021-06-14', authenticator=IAMAuthenticator(ASSISTANT_API_KEY))
assistant.set_service_url(ASSISTANT_URL)


def get_stock_data(symbol="NFLX", start="2015-01-01", api_key=ALPHA_VANTAGE_API_KEY):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",  # Change to 'TIME_SERIES_DAILY' from 'TIME_SERIES_DAILY_ADJUSTED'
        "symbol": symbol,
        "outputsize": "full",
        "apikey": api_key
    }
    retry_wait_time = 15
    for attempt in range(3):
        try:
            response = requests.get(url, params=params)
            data = response.json()

            print(f"Alpha Vantage API Response (Attempt {attempt+1}):", data)

            if "Time Series (Daily)" not in data:
                error_msg = data.get('Note') or data.get('Error Message') or str(data)
                raise ValueError(f"Alpha Vantage Error: {error_msg}")

            df = pd.DataFrame(data["Time Series (Daily)"]).T
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "5. adjusted close": "Adj Close",  # Optional if needed
                "6. volume": "Volume"
            })
            return df.loc[start:].astype(float)

        except Exception as e:
            logging.error(f"[Attempt {attempt + 1}] Error fetching stock data: {e}")
            time.sleep(retry_wait_time)
            retry_wait_time *= 2

    logging.error("All attempts to fetch stock data failed.")
    return pd.DataFrame()

def analyze_sentiment(news_list):
    sentiments = [TextBlob(news).sentiment.polarity for news in news_list]
    return sum(sentiments) / len(sentiments) if sentiments else 0


def prepare_lstm_data(df, sequence_length=60):
    data = df[['Close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i-sequence_length:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)
    return *train_test_split(X, y, test_size=0.2, shuffle=False), scaler


def train_lstm_model(df):
    if len(df) < 100:
        raise ValueError("Not enough data to train LSTM model.")
    X_train, _, y_train, _, scaler = prepare_lstm_data(df)
    if len(X_train) == 0:
        raise ValueError("Sequence preparation failed.")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
    return model, scaler


def predict_next_7_days(model, df, scaler, sequence_length=60):
    data = scaler.transform(df[['Close']].values)
    seq = data[-sequence_length:].reshape(1, sequence_length, 1)
    preds = []
    for _ in range(7):
        pred = model.predict(seq)[0][0]
        preds.append(scaler.inverse_transform([[pred]])[0][0])
        seq = np.append(seq[:, 1:, :], [[[pred]]], axis=1)
    return preds


def plot_candlestick(df):
    if df.empty:
        return None
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig.update_layout(
        title="Netflix Candlestick Chart",
        xaxis_title='Date', yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    return fig


def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    with open("user_input.wav", "wb") as f:
        f.write(audio.get_wav_data())
    with open("user_input.wav", "rb") as audio_file:
        result = stt.recognize(audio=audio_file, content_type='audio/wav').get_result()
    try:
        return result['results'][0]['alternatives'][0]['transcript']
    except Exception as e:
        logging.error(f"Voice error: {e}")
        return ""


def speak_text(text):
    import tempfile
    from playsound import playsound
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_data = tts.synthesize(text, voice='en-US_AllisonV3Voice', accept='audio/mp3').get_result().content
            tmp.write(audio_data)
            playsound(tmp.name)
    except Exception as e:
        logging.error(f"TTS error: {e}")

def speak_text_streamlit(text):
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            audio_data = tts.synthesize(
                text,
                voice='en-US_AllisonV3Voice',
                accept='audio/mp3'
            ).get_result().content
            tmp.write(audio_data)
            return audio_data  # ✅ Return the audio data
    except Exception as e:
        logging.error(f"TTS Streamlit error: {e}")
        return None  # ✅ Return None if it fails




def chatbot_response(user_input):
    try:
        response = assistant.message(workspace_id=WORKSPACE_ID, input={'text': user_input}).get_result()
        reply = response.get("output", {}).get("text", [""])[0]
        chart = None
        predictions = None

        if "stock price" in user_input.lower():
            df = get_stock_data()
            reply += f"\nCurrent Netflix Stock Price: ${df['Close'].iloc[-1]:.2f}" if not df.empty else "\nData unavailable."

        elif "sentiment" in user_input.lower():
            headlines = [
                "Netflix stock surges after strong earnings",
                "Concerns over Netflix's spending",
                "Positive reviews for new series",
                "Intense competition in streaming"
            ]
            score = analyze_sentiment(headlines)
            sentiment = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
            reply += f"\nSentiment Score: {score:.2f} ({sentiment})"

        elif "candlestick" in user_input.lower():
            df = get_stock_data()
            chart = plot_candlestick(df)
            reply += "\nDisplaying candlestick chart..." if chart else "\nFailed to generate chart."

        elif any(kw in user_input.lower() for kw in ["predict", "forecast", "next 7"]):
            df = get_stock_data()
            if not df.empty:
                model, scaler = train_lstm_model(df)
                predictions = predict_next_7_days(model, df, scaler)
                reply += "\nPredictions for next 7 days:\n" + "\n".join(
                    [f"Day {i+1}: ${p:.2f}" for i, p in enumerate(predictions)])
            else:
                reply += "\nInsufficient data for prediction."

        return reply, chart, predictions

    except Exception as e:
        logging.error(f"Bot error: {e}")
        return f"Error: {e}", None, None


def run_chatbot():
    print("Welcome to the Netflix Assistant!")
    speak_text("Welcome to the Netflix Assistant!")
    while True:
        mode = input("Type 't' for text or 'v' for voice ('exit' to quit): ").strip().lower()
        if mode == 'exit':
            print("Goodbye!")
            speak_text("Goodbye!")
            break
        user_input = input("You: ") if mode == 't' else get_voice_input()
        if not user_input.strip():
            print("No input detected.")
            continue
        reply, chart, predictions = chatbot_response(user_input)
        print("\nBot:", reply)
        speak_text(reply)
        if predictions:
            for i, price in enumerate(predictions):
                print(f"Day {i+1}: ${price:.2f}")


if __name__ == "__main__":
    run_chatbot()

