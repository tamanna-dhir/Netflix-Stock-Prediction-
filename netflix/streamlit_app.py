from main import (
    get_stock_data,
    speak_text_streamlit,
    analyze_sentiment,
    prepare_lstm_data,
    train_lstm_model,
    predict_next_7_days,
    plot_candlestick,
    get_voice_input,
    speak_text,
    chatbot_response
)

import streamlit as st
import base64
import datetime

# --- Set Netflix-style background ---
def set_background(image_file):
    with open(image_file, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{b64_img}");
            background-size: cover;
            background-position: center;
            color: white;
        }}
        .block-container {{
            background-color: rgba(0, 0, 0, 0.85);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Cached stock data ---
@st.cache_data(ttl=3600)
def get_cached_stock_data():
    return get_stock_data("NFLX", start="2015-01-01")

# --- App Config ---
st.set_page_config(page_title="Netflix Stock Assistant", layout="wide")
set_background("assets/Netflix.jpg")

st.markdown("<h1 style='color:red;'>Netflix Voice Assistant</h1>", unsafe_allow_html=True)
st.markdown("Use the microphone or type your query below.")

# --- User Input ---
use_mic = st.button("🎤 Tap to Speak")
user_input = ""

if use_mic:
    with st.spinner("Listening..."):
        user_input = get_voice_input()
        st.markdown(f"You said: **{user_input}**")
else:
    user_input = st.text_input("Type your query here:")

# --- Main Response Logic ---
if user_input:
    with st.spinner("Processing..."):
        try:
            reply, chart, prediction_prices = chatbot_response(user_input)

            st.markdown(f"**Assistant:** {reply}")
            audio_bytes = speak_text_streamlit(reply)
            if audio_bytes:
                st.audio(audio_bytes, format='audio/mp3')
            else:
                st.warning("Could not play audio response.")

            # Show chart if returned
            if chart:
                unique_key = f"chart_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.plotly_chart(chart, use_container_width=True, key=unique_key)

            # Show predictions if available
            if prediction_prices:
                st.subheader("Predicted Netflix Stock Prices for Next 7 Days:")
                for i, price in enumerate(prediction_prices):
                    st.write(f"Day {i + 1}: ${price:.2f}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

    # Optional: Candlestick chart if requested
    if any(x in user_input.lower() for x in ["price", "chart", "candlestick"]):
        with st.spinner("Loading candlestick chart..."):
            try:
                data = get_cached_stock_data()
                fig = plot_candlestick(data)
                unique_key = f"candlestick_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                st.plotly_chart(fig, use_container_width=True, key=unique_key)
            except Exception as e:
                st.error(f"Unable to generate candlestick chart: {e}")

    # Optional: Sentiment analysis
    if any(x in user_input.lower() for x in ["sentiment", "news", "positive", "negative"]):
        sentiment = analyze_sentiment(user_input)
        st.success(f"Sentiment Analysis Result: {sentiment}")

    # Optional: LSTM prediction
    if any(x in user_input.lower() for x in ["predict", "future", "forecast", "next"]):
        st.markdown("### LSTM-based Future Price Prediction")
        try:
            data = get_cached_stock_data()
            model, scaler = train_lstm_model(data)
            predicted_prices = predict_next_7_days(model, data, scaler)

            st.subheader("Predicted Netflix Stock Prices for Next 7 Days:")
            for i, price in enumerate(predicted_prices):
                st.write(f"Day {i + 1}: ${price:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
