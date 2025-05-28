Predicting stock prices is one of the most fascinating and challenging problems in data science. With countless variables affecting the markets, combining machine learning with real-time data analysis and natural language interfaces can offer powerful insights. In this project, we explore how to predict the stock price of Netflix using an LSTM (Long Short-Term Memory) model, enhanced with the cognitive capabilities of IBM Watson.
While researching tools and frameworks to build this system, we noticed a lack of practical, integrated examples that combine time series forecasting with interactive voice-enabled assistants. So, we built one — and here it is. If you're interested in machine learning, voice AI, or financial forecasting, we hope this project gives you a head start.
This project brings together multiple advanced technologies to create a smart stock prediction assistant. Below is a breakdown of the core components and their roles:
LSTM (Long Short-Term Memory) Neural Networks
Purpose: Predict future Netflix stock prices based on historical data.
Why LSTM? LSTM models are a type of recurrent neural network (RNN) designed to handle time series data effectively. They are capable of learning long-term dependencies and patterns in stock price trends.
Implementation: The model is trained on past stock prices (Open, Close, High, Low) to forecast future values.
IBM Watson Assistant
Purpose: Provides a conversational interface (chatbot) to interact with users.
Capabilities: Users can ask questions like “What is the predicted price for tomorrow?” or “Show me the stock trend for last week,” and the assistant responds accordingly.
Benefit: Adds natural language processing and a user-friendly interface to the application.
 IBM Watson Speech-to-Text and Text-to-Speech
Purpose: Enables voice input and audio responses.
Usage: Users can speak their queries instead of typing, and the assistant replies with synthesized speech.
Advantage: Enhances accessibility and provides a hands-free experience.
 Alpha Vantage API
Purpose: Fetches real-time and historical stock market data.
Why Alpha Vantage? It offers free access to stock data, is easy to integrate, and supports multiple time intervals (daily, weekly, etc.).
Use Case: The historical data is preprocessed and fed into the LSTM model for training and prediction.
Streamlit
Purpose: Builds a fast, interactive web application for the assistant.
Features: Includes text and voice chat options, dynamic graphs (candlestick charts), and prediction visualizations.
Benefit: Makes deployment and user interaction seamless and visually appealing.
 Sentiment Analysis
Purpose: Analyze public sentiment about Netflix from social media or news data.
Value: Helps refine predictions by factoring in public mood and trends.
Key Libraries Used :
This project leverages a combination of data science, machine learning, NLP, and web development libraries. Below is a breakdown of the major Python libraries used:
 Data Handling & Visualization
pandas – For data manipulation, analysis, and cleaning of stock price data.
numpy – Used for numerical computations and array operations.
matplotlib & seaborn – This is used to plot basic line charts and exploratory data visualizations.
plotly – To create interactive candlestick charts and stock price visualizations.
 Machine Learning & Deep Learning
scikit-learn – Used for data preprocessing like MinMax scaling and train-test splitting.
tensorflow / keras – Implements the LSTM model for time series prediction.
APIs and Data Sources
alpha_vantage – For retrieving real-time and historical stock price data from the Alpha Vantage API.
IBM Watson Services
ibm_watson – Interfaces with IBM Watson services like Assistant, Speech-to-Text, and Text-to-Speech.
ibm_cloud_sdk_core – Required for authentication with IBM Cloud services.
Voice Input & Output
speech_recognition – Captures and transcribes user voice input into text.
pyttsx3 – Provides text-to-speech conversion for offline voice output.
pyaudio – Required by speech_recognition for accessing microphone input.
Web App Development
streamlit – The primary framework for building interactive web apps with UI support for voice/text input and visualization.
textblob / vaderSentiment – For sentiment analysis of text data related to Netflix.
joblib – To save and load machine learning models efficiently.







