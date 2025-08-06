# Netflix Stock Prediction using LSTM and IBM Watson

A machine learning project that predicts Netflix stock prices using Long Short-Term Memory (LSTM) neural networks integrated with IBM Watson services.

## 🚀 Features

- **LSTM Neural Network**: Advanced time series prediction model for stock price forecasting
- **IBM Watson Integration**: Leverages IBM Watson's AI capabilities for enhanced predictions
- **Interactive Web Interface**: Streamlit-based web application for easy interaction
- **Audio Feedback**: Voice response system using audio files
- **Real-time Visualization**: Candlestick charts and interactive plots
- **User Input Processing**: Custom audio input handling for voice commands

## 📁 Project Structure

```
netflix/
├── __pycache__/           # Python cache files
├── assets/                # Static assets and resources
├── candlestick_chart.html # Interactive candlestick chart visualization
├── input.wav             # Sample input audio file
├── main.py               # Core prediction logic and model
├── requirements.txt      # Python dependencies
├── response.mp3          # Audio response file
├── streamlit_app.py      # Streamlit web application
├── user_input.wav        # User audio input processing
└── README.md             # Project documentation
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tamanna-dhir/Netflix-Stock-Prediction.git
   cd Netflix-Stock-Prediction/netflix
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up IBM Watson credentials**
   - Create an IBM Watson account
   - Configure your API keys and service credentials
   - Update the configuration in your main application files

## 🎯 Usage

### Running the Streamlit Web App

```bash
streamlit run streamlit_app.py
```

This will launch the interactive web interface where you can:
- Input stock prediction parameters
- View real-time predictions
- Interact with candlestick charts
- Use voice input features

### Running the Core Prediction Model

```bash
python main.py
```

This executes the main LSTM prediction model with IBM Watson integration.

## 📊 Model Details

### LSTM Architecture
- **Model Type**: Long Short-Term Memory (LSTM) Neural Network
- **Purpose**: Time series forecasting for Netflix stock prices
- **Training Data**: Historical Netflix stock data
- **Features**: Open, High, Low, Close prices, Volume

### IBM Watson Integration
- **Services Used**: IBM Watson AI services for enhanced prediction accuracy
- **Benefits**: Improved model performance through cloud-based AI capabilities
- **Real-time Processing**: Live data analysis and prediction updates

## 🎨 Visualization Features

- **Candlestick Charts**: Interactive HTML-based stock price visualization
- **Real-time Updates**: Dynamic chart updates with new predictions
- **User-friendly Interface**: Streamlit-powered web interface
- **Audio Feedback**: Voice responses for user interactions

## 📈 Key Components

### main.py
Core application containing:
- LSTM model implementation
- Data preprocessing pipelines
- IBM Watson service integration
- Prediction algorithms

### streamlit_app.py
Web interface featuring:
- Interactive dashboard
- User input forms
- Real-time chart displays
- Audio input/output handling

### Audio Processing
- Voice command recognition
- Audio response generation
- Real-time audio feedback system

## 🔧 Dependencies

Key Python packages (see `requirements.txt` for complete list):
- `tensorflow` / `keras` - LSTM model implementation
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `plotly` - Interactive visualizations
- `ibm-watson` - IBM Watson SDK
- `yfinance` - Stock data fetching
- `scikit-learn` - Machine learning utilities

## 📋 Requirements

- Python 3.7+
- IBM Watson account and API credentials
- Internet connection for real-time data
- Audio input/output capabilities (for voice features)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request



## 🙏 Acknowledgments

- IBM Watson for AI services
- Netflix for providing historical stock data
- TensorFlow/Keras community for LSTM implementation resources
- Streamlit for the web application framework



## 🚀 Future Enhancements

- [ ] Multi-stock prediction support
- [ ] Advanced technical indicators integration
- [ ] Mobile app development
- [ ] Real-time news sentiment analysis
- [ ] Portfolio optimization features
- [ ] Enhanced voice command capabilities

---

**Disclaimer**: This project is for educational and research purposes only. Stock predictions should not be used as the sole basis for investment decisions. Always consult with financial advisors before making investment choices.
### Project Preview

![Netflix Stock Prediction](assets/netflix.jpg)
