# Nifty 50 LSTM Stock Price Prediction

A deep learning project that uses an LSTM (Long Short-Term Memory) neural network to predict Nifty 50 index prices, complete with a Streamlit web dashboard for interactive visualization.

---

## Project Structure

```
Stock Prediction/
└── Data  collection/
    ├── data_fetch.ipynb        # Data download & feature engineering
    ├── model_train.ipynb       # LSTM model training & evaluation
    ├── predict.ipynb           # Standalone prediction & forecasting
    ├── app.py                  # Streamlit dashboard
    ├── data/
    │   ├── nifty50_raw.csv     # Raw OHLCV data (2018–2024)
    │   ├── processed_data.pkl  # Scaled sequences + scalers (dict format)
    │   └── forecast_30days.csv # 30-day forecast output
    └── models/
        └── lstm_model.h5       # Best saved LSTM model checkpoint
```

---

## Features

- **11 input features**: Close, Open, High, Low, Volume, MA20, MA50, RSI, Bollinger Bands (upper/lower), Returns
- **2-layer LSTM** architecture (48 → 24 units) with Dropout and L2 regularization
- **Huber loss** with Adam optimizer and ReduceLROnPlateau scheduling
- **7-day and 30-day price forecasting**
- **Streamlit dashboard** with interactive Plotly charts for actual vs predicted prices and forecasts

---

## Model Architecture

| Layer | Units | Details |
|-------|-------|---------|
| LSTM  | 48    | `return_sequences=True` |
| Dropout | 0.3 | |
| LSTM  | 24    | `return_sequences=False` |
| Dropout | 0.3 | |
| Dense | 16    | ReLU + L2 regularization |
| Dense | 1     | Output |

- **Optimizer**: Adam (lr=5e-4, clipnorm=1.0)
- **Loss**: Huber
- **Window size**: 60 days
- **Train/Test split**: 80/20 (chronological)

---

## Performance

| Metric | Value |
|--------|-------|
| RMSE   | ₹1087 |
| MAE    | ₹946  |
| MAPE   | 4.89% |

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib yfinance streamlit plotly
```

### 2. Fetch data & build features
Run all cells in `data_fetch.ipynb` — saves `data/processed_data.pkl`.

### 3. Train the model
Run all cells in `model_train.ipynb` — saves `models/lstm_model.h5`.

### 4. Run the Streamlit dashboard
```bash
cd "Data  collection"
python -m streamlit run app.py
```

---

## Tech Stack

- Python 3.12
- TensorFlow / Keras
- scikit-learn
- pandas, NumPy
- Streamlit
- Plotly
- yfinance
