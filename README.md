---
title: Nifty 50 LSTM Predictor
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
startup_duration_timeout: 1h
---

# Nifty 50 Multi-Model Stock Price Prediction

A deep learning project for Nifty 50 forecasting using multiple neural architectures (LSTM, GRU, Transformer, and CNN+LSTM), with a Streamlit dashboard for model comparison, forecasting, and interactive technical analysis.

---

## Project Structure

```
Stock Prediction/
└── Data  collection/
    ├── data_fetch.ipynb        # Data download & feature engineering
    ├── model_train.ipynb       # Multi-model training & evaluation
    ├── predict.ipynb           # Multi-model inference + best-model forecasting
    ├── app.py                  # Streamlit dashboard
    ├── data/
    │   ├── nifty50_raw.csv     # Raw OHLCV data (2018–2024)
    │   ├── processed_data.pkl  # Scaled sequences + scalers (dict format)
    │   └── forecast_30days.csv # 30-day forecast output
    └── models/
        ├── lstm_model.h5             # LSTM checkpoint
        ├── gru_model.h5              # GRU checkpoint
        ├── transformer_model.keras   # Transformer checkpoint
        └── cnn_lstm_model.keras      # CNN+LSTM checkpoint
```

---

## Features

- **11 input features**: Close, Open, High, Low, Volume, MA20, MA50, RSI, Bollinger Bands (upper/lower), Returns
- **4 trained architectures**: LSTM, GRU, Transformer, CNN+LSTM
- **Chronological validation** with `shuffle=False` for time-series-safe training
- **Automatic model leaderboard** (RMSE, MAE, MAPE) in prediction notebook and dashboard
- **Best-model-driven forecasting** for 7 to 60 business days
- **Streamlit dashboard** with model selection, comparison overlays, forecast download, and technical indicators

---

## Model Architecture Summary

- **LSTM**: 2 stacked LSTM layers (48 -> 24), Dropout, Dense head
- **GRU**: 2 stacked GRU layers (48 -> 24), Dropout, Dense head
- **Transformer**: MultiHeadAttention + feed-forward block + dense regression head
- **CNN+LSTM**: stacked Conv1D blocks + deep LSTM stack + dense head

Common setup:
- **Loss**: Huber
- **Optimizers**: Adam with gradient clipping (`clipnorm=1.0`)
- **Window size**: 60 days
- **Train/Test split**: chronological (time-series-safe)

---

## Performance

Performance is model-dependent and can change with retraining. Use:

- `predict.ipynb` for a sorted model leaderboard (`RMSE`, `MAE`, `MAPE`)
- Streamlit dashboard tab **Model Performance** for active model diagnostics and full-model leaderboard

---

## Setup & Usage

### 1. Install dependencies
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib yfinance streamlit plotly
```

### 2. Fetch data & build features
Run all cells in `data_fetch.ipynb` — saves `data/processed_data.pkl`.

### 3. Train models
Run all cells in `model_train.ipynb` — saves model checkpoints in `models/`:

- `lstm_model.h5`
- `gru_model.h5`
- `transformer_model.keras`
- `cnn_lstm_model.keras`

### 4. Run prediction workflow
Run `predict.ipynb` to:

- Load all available trained models
- Evaluate each model on test data
- Auto-select the best model (lowest MAPE)
- Generate and save `data/forecast_30days.csv` with model name metadata

### 5. Run the Streamlit dashboard
```bash
python -m streamlit run app.py
```

The dashboard now:

- Loads all available models from `Data  collection/models`
- Lets you choose the active model in the sidebar
- Supports optional prediction overlay for all models
- Uses selected model for metrics and forecast download

### 6. Deploy on Hugging Face Spaces (Docker)
1. Create a new Hugging Face Space and choose `Docker` as the SDK.
2. Point the Space to this GitHub repository.
3. Hugging Face builds from `Dockerfile` and runs root `app.py`, which forwards to `Data  collection/app.py`.
4. After each push to `main`, the Space rebuilds and redeploys.

---

## Tech Stack

- Python 3.12
- TensorFlow / Keras
- scikit-learn
- pandas, NumPy
- Streamlit
- Plotly
- yfinance
