# Airbnb Price Prediction con XGBoost

Stima dei prezzi di affitto di annunci Airbnb simulati, basata su regressione con XGBoost.

## Obiettivo

- Costruire una pipeline predittiva su dati immobiliari simulati
- Valutare le performance tramite MAE
- Produrre una visualizzazione interpretabile delle previsioni

## Stack Tecnico

- Python 3
- XGBoost
- Scikit-learn
- Pandas, Matplotlib

## Struttura

📦 airbnb-price-prediction
├── airbnb_price_prediction.py
├── airbnb_mock_data.csv
├── airbnb_price_prediction_plot.png
└── README.md

## Setup

```bash
pip install pandas scikit-learn matplotlib xgboost
python airbnb_price_prediction.py

CSV completo degli annunci simulati
Grafico confronto predizioni vs. valori reali
