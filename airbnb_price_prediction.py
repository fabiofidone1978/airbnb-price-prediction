# airbnb_price_prediction.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# Progetto: Airbnb Rental Price Prediction (Mock, senza SHAP)
# -----------------------------

# Step 1: Creazione dataset simulato
np.random.seed(42)
n = 500

data = pd.DataFrame({
    'price': np.random.gamma(2, 40, n) + 30,
    'accommodates': np.random.randint(1, 6, n),
    'bedrooms': np.random.randint(1, 4, n),
    'bathrooms': np.random.choice([1, 1.5, 2], n),
    'room_type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], n),
    'neighbourhood': np.random.choice(['Kreuzberg', 'Mitte', 'Neukolln', 'Prenzlauer Berg'], n)
})

# Step 2: Preprocessing
categorical_cols = ['room_type', 'neighbourhood']
numerical_cols = ['accommodates', 'bedrooms', 'bathrooms']

ohe = OneHotEncoder(sparse_output=False)
X_cat = ohe.fit_transform(data[categorical_cols])
X_num = StandardScaler().fit_transform(data[numerical_cols])

X = np.hstack([X_num, X_cat])
y = data['price']

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Step 4: Model training
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f} €")

# Step 6: Visualizzazione semplice del confronto predizioni vs. reali
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valori Reali (€)")
plt.ylabel("Valori Predetti (€)")
plt.title("Confronto Predizioni vs. Reali - XGBoost")
plt.grid(True)
plt.tight_layout()
plt.savefig("airbnb_price_prediction_plot.png")

# Esporta i dati simulati
data.to_csv("airbnb_mock_data.csv", index=False)
