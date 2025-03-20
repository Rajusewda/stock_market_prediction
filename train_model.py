import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load processed stock data
data = pd.read_csv("processed_stock_data.csv", parse_dates=["Date"], index_col="Date")

# Prepare features (X) and target variable (y)
data["Prediction"] = data["Close"].shift(-1)  # Shift Close price to predict the next day's price
data = data.dropna()  # Remove NaN values

X = data[["50_MA", "200_MA"]]  # Features (Moving Averages)
y = data["Prediction"]  # Target (Next day's Close price)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Plot actual vs predicted prices
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Prices", color="blue")
plt.plot(predictions, label="Predicted Prices", color="red")
plt.legend()
plt.title("Stock Price Prediction using Random Forest")
plt.show()
