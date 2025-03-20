import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load processed stock data
data = pd.read_csv("processed_stock_data.csv", parse_dates=["Date"], index_col="Date")

# Prepare features (X) and target variable (y)
data["Prediction"] = data["Close"].shift(-1)
data = data.dropna()

X = data[["50_MA", "200_MA"]]
y = data["Prediction"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Stock Market Prediction App 📈")
st.write("Predict stock prices using Machine Learning!")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.write(data.tail())

# Predict on latest data
latest_features = X.iloc[-1].values.reshape(1, -1)
predicted_price = model.predict(latest_features)[0]

st.subheader("📊 Prediction for Next Day:")
st.write(f"Predicted Close Price: **{predicted_price:.2f} INR**")

# Plot actual vs predicted
st.subheader("📉 Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.values, label="Actual Prices", color="blue")
ax.plot(model.predict(X_test), label="Predicted Prices", color="red")
ax.legend()
st.pyplot(fig)

