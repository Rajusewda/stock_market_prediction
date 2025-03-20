import pandas as pd

# Load data
data = pd.read_csv("reliance_stock_data.csv", parse_dates=["Date"], index_col="Date")

# Feature Engineering
data["50_MA"] = data["Close"].rolling(window=50).mean()
data["200_MA"] = data["Close"].rolling(window=200).mean()

# Drop rows with NaN values
data = data.dropna()

# Save the cleaned data
data.to_csv("processed_stock_data.csv")

print("Preprocessing complete! Data saved.")
