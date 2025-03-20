import yfinance as yf
import pandas as pd

# Fetch historical data for Reliance Industries (RELIANCE.NS)
stock = yf.Ticker("RELIANCE.NS")
data = stock.history(period="5y")  # Last 5 years of data

# Save to CSV
data.to_csv("reliance_stock_data.csv")

print("Stock data saved successfully!")
