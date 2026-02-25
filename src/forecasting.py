import pandas as pd
from prophet import Prophet

# Load the cleaned dataset
df = pd.read_csv("../data/cleaned_data/cleaned_amazon_sales_dataset.csv")

# Convert 'order_date' to datetime format
df['order_date'] = pd.to_datetime(df['order_date'])

# Prepare the data for Prophet
prophet_df = df.groupby('order_date')['total_revenue'].sum().reset_index()
prophet_df.columns = ['ds', 'y']    # Prophet requires the columns to be named 'ds' for the date and 'y' for the value

print(prophet_df.head())
