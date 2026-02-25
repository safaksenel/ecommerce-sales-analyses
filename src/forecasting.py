import pandas as pd
from prophet import Prophet


### Preparing the data for forecasting with Prophet###

# Load the cleaned dataset
df = pd.read_csv("../data/cleaned_data/cleaned_amazon_sales_dataset.csv")

# Convert 'order_date' to datetime format
df['order_date'] = pd.to_datetime(df['order_date'])

# Prepare the data for Prophet
prophet_df = df.groupby('order_date')['total_revenue'].sum().reset_index()
prophet_df.columns = ['ds', 'y']    # Prophet requires the columns to be named 'ds' for the date and 'y' for the value

print(prophet_df.head())



### Fitting the Model ###

# Initialize the Prophet model (with yearly and weekly seasonality)
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)

# Fit the model to the data (this is where the actual forecasting happens)
model.fit(prophet_df)

# Create a DataFrame to hold predictions for the next 30 days
future = model.make_future_dataframe(periods=30)

# Generate the forecast
forecast = model.predict(future)

print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())



### Visualize the forecast ###

import matplotlib.pyplot as plt

# Plot the forecast
fig1 = model.plot(forecast)
plt.title("Amazon Toplam Gelir Tahmini")
plt.xlabel("Tarih")
plt.ylabel("Toplam Gelir (Revenue)")
plt.show()

# Plot the forecast components (trend, yearly seasonality, weekly seasonality)
fig2 = model.plot_components(forecast)
plt.show()