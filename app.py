import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("Comodity Price Predictor.keras")

# Load dataset
data = pd.read_csv("Vegetables.csv")
data = data.rename(columns={'Price (Rs./Quintal)': 'Price'})
data.dropna(inplace=True)

# Title
st.title("🧠 Commodity Price Predictor")
st.markdown("Enter the **location** and **commodity name**, then click **Predict** to get the price forecast for the next 3 days.")

# Inputs
location = st.text_input("Enter Location").strip().title()
commodity = st.text_input("Enter Commodity Name").strip().title()

# Predict button
if st.button("Predict"):
    # Filter data based on user inputs
    filtered_data = data[(data['State'] == location) & (data['Commodity'] == commodity)]

    if filtered_data.empty:
        st.error("No data available for the given location and commodity. Please try different inputs.")
    else:
        filtered_data = filtered_data.sort_values("Date")
        price_data = pd.DataFrame(filtered_data['Price'])

        # Train-test split
        train_data = price_data[0:int(len(price_data) * 0.80)]
        test_data = price_data[int(len(price_data) * 0.80):]

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data)

        # Prepare input for model training
        x_train, y_train = [], []
        for i in range(10, len(train_scaled)):
            x_train.append(train_scaled[i-10:i])
            y_train.append(train_scaled[i, 0])
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Prepare test data
        past_10_days = train_data.tail(10)
        test_data_full = pd.concat([past_10_days, test_data], ignore_index=True)
        test_scaled = scaler.transform(test_data_full)

        x_test = []
        for i in range(10, len(test_scaled)):
            x_test.append(test_scaled[i-10:i])
        x_test = np.array(x_test)

        # Make predictions
        last_10_days = test_scaled[-10:].reshape(-1, 1)
        predicted_prices = []

        for _ in range(3):
            next_10_days_scaled = last_10_days.reshape(1, 10, 1)
            predicted_scaled = model.predict(next_10_days_scaled, verbose=0)
            predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]
            predicted_prices.append(predicted_price)
            last_10_days = np.append(last_10_days[1:], predicted_scaled).reshape(-1, 1)

        # Display results
        st.subheader(f"📊 Predicted Prices for {commodity} in {location} (Next 3 Days)")
        for i, price in enumerate(predicted_prices, 1):
            st.write(f"Day {i}: ₹{price:.2f} per quintal | ₹{price/100:.2f} per kg")
