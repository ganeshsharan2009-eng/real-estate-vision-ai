import pandas as pd
from sklearn.linear_model import LinearRegression

# ----------------------------
# 1. Sample dataset
# ----------------------------
data = {
    "Address": ["123 Main St", "456 Oak Ave", "789 Pine Rd"],
    "Price": [500000, 650000, 400000],
    "Bedrooms": [3, 4, 2],
    "Bathrooms": [2, 3, 1],
    "LotSize": [0.25, 0.30, 0.20],
    "SchoolRating": [8, 9, 7],
    "CrimeRate": ["Low", "Medium", "High"]
}

df = pd.DataFrame(data)

# ----------------------------
# 2. Prepare features for model
# ----------------------------
# Encode CrimeRate as numeric
crime_map = {"Low": 0, "Medium": 1, "High": 2}
df["CrimeRateEncoded"] = df["CrimeRate"].map(crime_map)

# Use selected features
X = df[["Bedrooms", "Bathrooms", "LotSize", "SchoolRating", "CrimeRateEncoded"]]
y = df["Price"]

# ----------------------------
# 3. Train regression model
# ----------------------------
model = LinearRegression()
model.fit(X, y)

# ----------------------------
# 4. Predict new properties
# ----------------------------
new_properties = [
    [3, 2, 0.25, 8, 0],  # Example: similar to first property
    [4, 3, 0.3, 9, 1],   # Example: similar to second property
    [2, 1, 0.2, 7, 2]    # Example: similar to third property
]

predicted_prices = model.predict(new_properties)

# ----------------------------
# 5. Save predictions to CSV
# ----------------------------
df_pred = pd.DataFrame(new_properties, columns=["Bedrooms", "Bathrooms", "LotSize", "SchoolRating", "CrimeRateEncoded"])
df_pred["PredictedPrice"] = predicted_prices
df_pred.to_csv("backend/predictions.csv", index=False)

# ----------------------------
# 6. Print results
# ----------------------------
print("Predictions saved to backend/predictions.csv")
print(df_pred)

