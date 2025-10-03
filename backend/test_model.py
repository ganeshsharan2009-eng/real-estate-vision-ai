import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
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
print(df)

# Train a simple regression model
X = df[["Bedrooms", "Bathrooms", "LotSize"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predict price for a new property
new_property = [[3, 2, 0.25]]
predicted_price = model.predict(new_property)
print("Predicted price:", predicted_price)
