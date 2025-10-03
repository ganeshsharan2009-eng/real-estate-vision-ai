import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Create a sample dataset
data = {
    "Address": [
        "123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm St", "654 Maple Ave",
        "987 Cedar Rd", "159 Birch St", "753 Walnut Ave", "852 Cherry Rd", "951 Aspen Ln",
        "147 Spruce St", "258 Hickory Ave", "369 Magnolia Rd", "741 Poplar St", "852 Fir Ave",
        "963 Pinecone Rd", "357 Cypress St", "468 Larch Ave", "579 Sequoia Rd", "680 Willow Ln"
    ],
    "Price": [
        500000, 650000, 400000, 550000, 700000,
        420000, 530000, 610000, 480000, 590000,
        520000, 640000, 450000, 580000, 600000,
        430000, 560000, 620000, 470000, 495000
    ],
    "Bedrooms": [3, 4, 2, 3, 4, 2, 3, 3, 2, 3, 3, 4, 2, 3, 3, 2, 3, 4, 2, 3],
    "Bathrooms": [2, 3, 1, 2, 3, 1, 2, 2, 1, 2, 2, 3, 1, 2, 2, 1, 2, 3, 1, 2],
    "LotSize": [0.25, 0.30, 0.20, 0.28, 0.35, 0.22, 0.27, 0.29, 0.23, 0.26,
                0.24, 0.32, 0.21, 0.27, 0.28, 0.22, 0.26, 0.33, 0.20, 0.25],
    "SchoolRating": [8, 9, 7, 8, 10, 7, 8, 9, 7, 8, 8, 9, 7, 8, 9, 7, 8, 10, 7, 8],
    "CrimeRate": [
        "Low", "Medium", "High", "Low", "Medium",
        "High", "Low", "Medium", "High", "Low",
        "Low", "Medium", "High", "Low", "Medium",
        "High", "Low", "Medium", "High", "Low"
    ]
}

df = pd.DataFrame(data)

# Step 2: Encode CrimeRate
df['CrimeRateEncoded'] = df['CrimeRate'].map({'Low':0, 'Medium':1, 'High':2})

# Step 3: Train a simple linear regression model
features = ['Bedrooms', 'Bathrooms', 'LotSize', 'SchoolRating', 'CrimeRateEncoded']
X = df[features]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

# Step 4: Make predictions
df['PredictedPrice'] = model.predict(X)

# Step 5: Save to CSV
df.to_csv('predictions.csv', index=False)

print("Sample dataset created and predictions saved to predictions.csv")
print(df.head())
