import pandas as pd

# Load the sample dataset
file_path = "../data/sample_properties.csv"
df = pd.read_csv(file_path)

# Print the dataset
print("Sample Dataset:")
print(df)
