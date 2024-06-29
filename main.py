import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load the data to a DataFrame
df = pd.read_csv("creditcard.csv")

# Print basic information about the data to understand it
print(df.info())
print("====================================================")
print(df.describe())
print("====================================================")
print(df.head())

# Check for any missing values in any field
print(df.isnull().sum())

# Drop rows missing values if there are any
print(df.dropna())

# Separate Features from the Target
features = df.drop(columns=['Class'])
target = df['Class']

# Normalise numericals fields
numerical_fields = features.select_dtypes(include=['int64', 'float64']).columns
# Create a scaler object
scaler = StandardScaler()

# Scale numerical fields
features = scaler.fit_transform(features)

# See the difference with the original dataset
# print(df)
# print(df[numerical_fields])

