import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


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

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.2, random_state=42)

# print(f'X_train shape: {features_train.shape}')
# print(f'X_test shape: {features_test.shape}')
# print(f'y_train shape: {target_train.shape}')
# print(f'y_test shape: {target_test.shape}')

# Create different pipelines for different models
# Logistic Regression Pipeline
lr_pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', LogisticRegression())])
# Random Forest Pipeline
rf_pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', RandomForestClassifier())])
# Gradient Boosting Pipeline
gb_pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', GradientBoostingClassifier())])

pipelines = [lr_pipeline, rf_pipeline, gb_pipeline]

# Train the models
for pipeline in pipelines:
    pipeline.fit(features_train, target_train)