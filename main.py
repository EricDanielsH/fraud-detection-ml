import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV

# Load the data to a DataFrame
df = pd.read_csv("creditcard.csv")

# Print basic information about the data to understand it
# print(df.info())
# print("====================================================")
# print(df.describe())
# print("====================================================")
# print(df.head())

# Check for any missing values in any field
print(df.isnull().sum())

# Drop rows missing values if there are any
df = df.dropna()

# Separate Features from the Target
features = df.drop(columns=['Class'])
target = df['Class']

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

# Train the models and optimise
for pipeline in pipelines:
    pipeline.fit(features_train, target_train)
    
    # Evaluate the best models with the test set
    # Predict the class labels for the testing training set
    target_pred = pipeline.predict(features_test)
    # Predict the probability estimates for the positive class(1). ROC AUC
    target_prob = pipeline.predict_proba(features_test)[:, 1]

    print(f"Classification report for {pipeline.steps[-1][1].__class__.__name__}:")
    print(classification_report(target_test, target_pred))
    
    print(f"ROC-AUC score for {pipeline.steps[-1][1].__class__.__name__}: {roc_auc_score(target_test, target_prob)}\n")

    
