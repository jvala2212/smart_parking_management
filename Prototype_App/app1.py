# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# Load the preprocessed data
parking_data = pd.read_csv("preprocessed_data.csv")
target_column = 'TOTALSPACECOUNT'
X = parking_data.drop(columns=[target_column, 'NAME', 'ADDRESS'])
y = parking_data[target_column]

# Impute missing values in the target variable
y = y.fillna(y.mean())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the preprocessing pipeline
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(exclude=['number']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ]
)

# Create the Gradient Boosting Regressor model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Create the final pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Streamlit App
st.title("Parking Space Prediction App")

# Input form for user to input features
st.header("Enter Features:")
feature_inputs = {}

# Iterate over numeric features
for feature in numeric_features:
    feature_inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Iterate over categorical features
for feature in categorical_features[:-1]:
    unique_values = X[feature].unique()
    selected_value = st.selectbox(f"Select {feature}", unique_values)
    feature_inputs[feature] = selected_value

# Create a date input box
date_input = st.date_input("Enter Date")

# Create a time input box
time_input = st.time_input("Enter Time")

# Combine date and time inputs
datetime_input = pd.to_datetime(str(date_input) + ' ' + str(time_input))

# Add the datetime input to the feature inputs
feature_inputs['LASTUPDATE'] = datetime_input

# Convert input features to DataFrame
input_data = pd.DataFrame([feature_inputs])
# Predict button
if st.button("Predict"):
    # Make predictions
    prediction = model_pipeline.predict(input_data)

    # Round the prediction to an integer value
    rounded_prediction = int(round(prediction[0]))

    # Display the rounded prediction
    st.header("Prediction:")
    st.write(f"Predicted Total Space Count: {rounded_prediction}")