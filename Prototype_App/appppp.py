import streamlit as st
import pandas as pd
import pickle

# Load the preprocessed data
parking_data = pd.read_csv("preprocessed_data.csv")
target_column = 'TOTALSPACECOUNT'
X = parking_data.drop(columns=[target_column, 'NAME', 'ADDRESS'])
y = parking_data[target_column]

# Impute missing values in the target variable
y = y.fillna(y.mean())

# Streamlit App
st.title("Parking Space Prediction App")

# Input form for user to input features
st.header("Enter Features:")
feature_inputs = {}

# Iterate over numeric features
for feature in X.select_dtypes(include=['number']).columns:
    feature_inputs[feature] = st.number_input(f"Enter {feature}", value=0.0)

# Iterate over categorical features
for feature in X.select_dtypes(exclude=['number']).columns:
    unique_values = X[feature].unique()
    selected_value = st.selectbox(f"Select {feature}", unique_values)
    feature_inputs[feature] = selected_value

# Convert input features to DataFrame
input_data = pd.DataFrame([feature_inputs])

# Load the Gradient Boosting Regressor model from the pickle file
with open('gradient_boosting_regressor_model.pkl', 'rb') as model_file:
    loaded_model_pipeline = pickle.load(model_file)

# Make predictions
prediction = loaded_model_pipeline.predict(input_data)

# Display the prediction
st.header("Prediction:")
st.write(f"Predicted Total Space Count: {prediction[0]}")

