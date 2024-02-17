# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# Load the preprocessed data
parking_data = pd.read_csv("updated_data.csv")
target_column = 'AVAILABLESPACECOUNT'  # Change this line
X = parking_data.drop(columns=[target_column,'NAME','ADDRESS','ADASPACECOUNT','EVSPACECOUNT'])
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

# Create the Linear Regression model
model = LinearRegression()

# Create the final pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Streamlit App
st.title("Parking Space Prediction App")

# Provide instructions
st.markdown("""
    ## Instructions
    - This application predicts the available parking space count based on user inputs.
    - Please select the desired facility by choosing the 'FACILITYID' from the dropdown menu.
    - Fill in the required information, and click the 'Predict' button to get the prediction.
""")


# Input form for user to input features
st.header("Enter Features:")
feature_inputs = {}

# Select ownership
ownerships = X['OWNERSHIP'].unique()
selected_ownership = st.selectbox("1. Select OWNERSHIP", ownerships)
feature_inputs['OWNERSHIP'] = selected_ownership

# Select parking type
available_parking_types = X[X['OWNERSHIP'] == selected_ownership]['PARKINGTYPE'].unique()
selected_parking_type = st.selectbox("2. Select PARKINGTYPE", available_parking_types)

if selected_parking_type not in available_parking_types:
    st.warning(f"Warning: Parking type '{selected_parking_type}' is not available for the selected ownership.")

feature_inputs['PARKINGTYPE'] = selected_parking_type

# Get available levels based on selected ownership and parking type
available_levels = X[(X['OWNERSHIP'] == selected_ownership) & (X['PARKINGTYPE'] == selected_parking_type)]['NUMBEROFLEVELS'].unique()

# Select number of levels
feature_inputs['NUMBEROFLEVELS'] = st.selectbox("3. Select NUMBEROFLEVELS", available_levels)

# Get available facility IDs based on selected ownership and parking type
available_facilities = X[(X['OWNERSHIP'] == selected_ownership) & (X['PARKINGTYPE'] == selected_parking_type)]['FACILITYID'].unique()

# Select facility ID
feature_inputs['FACILITYID'] = st.selectbox("4. Select FACILITYID", available_facilities)

# Select reserved space count
feature_inputs['RESERVEDSPACECOUNT'] = st.number_input("5. Enter RESERVEDSPACECOUNT", value=0)

# Create a date input box
date_input = st.date_input("6. Enter Date")

# Create a time input box
time_input = st.time_input("7. Enter Time")

# Combine date and time inputs
datetime_input = pd.to_datetime(str(date_input) + ' ' + str(time_input))

# Add the datetime input to the feature inputs
feature_inputs['LASTUPDATE'] = datetime_input

# Calculate the total capacity based on the selected 'OWNERSHIP', 'PARKINGTYPE', and 'FACILITYID'
total_capacity = X[(X['OWNERSHIP'] == selected_ownership) & 
                   (X['PARKINGTYPE'] == selected_parking_type) & 
                   (X['FACILITYID'] == feature_inputs['FACILITYID'])]['TOTALSPACECOUNT'].iloc[0]
feature_inputs['TOTALSPACECOUNT'] = total_capacity


# Convert input features to DataFrame
input_data = pd.DataFrame([feature_inputs])

# Predict button
if st.button("Predict"):
    # Make predictions
    prediction = model_pipeline.predict(input_data)

    # Round the prediction to an integer value
    rounded_prediction = int(round(prediction[0]))

    # Display the rounded prediction, input TOTALSPACECOUNT, total capacity, and facility information
    st.header("Prediction:")
    st.write(f"Predicted Available Space Count: {rounded_prediction}")
    st.write(f"Total Parking Space Capacity: {total_capacity}")
    
    