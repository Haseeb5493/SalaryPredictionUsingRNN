import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# Load the model
import tensorflow as tf
my_model = tf.keras.models.load_model('model.h5')  # Use your regression model

# Load encoders and scaler
with open('onehot_encoder_geo.pkl','rb') as file:
    Label_encoder_geography = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    Label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

st.title('Customer Salary Prediction')

# User INPUT
geography = st.selectbox('Geography', Label_encoder_geography.categories_[0])
gender    = st.selectbox('Gender', Label_encoder_gender.classes_)
age       = st.slider('Age',18,92)
balance   = st.number_input('Balance')
credit_score     = st.number_input('Credit Score')
tenure           = st.slider('Tenure',0,10)      
num_of_products  = st.slider('Number of products',1,4)
has_cr_card      = st.selectbox('Has credit card',[0,1])
active_member    = st.selectbox('Is active member',[0,1])

# Create input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [Label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [active_member],
})

# Encode Geography
geo_encoded = Label_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=Label_encoder_geography.get_feature_names_out(['Geography']))

# Combine all features
full_input = pd.concat([geo_encoded_df, input_data], axis=1)

# Ensure correct column order
expected_columns = scaler.feature_names_in_
full_input = full_input[expected_columns]

# Scale the input
data_scaled = scaler.transform(full_input)
 
# Predict salary
predicted_salary = my_model.predict(data_scaled)[0][0]

st.write(f"Predicted Estimated Salary: {predicted_salary}")
