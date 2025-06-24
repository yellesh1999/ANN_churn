import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

#load trained model
model = tf.keras.models.load_model('My_model.keras')

with open('ohe_encoder_Geo.pkl','rb') as file:
    ohe_encoder_Geo = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler= pickle.load(file)

## stream lit
st.title('customer churn prediction')

#user input
geography = st.selectbox('Geography',ohe_encoder_Geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])   

input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})
 ## one hot encoded geography
geo_encoded=ohe_encoder_Geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=ohe_encoder_Geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_scaled = scaler.transform(input_data)

#predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'churn probability:{prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('the customer is likely to churn.')
else:
    st.write('the customer is not likely to churn')
