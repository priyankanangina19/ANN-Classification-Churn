import pandas as Pd
import numpy as np
import pickle as pk 
import streamlit as st
import tensorflow as tf 
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import StandardScaler,LabelEncoder

model = tf.keras.models.load_model('model.h5')

#load rthe encoder and scaler
with open('onehot_encoder_geo.pk1','rb') as file:
    label_encoder_geo = pk.load(file)
with open('label_encoder_gender.pk1','rb') as file:
    label_encoder_gender = pk.load(file)
with open('scaler.pk1','rb') as file:
    scaler = pk.load(file)


st.title('Customer churn prediction')

#user input

geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age' , 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('credit score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products' , 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Memeber', [0,1])

##input data
input_data = Pd.DataFrame({
   "CreditScore": [credit_score],
   "Gender": [label_encoder_gender.transform([gender])[0]],
   "Age": [age],
   "Tenure": [tenure],
   "Balance": [balance],
   "NumOfProducts": [num_of_products],
   "HasCrCard": [has_cr_card],
   "IsActiveMember": [is_active_member],
   "EstimatedSalary": [estimated_salary]
})

##onehot encoding for geography

geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = Pd.DataFrame(geo_encoded, columns= label_encoder_geo.get_feature_names_out(['Geography']))

##combine one-hot encoded columns with input data
input_data = Pd.concat([input_data.reset_index(drop = True), geo_encoded_df], axis=1)

##scale the dta
input_data_scaled = scaler.transform(input_data)

##predic the churn
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

st.write(f'churn probability:{prediction_prob:.2f}')

if(prediction_prob>0.5):
    st.write('The customer is likely to churn.')
if(prediction_prob<0.5):
    st.write('The Customer is not likely to churn')



    
