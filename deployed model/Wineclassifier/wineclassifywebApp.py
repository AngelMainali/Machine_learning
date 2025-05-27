# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 09:55:58 2025

@author: Dell
"""

import numpy as np
import pickle
import streamlit as st

#loading saved model
loaded_model=pickle.load(open('C:/Users/Dell/Desktop/machine_leraning/Deployed Model/WineClassifier/trained_model.sav','rb'))



#creating a function for Prediction

def wineClassifier(input_data):
    
    #change the input data in  numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the data as we are predicting label for only one instances
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)


    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction==1:
      return 'Good Quality Wine'
    else:
      return'Bad quality wine'

def main():
    
    
    #giving a title for a user interface
    st.title('Wine Classifier Web App')
    
    
    #input data fields
    # [fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol ]
    
    fixed_acidity=st.text_input('Number of Fixed Acidity')
    volatile_acidity=st.text_input('Number of volatile Acidity')
    citric_acid=st.text_input('Number of citric acid')
    residual_sugar=st.text_input('Number of residual sugar')
    chlorides=st.text_input('Number of chlorides')
    free_sulfur_dioxide=st.text_input('free sulfur dioxide')
    total_sulfur_dioxide=st.text_input('Number of total sulfur dioxide')
    density=st.text_input('Number of density')
    pH=st.text_input('Number of pH')
    sulphates=st.text_input('Number of sulphates')
    alcohol=st.text_input('Number of lcohol')
    
    
    #code for prediction
    result= ''
    
    
    #creating a button for prediction
    
    if st.button('Wine Classifier'):
        result=wineClassifier([fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol])
        
    st.success(result)   
    
    
    
if __name__=='__main__':
    main()    
    