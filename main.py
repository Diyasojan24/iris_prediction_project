import streamlit as st
import pandas as pd
import numpy as np
from os import path
import pickle

st.title("Iris dataset")
df_iris=pd.read_csv(path.join("Data","iris.csv"))
st.write(df_iris)
st.title("scatter chart of Iris dataset")
st.scatter_chart(df_iris[['sepal_length','sepal_width']])
st.title("Flower species Predictor")
petal_length=st.number_input("please choose the petal length",placeholder="please enter a value between 1.0 and 6.9",min_value=1.0,max_value=6.9,value=None)
petal_width=st.number_input("please choose the petal width",placeholder="please enter a value between 0.1 and 2.5",min_value=0.1,max_value=2.5,value=None)
#sepal_length=st.slider("a")
sepal_length=st.number_input("please choose the sepal length",placeholder="please enter a value between 4.3 and 7.9",min_value=4.3,max_value=7.9,value=None)
sepal_width=st.number_input("please choose the sepal width",placeholder="please enter a value between 2.0 and 4.4",min_value=2.0,max_value=4.4,value=None)
#prepare the dataframe for prediction
df_user_input=pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]],
             columns=['sepal_length','sepal_width','petal_length','petal_width'])

#using the .pkl file,creating an ML model named 'iris_predictor'
model_path=path.join("Model","iris_model.pkl")
with open (model_path,'rb')as file:
    iris_predictor=pickle.load(file)

dict_species={0:'setosa',1:'versicolor',2:'virginica'}
#reading the dataset
if st.button("predict_species"):
    if((petal_length==None)or(petal_width==None)or(sepal_length==None)or(sepal_width==None)):
        #will be executed when any of the values is not enterd properly
         st.write("please fill all values")
    else:
    #prediction can be done here
         predicted_species= iris_predictor.predict(df_user_input)
         #predicted_species[0] will give us the value in the dataframe
    #we use that value to find the corresponding species from the
    #dictionary 'dict_species'
         st.write("the species is:",dict_species[predicted_species[0]])

