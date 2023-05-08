#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


  
# loading in the model to predict on the data
pickle_in = open('VGG16.pkl', 'rb')
model = pickle.load(pickle_in)
  
def welcome():
    return 'welcome all'
  
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x)

    preds = model.predict(x)
    return preds

      
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title("IMAGE CLASSIFICATION")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit IMAGE CLASSIFICATION ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    message = st.file_uploader("IMAGE", "jpg")
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    output=''
    
    if st.button("Predict"):
        preds = model_predict(message,model)
        predicted_label_index = np.argmax(preds)# Simple argmax
        flowers_labels_dict = {
            0 :' ROSES ',
            1 :' DAISY ',
            2:' DANDELION ' ,
            3:' SUNFLOWERS ' ,
            4:' TULIPS '
        }
        result = flowers_labels_dict[predicted_label_index]
    st.success('The output is {}'.format(result))
     
if __name__=='__main__':
    main()


# In[ ]:




