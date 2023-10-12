#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.feature_extraction.text import CountVectorizer
  
# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)
  
def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs

messages = pd.read_csv('smsspamclassifier.txt', sep='\t',
                           names=["label", "message"])
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

def prediction(text_message):
    review = re.sub('[^a-zA-Z]', ' ', text_message)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    cv = CountVectorizer(max_features=5000)
    X = cv.fit_transform(corpus).toarray()
    prediction = classifier.predict(X)
    return prediction
      
# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title("SPAM PREDICTION")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit SPAM PREDICTION ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    text_message = st.text_input("MESSAGE", "Type Here")
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    output=''
    text_message=str(text_message)
    if st.button("Predict"):
        result = prediction(text_message)
        if result[-1]=='1':
            output='SPAM'
        else:
            output='NOT SPAM'
    st.success('The output is {}'.format(output))
     
if __name__=='__main__':
    main()


# In[ ]:




