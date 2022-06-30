#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st


# In[12]:


import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import re
pip install nltk
import nltk
nltk.download('punkt')
nltk.download('all-corpora')
nltk.download('wordnet')
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []


# In[4]:


model = pickle.load(open("nlp_classification.pkl","rb"))


# In[26]:


st.set_page_config(page_title="Review Classification")
st.title("App ReviewS classification ")
st.write("A simple machine learning app to predict a positive review with low rating")


# In[8]:


def main():          #to display streamlit app
    st.info(__doc__)
    file=st.file_uploader("Upload file", type=["csv"])
    show_file=st.empty()
    
    if not file:
        show_file.info("please upload a file of csv")
        return
    content=file.getvalue()
    data=pd.read_csv(file)
    st.dataframe(data.head(10))
    file.close()


# In[9]:


main()


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  
import joblib


# In[16]:


stop_words = stopwords.words("english")


# In[18]:


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):# function to clean the review
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
 
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
 
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
 
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
 
    # Return a list of words
    return text


# In[27]:


text=[]
if st.button("Predict"):
    for i in range (Star[i]):
        if Star[i]<3:
            Star[i]==0
        else:
            Star[i]==1
    for i in range (Star[i]):
        if Star[i]==0:
            text.append(Text[i])
    clean_review = text_cleaning(text)
 
    # load the model and make prediction
    model = joblib.load("nlp_classification.pkl")
 
    # make prection
    result = model.predict([clean_review])
    result1=[]
 
    #display positive reviews only
    for i in range (result[i]):
        if result[i]==1 and Star[i]==0:
            result1.append(text[i])


# In[28]:


st.write("## Thank you for Visiting \nProject by Amulya Ch")


# In[29]:


if __name__=="__main__":
    main()


# In[ ]:




