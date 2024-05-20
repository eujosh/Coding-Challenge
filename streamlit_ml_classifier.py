import pandas as pd
import streamlit as st
import joblib

st.title("Comment Classification App")
st.write("")

model = joblib.load('ml_classifier.pkl')

import re, string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):  # Fixed typo in function name
    text = str(text)
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]
    return ' '.join(words)

keywords_medical = ['medical', 'doctor', 'surgery', 'consultant', 'treatment', 'clinic', 'patient', 'healthcare', 'emergency']
keywords_vet = ['consult', 'animal', 'treatment', 'vet','veterinary','dog', 'care', 'clinic', 'pet']

# Function to check if any keyword is present in a comment
def find_keywords(comment, keywords):
    return 1 if any(keyword.lower() in str(comment).lower() for keyword in keywords) else 0


text = st.text_input("Write a comment")

if st.button("Submit"):  # Corrected button creation syntax
    if text:
        with st.spinner('Just wait....'):
            data = pd.DataFrame({'comments':[text], 'count':[len(text.split(" "))]})  # Fixed DataFrame initialization
            data['comments'] = data['comments'].apply(preprocess_text)
            # Apply the function to create new columns
            data['medic'] = data['comments'].apply(lambda x: find_keywords(x, keywords_medical))
            data['vet'] = data['comments'].apply(lambda x: find_keywords(x, keywords_vet))
            result = model.predict(data)[0]

        st.success(result)
    else:
        st.error("Please enter a comment to classify.")