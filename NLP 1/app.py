import streamlit as st
import pandas as pd
import pickle
import neattext.functions as nfx
import re
import contractions
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

with open('vectorizer_tf.pkl', 'rb') as f:
    vect = pickle.load(f)

with open('best_model_NB.pkl', 'rb') as f:
    nb_model = pickle.load(f)

# Define text cleaning function
pattern = r"\&\#[0-9]+\;"

def clean(text):
    text = re.sub(r'https?://\S+', ' ', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\n', ' ', text)  # Remove newline characters
    text = re.sub(r'\r', ' ', text)  # Remove return characters
    text = contractions.fix(text)
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuations
    text = re.sub(r'@', '', text)  # Remove '@' characters
    text = nfx.normalize(text)  # Normalize text
    text = re.sub(r"\&\#[0-9]+\;", ' ', text)  # Remove HTML entities
    text = nfx.remove_userhandles(text)  # Remove user handles
    text = re.sub(r'\b\w{1}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)

    pos = scores['pos']
    neg = scores['neg']
    neu = scores['neu']
    compound = scores['compound']

    #return tuple
    return pos, neg, neu, compound

st.title("Sentiment Analysis and Test Labeling")

# get the user input 
user = st.text_input("Enter your text:")
cleaned_text = clean(user)

if cleaned_text:
    data = vect.transform([cleaned_text])

    if st.button("Label the sentence!"):
        output = nb_model.predict(data)
        if output[0] == 0:
            st.write('Bad Analysis')
        else:
            st.write('Good Analysis')

        pos, neg, neu, compound = analyze_sentiment(cleaned_text)
        
        fig = px.pie(values=[pos, neg, neu], 
                     names=['Positive', 'Negative', 'Neutral'], 
                     title='Sentiment Distribution of the Input Text')


        st.plotly_chart(fig) #display the chart
       
        st.write(f"Positive: {pos}, Negative: {neg}, Neutral: {neu}, Compound: {compound}")  # display the actual sentiment scores
