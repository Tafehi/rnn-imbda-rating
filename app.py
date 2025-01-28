import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import streamlit as st



model = load_model('simple_rnn_imdb.h5')

import requests

# Load the IMDB dataset word index
url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json'
response = requests.get(url)
word_index = response.json()
reverse_word_index = {value: key for key, value in word_index.items()}



# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


### Prediction  function

def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)

    prediction=model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]

## Streamlit Application

st.title('Sentiment Analysis with RNNs')
st.write('This is a simple example of a sentiment analysis model using a RNN.')



# Text input field
review = st.text_input("Enter a review to analyze:")

# Display the input text
if review:
    st.write(f"You entered: {review}")
    predict_sentiment(review)
    sentiment, confidence = predict_sentiment(review)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Confidence: {confidence:.4f}')