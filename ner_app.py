import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Dictionary that maps integer values to named entities
idx2tag = {
    0: 'O',
    1: 'I-abstract',
    2: 'I-event',
    3: 'I-place',
    4: 'B-abstract',
    5: 'B-person',
    6: 'I-person',
    7: 'I-object',
    8: 'B-place',
    9: 'B-object',
    10: 'B-event',
    11: 'I-time',
    12: 'I-organization',
    13: 'I-substance',
    14: 'B-time',
    15: 'B-organization',
    16: 'B-substance',
    17: 'I-quantity',
    18: 'I-plant',
    19: 'B-plant',
    20: 'B-animal',
    21: 'I-animal',
    22: 'B-quantity',
}

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the pre-trained model
model = load_model('bilstm.h5')

# Function to predict named entities in text
def predict_entities(text):
    sequence = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(sequence, padding='post', maxlen=50)
    predictions = model.predict(pad_seq)
    return predictions

# Function to convert predictions into named entity tags
def get_named_entities(predictions):
    pred_tags = [[idx2tag[np.argmax(pred)]] for pred in predictions[0]]
    return pred_tags

# Streamlit app
st.title('Named Entity Recognition App')
input_text = st.text_input('Enter text here')
if st.button('Predict'):
    predictions = predict_entities(input_text)
    pred_tags = get_named_entities(predictions)
    for token, tag in zip(input_text.split(), pred_tags):
        st.write(f'{token} - {tag}')



















