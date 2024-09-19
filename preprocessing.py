import nltk
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence):
    return ' '.join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)])

# Load intents file
with open('intents.json') as file:
    data = json.load(file)

patterns = []
responses = []
classes = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(clean_up_sentence(pattern))
        responses.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Encode classes
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(responses)

# Create and train the model
model = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])
model.fit(patterns, y)

# Save model and encoders
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('label_encoder.pkl', 'wb') as le_file:
    pickle.dump(label_encoder, le_file)
with open('classes.pkl', 'wb') as classes_file:
    pickle.dump(classes, classes_file)
