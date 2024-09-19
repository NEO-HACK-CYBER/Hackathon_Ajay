import nltk
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline

# Load model and data
lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
    intents = json.load(file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('label_encoder.pkl', 'rb') as le_file:
    label_encoder = pickle.load(le_file)
with open('classes.pkl', 'rb') as classes_file:
    classes = pickle.load(classes_file)

def clean_up_sentence(sentence):
    return ' '.join([lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence)])

def predict_class(sentence):
    bow = clean_up_sentence(sentence)
    prediction = model.predict([bow])[0]
    return {'intent': classes[prediction], 'probability': '1.0'}

def get_response(intent, intents_json):
    for i in intents_json['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "Sorry, I don't understand."

# Main loop to interact with the bot
print("GO! Study Bot is running!")

while True:
    message = input("You: ")
    intent_info = predict_class(message)
    response = get_response(intent_info['intent'], intents)
    print(f"Bot: {response}")
