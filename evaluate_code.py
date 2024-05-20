import os
import pickle
import speech_recognition as sr
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

model_path = '/content/emotion_detection_model.pkl'
tokenizer_path = '/content/tokenizer.pkl'

with open(model_path, 'rb') as f:
    trained_model = pickle.load(f)

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

emotion_labels = ['Sad', 'Happy', 'Fear', 'Angry', 'Disgust', 'Neutral', 'Surprised']

label_encoder = LabelEncoder()
label_encoder.fit(emotion_labels)

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)

    with audio_file as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio"

def preprocess_text(text, tokenizer, max_sequence_length):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
    tokens = tokenizer.sequences_to_texts(sequences)[0].split()  # Extract tokens
    return tokens, sequences, padded_sequences

def predict_emotion(audio_path, model, tokenizer, max_sequence_length, label_encoder):
    text = speech_to_text(audio_path)
    print(f"Recognized Text: {text}")

    if text in ["Could not understand the audio", "Could not request results; check your network connection"]:
        return text

    tokens, sequences, padded_sequences = preprocess_text(text, tokenizer, max_sequence_length)
    print(f"Tokens: {tokens}")

    prediction = model.predict(padded_sequences)
    predicted_label = np.argmax(prediction, axis=1)

    predicted_emotion = label_encoder.inverse_transform(predicted_label)[0]

    return predicted_emotion

input_audio_path = '/content/custom2.wav'

max_sequence_length = 8  # Replace with the actual max_sequence_length used during training

predicted_emotion = predict_emotion(input_audio_path, trained_model, tokenizer, max_sequence_length, label_encoder)
print(f"Predicted Emotion: {predicted_emotion}")
