import os
import speech_recognition as sr
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
    except sr.RequestError:
        return "Could not request results; check your network connection"

def collect_data(dataset_folder):
    data = []
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith('.wav'):  # Process only audio files
                audio_path = os.path.join(root, file)
                emotion_label = os.path.basename(root)  # Assuming the subfolder name is the emotion label

                # Convert speech to text
                text = speech_to_text(audio_path)

                # Append transcribed text and emotion label to data list
                data.append((text, emotion_label))
    return data

def train_model(data):
    # Extract texts and labels from the data
    texts = [text for text, _ in data]
    labels = [label for _, label in data]

    # Tokenize the texts
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad sequences to ensure uniform length
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

    # Define the LSTM model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100
    lstm_units = 128

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
        Bidirectional(LSTM(units=lstm_units)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    return model, tokenizer, max_sequence_length

# Path to the dataset folder
dataset_folder = '/content/drive/MyDrive/Emotion_Audio/Emotion_Audio'

# Collect data from the dataset folder
data = collect_data(dataset_folder)

# Train the LSTM model using the collected data
trained_model, tokenizer, max_sequence_length = train_model(data)

# Save the trained model and tokenizer as pickle files
pickle_file_path = '/content/emotion_detection_model.pkl'
tokenizer_file_path = '/content/tokenizer.pkl'

with open(pickle_file_path, 'wb') as f:
    pickle.dump(trained_model, f)

with open(tokenizer_file_path, 'wb') as f:
    pickle.dump(tokenizer, f)

print(f"Model saved as pickle file: {pickle_file_path}")
print(f"Tokenizer saved as pickle file: {tokenizer_file_path}")
