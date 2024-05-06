from google.colab import drive
drive.mount('/content/drive')
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
# Emotion mapping based on folder names
emotion_mapping = {
    'Sad': 0,
    'Happy': 1,
    'Fear': 2,
    'Angry': 3,
    'Disgust': 4,
    'Neutral': 5,
    'Surprised': 6
}

# Function to load data from a folder structure
def load_data(dataset_path):
    features = []
    labels = []
    for emotion, idx in emotion_mapping.items():
        emotion_path = os.path.join(dataset_path, emotion)
        if not os.path.exists(emotion_path):
            continue
        files = [os.path.join(emotion_path, file) for file in os.listdir(emotion_path) if file.endswith('.wav')]
        for file in files:
            data, sample_rate = librosa.load(file, sr=None)
            mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
            if mfccs.shape[1] > 1:
                mfccs = np.mean(mfccs.T, axis=0)
                features.append(mfccs)
                labels.append(idx)
            else:
                logging.warning(f"File {file} is too short for MFCC features.")
    return np.array(features), to_categorical(np.array(labels), num_classes=len(emotion_mapping))

# Define the LSTM model architecture
def build_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(len(emotion_mapping), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model
# Load data and train the model
dataset_path = '/content/drive/MyDrive/Emotion_Audio/Emotion_Audio'
features, labels = load_data(dataset_path)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Build the model
model = build_model((40, 1))  # Ensure features are reshaped correctly if needed
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Save the final model
model.save('/content/drive/MyDrive/Models/final_emotion_model.h5')
def predict_emotion_from_audio(file_path, model):
    data, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    if mfccs.shape[1] > 1:
        mfccs = np.mean(mfccs.T, axis=0)
        mfccs = np.expand_dims(np.expand_dims(mfccs, axis=0), axis=2)
        predictions = model.predict(mfccs)
        emotion_index = np.argmax(predictions)
        return list(emotion_mapping.keys())[emotion_index]
    else:
        logging.error("Audio file is too short for predictions.")
        return None
from google.colab import files
import librosa
import numpy as np

# Load the trained model
model_path = '/content/drive/MyDrive/Models/final_emotion_model.h5'
model = load_model(model_path)

# Emotion mapping
emotion_mapping = {
    'Sad': 0,
    'Happy': 1,
    'Fear': 2,
    'Angry': 3,
    'Disgust': 4,
    'Neutral': 5,
    'Surprised': 6
}

def predict_emotion_from_audio(model):
    uploaded = files.upload()  # This will prompt you to upload a file
    if not uploaded:
        return "No file uploaded."
    file_path = next(iter(uploaded))
    data, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    if mfccs.shape[1] > 1:
        mfccs = np.mean(mfccs.T, axis=0)
        mfccs = np.expand_dims(np.expand_dims(mfccs, axis=0), axis=2)
        predictions = model.predict(mfccs)
        emotion_index = np.argmax(predictions)
        return f"Predicted Emotion for '{file_path}': {list(emotion_mapping.keys())[emotion_index]}"
    else:
        return "Audio file is too short for predictions."

# Call the function to upload a file and predict emotion
print(predict_emotion_from_audio(model))
