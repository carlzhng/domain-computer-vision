import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# 1. Load Data
print("Loading data from CSV...")
df = pd.read_csv('data/fingerCoords.csv', header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Build Model
print("Building the neural network...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(42,)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Train
print("Training starting (50 epochs)...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# 4. Export to TFLite
print("Exporting to TFLite...")
os.makedirs('model', exist_ok=True)

# Try saving the model to a file first, then converting the file
model.save('model/temp_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model('model/temp_model.h5'))

tflite_model = converter.convert()

with open('model/fingerClassifier.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"SUCCESS! Model saved. File size: {len(tflite_model)} bytes")