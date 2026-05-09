import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tf_keras
from sklearn.model_selection import train_test_split

# 1. Load Data
print("Loading data from CSV...")
df = pd.read_csv('data/fingerCoords.csv', header=None)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Build Model
print("Building the neural network...")
# Use tf_keras to build so it's compatible with the converter later
model = tf_keras.models.Sequential([
    tf_keras.layers.Input(shape=(42,)),
    tf_keras.layers.Dense(25, activation='relu'),
    tf_keras.layers.Dense(15, activation='relu'),
    tf_keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 3. Train
print("Training starting (50 epochs)...")
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# 4. Export to TFLite
print("Exporting to TFLite...")
os.makedirs('model', exist_ok=True)

# CHANGE 2: Save using tf_keras
model.save('model/temp_model.h5')

# CHANGE 3: Load using tf_keras for the converter
loaded_model = tf_keras.models.load_model('model/temp_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

tflite_model = converter.convert()

with open('model/fingerClassifier.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"SUCCESS! Model saved. File size: {len(tflite_model)} bytes")