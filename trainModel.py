import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tf_keras
from sklearn.model_selection import train_test_split
from utils.hand_processing import labels

# 1. Load Data
csv_path = '/fingercords.csv' 

print(f"Loading data from {csv_path}...")
df = pd.read_csv(csv_path, header=None)

# Separate features (X) and labels (y)
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

num_classes = len(labels)

input_features = X.shape[1] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build Model
print(f"Building neural network for {num_classes} possible classes with {input_features} inputs...")

model = tf_keras.models.Sequential([
    # This dynamically adapts to 84 inputs
    tf_keras.layers.Input(shape=(input_features,)), 
    
    tf_keras.layers.Dense(128, activation='relu'),
    tf_keras.layers.Dropout(0.2), 
    tf_keras.layers.Dense(64, activation='relu'),
    tf_keras.layers.Dense(32, activation='relu'),
    
    # Final layer: num_classes must stay consistent!
    tf_keras.layers.Dense(num_classes, activation='softmax') 
])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# 3. Train
print("Training starting...")
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 4. Export to TFLite
print("Exporting to TFLite...")
os.makedirs('model', exist_ok=True)

# Save and Convert
model.save('model/temp_model.h5')
loaded_model = tf_keras.models.load_model('model/temp_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

with open('model/fingerClassifier.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"SUCCESS! Model saved. Architecture: {input_features} in -> {num_classes} out.")