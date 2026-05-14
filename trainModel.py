import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tf_keras
from sklearn.model_selection import train_test_split

# 1. Load Data
print("Loading data from CSV...")
df = pd.read_csv('data/fingercords.csv', header=None)

# Separate features and labels
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

# Get the number of unique classes labels (should be 6)
num_classes = len(np.unique(y))
# Get the number of features
input_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build Model
print(f"Building neural network for {num_classes} classes with {input_features} inputs...")

model = tf_keras.models.Sequential([
    # Dynamically set input shape based on your CSV columns
    tf_keras.layers.Input(shape=(input_features,)), 
    
    tf_keras.layers.Dense(64, activation='relu'),
    tf_keras.layers.Dropout(0.2), # Added dropout to prevent overfitting on small datasets
    tf_keras.layers.Dense(32, activation='relu'),
    tf_keras.layers.Dense(16, activation='relu'),
    
    tf_keras.layers.Dense(num_classes, activation='softmax') 
])

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

# 3. Train
print("Training starting (80 epochs)...")
model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 4. Export to TFLite
print("Exporting to TFLite...")
os.makedirs('model', exist_ok=True)

model.save('model/temp_model.h5')
loaded_model = tf_keras.models.load_model('model/temp_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

tflite_model = converter.convert()

with open('model/fingerClassifier.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"SUCCESS! Model saved for {num_classes} signs. File size: {len(tflite_model)} bytes")