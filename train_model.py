import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- CONFIGURATION ---
WINDOW_SIZE = 30  # How many frames the LSTM looks at (e.g., 1 second)
NUM_FEATURES = 4  # left_ear, right_ear, avg_ear, mar
NUM_CLASSES = 3   # 0: Active, 1: Drowsy, 2: Yawn

# --- 1. LOAD AND CHECK DATA ---
df = pd.read_csv("drowsiness_data.csv")
print(f"Total Rows: {len(df)}")
print("Class Balance:\n", df['label'].value_counts())

# Basic Cleanup: Drop any accidental missing values
df.dropna(inplace=True)

# Select Features and Labels
# We usually use 'avg_ear' and 'mar'. 
# If you want to use left/right separately, change the columns below.
features = df[['ear_left', 'ear_right', 'ear_avg', 'mar']].values
labels = df['label'].values

# --- 2. CREATE SEQUENCES (SLIDING WINDOW) ---
# LSTM requires input shape: (Samples, Time Steps, Features)
# We turn the stream of data into overlapping "windows"
X = []
y = []

for i in range(len(features) - WINDOW_SIZE):
    window = features[i : i + WINDOW_SIZE] # Grab 30 frames
    label = labels[i + WINDOW_SIZE]        # Predict the label of the *next* frame
    X.append(window)
    y.append(label)

X = np.array(X)
y = np.array(y)

# One-Hot Encode Labels (0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1])
y = to_categorical(y, num_classes=NUM_CLASSES)

print(f"Input Shape (X): {X.shape}") # Should be (Samples, 30, 4)
print(f"Labels Shape (y): {y.shape}")

# --- 3. SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# --- 4. BUILD LSTM MODEL ---
model = Sequential([
    # Layer 1: LSTM
    # return_sequences=True because we are stacking another LSTM layer
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, NUM_FEATURES)),
    Dropout(0.2), # Prevents overfitting on small datasets
    
    # Layer 2: LSTM
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Layer 3: Dense (Classifier)
    Dense(32, activation='relu'),
    
    # Output Layer
    Dense(NUM_CLASSES, activation='softmax') # Softmax for probability (e.g., 0.8, 0.1, 0.1)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# --- 5. TRAIN ---
print("Starting Training...")
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_data=(X_test, y_test)
)

# --- 6. SAVE MODEL ---
model.save("drowsiness_model.h5")
print("Model saved as 'drowsiness_model.h5'")

# --- 7. EVALUATE ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Final Test Accuracy: {accuracy*100:.2f}%")