import os
# Suppress "oneDNN" warnings to keep the terminal clean
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import mediapipe as mp
from threading import Thread
import time

# --- ROBUST IMPORT SECTION ---
# This tries two different ways to find the model loader to prevent crashes
try:
    from tensorflow.keras.models import load_model
except (ImportError, AttributeError):
    try:
        from keras.models import load_model
    except ImportError:
        print("CRITICAL ERROR: Keras not found. Please ensure tensorflow is installed.")
        exit()

# --- CONFIGURATION ---
MODEL_PATH = 'drowsiness_model.h5'
ALARM_PATH = 'alarm.wav' 

# SENSITIVITY SETTINGS
# 0.5 = 50% confidence. You can raise this to 0.7 if it's still too sensitive.
CONFIDENCE_THRESHOLD = 0.5 
# INCREASED TO 25: Waits ~1 second of closed eyes before alarming (ignores blinks)
CONSECUTIVE_FRAMES_THRESHOLD = 25 

# --- LOAD MODEL ---
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- AUDIO SETUP ---
alarm_active = False

def play_alarm():
    """Plays the alarm sound in a background thread."""
    global alarm_active
    try:
        from playsound import playsound
        while alarm_active:
            playsound(ALARM_PATH)
    except Exception:
        # Fallback: Just print text if playsound fails or file is missing
        print("\n!!! WAKE UP !!!")
        time.sleep(1)

# --- MEDIAPIPE SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True
)

# Landmark Indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17]

# --- HELPER FUNCTIONS ---
def distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_ear(landmarks, indices):
    """Calculate Eye Aspect Ratio"""
    v1 = distance(landmarks[indices[1]], landmarks[indices[5]])
    v2 = distance(landmarks[indices[2]], landmarks[indices[4]])
    h = distance(landmarks[indices[0]], landmarks[indices[3]])
    return (v1 + v2) / (2.0 * h)

def get_mar(landmarks, indices):
    """Calculate Mouth Aspect Ratio"""
    v1 = distance(landmarks[indices[1]], landmarks[indices[5]])
    v2 = distance(landmarks[indices[2]], landmarks[indices[4]])
    h = distance(landmarks[indices[0]], landmarks[indices[3]])
    return (v1 + v2) / (2.0 * h)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
buffer_data = [] # Stores last 30 frames
drowsy_counter = 0
status = "ACTIVE"
color = (0, 255, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark
            
            # 1. Extract Features
            left_ear = get_ear(lm, LEFT_EYE)
            right_ear = get_ear(lm, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = get_mar(lm, MOUTH)
            
            # 2. Add to Buffer (Sliding Window)
            buffer_data.append([left_ear, right_ear, avg_ear, mar])
            if len(buffer_data) > 30:
                buffer_data.pop(0)
            
            # 3. Predict (Only once we have 30 frames)
            if len(buffer_data) == 30:
                input_seq = np.array([buffer_data])
                prediction = model.predict(input_seq, verbose=0)
                class_id = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # --- LOGIC & ALERTS ---
                
                # CLASS 0: ACTIVE
                if class_id == 0:
                    status = "ACTIVE"
                    color = (0, 255, 0) # Green
                    drowsy_counter = 0
                    alarm_active = False
                    
                # CLASS 1: DROWSY
                elif class_id == 1:
                    drowsy_counter += 1
                    status = "DROWSY"
                    color = (0, 0, 255) # Red
                    
                    # Only alarm if drowsy for 25 consecutive frames (~1 sec)
                    if drowsy_counter > CONSECUTIVE_FRAMES_THRESHOLD:
                        if not alarm_active:
                            alarm_active = True
                            t = Thread(target=play_alarm)
                            t.daemon = True
                            t.start()
                            
                # CLASS 2: YAWN
                elif class_id == 2:
                    # GEOMETRY CHECK: 
                    # Only classify as Yawn if the mouth is actually open (MAR > 0.3).
                    # This prevents "Blinking" from being detected as a Yawn.
                    if mar > 0.3:
                        status = "YAWN"
                        color = (0, 255, 255) # Yellow
                        drowsy_counter = 0
                        alarm_active = False
                    else:
                        # Model said Yawn, but mouth is closed? Ignore it.
                        status = "ACTIVE"
                        color = (0, 255, 0)
                        drowsy_counter = 0
                        alarm_active = False

            # Display Status on Screen
            cv2.putText(frame, f"Status: {status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            # Debug values (Optional - helps you see if MAR > 0.3 works)
            cv2.putText(frame, f"MAR: {mar:.2f} | Frames: {drowsy_counter}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Driver Drowsiness System', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        alarm_active = False
        break

cap.release()
cv2.destroyAllWindows()