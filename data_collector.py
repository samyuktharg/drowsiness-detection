import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

# --- 1. Setup MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True # Enables detailed eye/iris locations
)

# --- 2. Define Landmark Indices ---
# These specific points map to the eye and mouth corners/edges
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 39, 181, 0, 17] # Outer lips

# --- 3. Helper Functions ---
def distance(p1, p2):
    """Euclidean distance between two points."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_ear(landmarks, indices):
    """Calculates Eye Aspect Ratio (EAR)."""
    # Vertical lines
    v1 = distance(landmarks[indices[1]], landmarks[indices[5]])
    v2 = distance(landmarks[indices[2]], landmarks[indices[4]])
    # Horizontal line
    h = distance(landmarks[indices[0]], landmarks[indices[3]])
    
    # EAR Formula
    ear = (v1 + v2) / (2.0 * h)
    return ear

def get_mar(landmarks, indices):
    """Calculates Mouth Aspect Ratio (MAR)."""
    v1 = distance(landmarks[indices[1]], landmarks[indices[5]])
    v2 = distance(landmarks[indices[2]], landmarks[indices[4]])
    h = distance(landmarks[indices[0]], landmarks[indices[3]])
    
    mar = (v1 + v2) / (2.0 * h)
    return mar

# --- 4. CSV Setup ---
file_name = "drowsiness_data.csv"
file_exists = os.path.isfile(file_name)

# Open CSV in Append mode ('a') so we don't overwrite previous sessions
f = open(file_name, 'a', newline='')
writer = csv.writer(f)

# Write header only if file is new
if not file_exists:
    writer.writerow(['ear_left', 'ear_right', 'ear_avg', 'mar', 'label'])
    print("Created new CSV file.")
else:
    print("Appending to existing CSV file.")

# --- 5. Main Loop ---
cap = cv2.VideoCapture(0) # 0 for default webcam

print("Starting Collection... Press 'a' (Active), 'd' (Drowsy), 'y' (Yawn) to record. 'q' to Quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror view and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    h, w, _ = frame.shape
    
    label_text = "Press a/d/y to Record"
    color = (255, 255, 255) # White

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Calculate Features
            left_ear = get_ear(landmarks, LEFT_EYE)
            right_ear = get_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = get_mar(landmarks, MOUTH)
            
            # --- Recording Logic ---
            k = cv2.waitKey(1)
            
            if k == ord('a'): # ACTIVE
                writer.writerow([left_ear, right_ear, avg_ear, mar, 0])
                label_text = "RECORDING: ACTIVE (0)"
                color = (0, 255, 0) # Green
                
            elif k == ord('d'): # DROWSY
                writer.writerow([left_ear, right_ear, avg_ear, mar, 1])
                label_text = "RECORDING: DROWSY (1)"
                color = (0, 0, 255) # Red
                
            elif k == ord('y'): # YAWN
                writer.writerow([left_ear, right_ear, avg_ear, mar, 2])
                label_text = "RECORDING: YAWN (2)"
                color = (0, 255, 255) # Yellow

            elif k == ord('q'): # QUIT
                print("Data saved. Exiting.")
                f.close()
                cap.release()
                cv2.destroyAllWindows()
                exit()
            
            # Display values on screen
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, label_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Data Collector', frame)

cap.release()
cv2.destroyAllWindows()