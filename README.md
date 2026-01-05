# Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Active-success)

**A deep learning-based safety system designed to prevent accidents by detecting driver drowsiness in real-time.**

## Project Overview
Drowsy driving is a major cause of road accidents worldwide. This project uses computer vision and deep learning to monitor a driver's facial features and alert them if signs of fatigue are detected.

Unlike simple threshold-based systems, this project utilizes a **Long Short-Term Memory (LSTM)** neural network. This allows the model to understand *temporal dependencies*—meaning it can distinguish between a natural blink (short duration) and a microsleep (long duration) by analyzing the sequence of frames over time.

## Key Features
* **Real-Time Detection:** Uses **MediaPipe Face Mesh** for ultra-fast and accurate landmark tracking (468 points).
* **Smart Blink Filtering:** The system ignores normal blinks and only triggers an alarm if eyes remain closed for a specific duration (approx. 1 second).
* **False Yawn Rejection:** Integrates geometric logic (Mouth Aspect Ratio) to verify if the mouth is physically open, preventing false "yawn" detections from random facial movements.
* **High Accuracy:** The custom-trained LSTM model achieved **97.67% test accuracy** on the validation dataset.
* **Audio Alerts:** Plays a loud alarm sound immediately upon detecting drowsiness.

## Tech Stack
* **Language:** Python 3.10
* **Deep Learning:** TensorFlow / Keras (LSTM, Dense layers)
* **Computer Vision:** OpenCV, MediaPipe
* **Data Handling:** NumPy, Pandas

## How It Works
The system follows a 4-stage pipeline:
1.  **Feature Extraction:** MediaPipe extracts eye and mouth coordinates from the live video feed.
2.  **Geometric Analysis:** It calculates:
    * **EAR (Eye Aspect Ratio):** Measures how "open" the eye is.
    * **MAR (Mouth Aspect Ratio):** Measures how "open" the mouth is.
3.  **Sequence Creation:** A sliding window of **30 frames** is created to capture the movement history.
4.  **Classification:** The LSTM model analyzes this sequence to predict one of three states:
    * **Active:** Normal driving.
    * **Drowsy:** Eyes closed for a prolonged period.
    * **Yawn:** Mouth open wide for a prolonged period.

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/samyuktharg/drowsiness-detection.git](https://github.com/samyuktharg/drowsiness-detection.git)
    cd drowsiness-detection
    ```

2.  **Create a Virtual Environment (Recommended)**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

**1. Start the System:**
Run the inference script to open the webcam and start detection.
```bash
python inference.py
