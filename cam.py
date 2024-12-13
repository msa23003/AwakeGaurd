import streamlit as st
import cv2
import imutils
import dlib
from pygame import mixer
from scipy.spatial import distance
from twilio.rest import Client
from imutils import face_utils
from datetime import datetime
import pandas as pd
import os
from PIL import Image
import base64
from geopy.geocoders import Nominatim
import time

# Twilio configuration
account_sid = "AC2873fa6cfd374bd8f2bc736b6fda2444"
auth_token = "dd198819226e8b648213b5dcbbba9b92"
client = Client(account_sid, auth_token)

# Initialize pygame mixer for playing alert sound
mixer.init()
mixer.music.load(r"C:\Users\Dell\OneDrive\Desktop\driver-drowsiness-detection\drowsy_driver\music.wav")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for drowsiness detection
thresh = 0.25
frame_check = 20  # Reduced to half for quicker response

# Initialize dlib's face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"C:\Users\Dell\OneDrive\Desktop\driver-drowsiness-detection\drowsy_driver\shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize alert log DataFrame to store alert information
alert_log = pd.DataFrame(columns=["Screenshot", "Time", "Alert Number", "Location"])

# Ensure screenshots directory exists
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

# Function to get geolocation of the user
def get_location():
    geolocator = Nominatim(user_agent="drowsiness_detection")
    location = geolocator.geocode("your location")
    return location.address if location else "Location not available"

def start_detection():
    global alert_log
    cap = cv2.VideoCapture(0)
    flag = 0
    alert_number = 0
    alert_displayed = {1: False, 2: False, 3: False}  # To track which alert has been shown

    if not cap.isOpened():
        st.error("Error: Could not access the camera.")
        return

    while alert_number < 3:
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=1200)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)

        if len(subjects) > 0:
            for subject in subjects:
                shape = predict(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < thresh:
                    flag += 1
                    if flag >= frame_check:
                        # Alert triggered
                        if not alert_displayed[alert_number + 1]:
                            cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            if not mixer.music.get_busy():
                                mixer.music.play()
                                alert_number += 1
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                screenshot_path = f"screenshots/alert_{alert_number}_{timestamp.replace(':', '-')}.jpg"
                                cv2.imwrite(screenshot_path, frame)

                                # Get location
                                location = get_location()
                                # Shorten location to city/town name
                                location_parts = location.split(',')[0]
                                location_short = location_parts.split(' ')[0]

                                # Log alert details
                                alert_log = pd.concat([alert_log, pd.DataFrame({
                                    "Screenshot": [screenshot_path],
                                    "Time": [timestamp],
                                    "Alert Number": [alert_number],
                                    "Location": [location_short]
                                })], ignore_index=True)

                                alert_displayed[alert_number] = True  # Mark this alert as displayed

                            time.sleep(2)  # Add delay before next alert

                            # If 3 alerts have been triggered, break the loop
                            if alert_number >= 3:
                                break
                else:
                    flag = 0

        # Display the latest alert log and image
        for i in range(1, 4):
            if alert_displayed[i]:  # Display only if this alert has been triggered
                latest_alert = alert_log[alert_log['Alert Number'] == i].iloc[-1]
                st.markdown(f"<h3 style='color: red;'>⚠️ ALERT #{latest_alert['Alert Number']} ⚠️</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: gray;'>Time: {latest_alert['Time']} | Location: {latest_alert['Location']}</p>", unsafe_allow_html=True)
                
                # Display the image
                img = Image.open(latest_alert['Screenshot'])
                st.image(img, caption="Drowsiness Detected!", use_column_width=True)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit UI
st.title("🚗 AwakeGuard 🚗")
st.write("Click the button below to start the drowsiness detection.")

# Custom start detection button
start_button = st.button("Start Detection")

if start_button:
    # Start the detection if button clicked
    start_detection()

# Styling the page with custom HTML and CSS
st.markdown("""
    <style>
        body {
            background-color: #f0f4f8;
            font-family: 'Arial', sans-serif;
        }
        .alert-container {
            margin-top: 20px;
            padding: 15px;
            background-color: #ffe6e6;
            border-radius: 10px;
            border: 2px solid #f44336;
        }
        h3 {
            color: #f44336;
            font-size: 24px;
            text-align: center;
        }
        p {
            color: #333;
            font-size: 16px;
            text-align: center;
        }
        .stButton>button {
            background-color: #ff5722;
            color: white;
            border-radius: 8px;
            padding: 15px;
            font-size: 18px;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #f44336;
        }
    </style>
""", unsafe_allow_html=True)
