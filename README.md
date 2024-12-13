# AwakeGuard #

This project implements a real-time drowsiness detection system using computer vision and the Twilio API. It monitors a person's eye aspect ratio (EAR) to detect signs of drowsiness and triggers alerts via sound and phone calls.

## Features ##

* -Real-time detection: Uses a webcam to capture video and process frames to detect drowsiness.

* -Eye aspect ratio (EAR): * Calculates the EAR to determine if the user's eyes are closed for an extended period.

* -Alert mechanism: *

  -Displays alert messages on the video feed.

  -Plays an alarm sound.

  -Makes automated phone calls via the Twilio API when drowsiness persists.

* -Twilio integration: * Sends phone calls to notify when the user is potentially drowsy.

## Workflow ##

* 1. Twilio Alert Workflow *

The Twilio API is configured to make automated calls to a specified number if the drowsiness alarm is triggered more than 3 times. Below is an example of a Twilio call:

  -From: +14********

  -To: +91********

  -Message URL: http://demo.twilio.com/docs/voice.xml (can be customized for a personal message).


* 2. Alert Signal Indication *

The system displays an alert message and plays an alarm sound when drowsiness is detected.


* Requirements *

* Python Version *

  -Python 3.7 or newer

* Libraries: *

  -cv2 (OpenCV)

  -imutils

  -pygame

  -dlib

  -twilio

  -scipy

* Hardware: *

  -Webcam for real-time video capture

* Setup and Usage *

* Install Dependencies: *

pip install opencv-python imutils pygame dlib twilio scipy

* Download Facial Landmarks Model: *

Download the shape_predictor_68_face_landmarks.dat file from Dlib's repository and place it in the project directory.

* Run the Script: *

Execute the Python script:

python detect.py

* Twilio Configuration: *

  -Replace account_sid and auth_token with your Twilio account credentials.

  -Update the to and from phone numbers in the script.

## Key Variables ##

  -thresh: Threshold for the eye aspect ratio to indicate drowsiness.

  -frame_check: Number of consecutive frames with EAR below the threshold to trigger alerts.

  -alarm_counter: Tracks the number of times the alarm has been triggered to make phone calls.

## How It Works ##

1. Video Stream: Captures frames from the webcam.

2. Face Detection: Uses Dlib's frontal face detector to locate faces in the frame.

3. Landmarks Detection: Identifies 68 facial landmarks, focusing on the eyes.

4. EAR Calculation: Computes the average EAR for both eyes to determine if they are closed.

5. Alert Trigger: If EAR is below the threshold for a specified number of frames, it:

  -Displays alert messages.

  -Plays an alarm sound.

  -Makes a phone call using the Twilio API.

## Example Outputs ##

* -EAR Above Threshold: * No alerts triggered, EAR values are printed for debugging.

* -EAR Below Threshold: * Alerts are displayed on the video feed, and the alarm sound is played.

## Customization ##

  -Modify the thresh and frame_check values for sensitivity adjustment.

  -Customize the Twilio message by replacing the url parameter.

  -Change the alert sound by replacing music.wav with a different audio file.


## License ##
This project is licensed under the MIT License. Feel free to use, modify, and distribute it.

Note: Ensure your webcam is functional and the required libraries are installed before running the script.

