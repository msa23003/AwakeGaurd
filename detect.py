import time
from scipy.spatial import distance
from pygame import mixer
import imutils
import dlib
import cv2
from twilio.rest import Client
from imutils import face_utils

# Twilio configuration"
account_sid = ""
auth_token = ""
client = Client(account_sid, auth_token)

# Initialize pygame mixer for playing alert sound
mixer.init()
mixer.music.load("music.wav")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Constants for drowsiness detection
thresh = 0.25  # EAR threshold to indicate drowsiness
frame_check = 40  # Number of consecutive frames for which EAR is below threshold to trigger alert

# Initialize dlib's face detector and shape predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define the indexes for the left and right eye landmarks in the facial landmarks array
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start capturing video from the webcam (index 0)
cap = cv2.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

flag = 0  # Counter to track consecutive frames where drowsiness is detected
alarm_counter = 0  # Counter to track how many times the alarm has been triggered

while True:
    ret, frame = cap.read()  # Read a frame from the video stream
    frame = imutils.resize(frame, width=1200)  # Resize the frame for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)
    
    # Check if faces are detected
    if len(subjects) > 0:
        for subject in subjects:
            shape = predict(gray, subject)  # Predict facial landmarks
            shape = face_utils.shape_to_np(shape)  # Convert the shape to a NumPy array
            
            # Extract the left and right eye coordinates from the facial landmarks
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            # Calculate the eye aspect ratio (EAR) for both eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            
            # Average the EAR for both eyes
            ear = (leftEAR + rightEAR) / 2.0
            
            # Convex hulls around the left and right eye to visualize in the frame
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            
            # Draw the convex hulls around the eyes on the frame
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # Check if the EAR is below the threshold, indicating potential drowsiness
            if ear < thresh:
                flag += 1  # Increment the frame counter
                
                if flag >= frame_check:
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "****************ALERT!****************", (10, 325),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if not mixer.music.get_busy():  # Play alert sound if not already playing
                        mixer.music.play()
                        alarm_counter += 1
                        print(flag)
                        print(alarm_counter)
                        # Make a call if the alarm has been triggered more than 3 times
                        if alarm_counter >= 3:
                            call = client.calls.create(
                                to="+",
                                from_="+1",
                                url="http://demo.twilio.com/docs/voice.xml"  # You can customize the URL with your own message
                            )
                            alarm_counter = 0
                            flag = 0  # Reset the alarm counter after making the call
        
            else:
                print(ear)
                flag = 0  # Reset the frame counter if EAR is above the threshold
    else:
        print("No faces detected.")

    # Display the frame
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
