from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import time

# Initialize the sound system
mixer.init()
mixer.music.load('music.wav')  # Normal alert sound
special_alert = 'alert.mp3'  # Special alert sound for 1-minute eyes closed


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Threshold and frame check
thresh = 0.25
frame_check = 20

# Dlib's face detector and landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:\\Users\\Hp\\shape_predictor_68_face_landmarks.dat")

# Get left and right eye indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Initialize video capture
cap = cv2.VideoCapture(0)
flag = 0
closed_start_time = None  # To track the time when eyes are closed
special_alert_triggered = False  # To track if special alert is triggered

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

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
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "------DON'T SLEEP-----------", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "------BE ALERT!---------------", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()

                # Check if eyes have been closed for more than 1 minute
                if closed_start_time is None:
                    closed_start_time = time.time()  # Record when eyes were first closed
                else:
                    elapsed_time = time.time() - closed_start_time
                    if elapsed_time > 10 and not special_alert_triggered:  # 1 minute
                        # Trigger special alert
                        mixer.music.load(special_alert)  # Load special alert sound
                        mixer.music.play()
                        special_alert_triggered = True
                        print("Special Indication: Eyes closed for more than 1 minute!")
        else:
            flag = 0
            closed_start_time = None  # Reset the closed eye timer
            special_alert_triggered = False  # Reset the special alert

    # Show the frame
    cv2.imshow("Drowsiness Detector", frame)

    # Quit the program if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()