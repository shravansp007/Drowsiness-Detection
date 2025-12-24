"""from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")# Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap=cv2.VideoCapture(0)
flag=0
while True:
	ret, frame=cap.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)
	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)#converting to NumPy Array
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
			print (flag)
			if flag >= frame_check:
				cv2.putText(frame, "****************ALERT!****************", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (10,325),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				#print ("Drowsy")
		else:
			flag = 0
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
cv2.destroyAllWindows()
cap.release() 
"""

# drowsiness_mediapipe.py
import cv2
import time
import numpy as np
import mediapipe as mp
import threading
import winsound  # Windows only; if not on Windows remove beep and use print/visual alert

# EAR function (same formula as dlib-based approach)
def eye_aspect_ratio(eye):
    # eye is a list/array of 6 (x, y) points in order:
    # p1 (left), p2 (upper-left), p3 (upper-right), p4 (right), p5 (lower-right), p6 (lower-left)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

# Alarm thread function
def sound_alarm():
    # plays a beep repeatedly while called (non-blocking caller should manage when to stop)
    for _ in range(5):
        winsound.Beep(2500, 300)  # frequency, duration (ms)
        time.sleep(0.1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Landmark indices for left and right eyes from MediaPipe Face Mesh
# These are chosen to approximate the 6 points used in EAR formula
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22   # threshold for closed eyes (tweak if needed)
CONSEC_FRAMES = 20     # number of consecutive frames EAR below threshold to trigger alarm

cap = cv2.VideoCapture(0)
time.sleep(1.0)

COUNTER = 0
ALARM_ON = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    # convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Collect landmarks into numpy array in pixel coordinates
            lm = face_landmarks.landmark
            pts = np.array([(int(p.x * w), int(p.y * h)) for p in lm])

            # Extract eye points
            left_eye_pts = np.array([pts[i] for i in LEFT_EYE_IDX], dtype=np.float32)
            right_eye_pts = np.array([pts[i] for i in RIGHT_EYE_IDX], dtype=np.float32)

            leftEAR = eye_aspect_ratio(left_eye_pts)
            rightEAR = eye_aspect_ratio(right_eye_pts)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw contours for visualization
            cv2.polylines(frame, [left_eye_pts.astype(np.int32)], True, (0,255,0), 1)
            cv2.polylines(frame, [right_eye_pts.astype(np.int32)], True, (0,255,0), 1)

            cv2.putText(frame, f"EAR: {ear:.3f}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        t = threading.Thread(target=sound_alarm, daemon=True)
                        t.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 3)
            else:
                COUNTER = 0
                ALARM_ON = False

    cv2.imshow("Drowsiness Detection (MediaPipe)", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
