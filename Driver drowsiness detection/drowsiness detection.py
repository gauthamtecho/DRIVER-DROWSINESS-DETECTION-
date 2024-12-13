import cv2
import dlib
from scipy.spatial import distance as dist
from pygame import mixer
import imutils
from imutils import face_utils

mixer.init()
alarm_sound = mixer.Sound('C:\\Users\\Gautham Reddy\\OneDrive\\Desktop\\Driver drowsiness detection\\alarm.wav')
def compute_ear(eye):
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h = dist.euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h)


EAR_THRESHOLD = 0.25
ALERT_FRAMES = 20
frames = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:\\Users\\Gautham Reddy\\OneDrive\\Desktop\\Driver drowsiness detection\\shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(imutils.resize(frame, width=640), cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = face_utils.shape_to_np(predictor(gray, face))
        left_eye = landmarks[lStart:lEnd]
        right_eye = landmarks[rStart:rEnd]

        ear = (compute_ear(left_eye) + compute_ear(right_eye)) / 2.0
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD:
            frames += 1
            if frames >= ALERT_FRAMES:
                if not mixer.music.get_busy():
                    alarm_sound.play()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            frames = 0
            mixer.music.stop()

        cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
mixer.quit()
