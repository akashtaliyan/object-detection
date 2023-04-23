import cv2
import mediapipe as mp
import pyautogui
from time import sleep

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpFace = mp.solutions.face_mesh
face = mpFace.FaceMesh()
mpDraw = mp.solutions.drawing_utils

while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Hand Detection
    results = hands.process(imageRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 20:
                    mpDraw.draw_landmarks(
                        image, handLms, mpHands.HAND_CONNECTIONS)

    # Face Detection
    resultFace = face.process(imageRGB)
    if (resultFace.multi_face_landmarks):
        pass
        for faceLms in resultFace.multi_face_landmarks:
            mpDraw.draw_landmarks(
                image, faceLms, mpFace.FACEMESH_TESSELATION)

        print("No Face Detected")
    cv2.imshow("Output", image)
    cv2.waitKey(1)
