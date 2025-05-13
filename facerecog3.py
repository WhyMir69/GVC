import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import simpledialog
from threading import Thread
import mediapipe as mp

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "faces")
HANDSET_PATH = os.path.join(BASE_DIR, "hands")

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(HANDSET_PATH, exist_ok=True)

# Globals
detection_mode = "Multiple"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
label_map = {}
trained = False




# MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils



#---------------------------------------- for collecting data ----------------------------------------
def record_face_and_hand():
    hand_count = 0

    coordinates_movemenet = []
    coordinates_frame = []

    name = simpledialog.askstring("Input", "Enter person's name:")
    if not name:
        return

    face_dir = os.path.join(DATASET_PATH, name)
    hand_dir = os.path.join(HANDSET_PATH, name)
    

    coord_dir = os.path.join("coordinates", name) 

    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(hand_dir, exist_ok=True)
    os.makedirs(coord_dir, exist_ok=True)  
   

    cap = cv2.VideoCapture(0)
    count = 0
    max_images = 10

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        results = hands_detector.process(rgb)

        if len(faces) > 0 and results.multi_hand_landmarks:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_path = os.path.join(face_dir, f'{count}.jpg')
            cv2.imwrite(face_path, face_img)

            for hand_landmarks in results.multi_hand_landmarks:
                hand_count +=1
                h_img, w_img, _ = frame.shape
                x_coords = [lm.x * w_img for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h_img for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))

                padding = 20
                x_min = max(x_min - padding, 0)
                y_min = max(y_min - padding, 0)
                x_max = min(x_max + padding, w_img)
                y_max = min(y_max + padding, h_img)

                hand_crop = frame[y_min:y_max, x_min:x_max]
                gray_hand = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
                hand_path = os.path.join(hand_dir, f'{count}.jpg')
                cv2.imwrite(hand_path, gray_hand)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
              

                for idx, landmark in enumerate(hand_landmarks.landmark):
                    coordinates_frame.append(f"{landmark.x:.3f}")
                    coordinates_frame.append(f"{landmark.y:.3f}")
        
                coordinates_movemenet.append(coordinates_frame)
                coordinates_frame=[]   
            print("hand_count -->",hand_count)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 2)
            cv2.putText(frame, f"Captured: {count+1}/{max_images}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            count += 1
  
        print("coordinates_movemenet-->",coordinates_movemenet)
        cv2.imshow("Recording Face and Hand", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
  





def text_encode(text,coord_dir):
    num = 0
    coordinates_frame = []
    coord_file_path = os.path.join(coord_dir, "hand_landmarks.txt")
    coord_file = open(coord_file_path, "w")

    for idx, landmark in enumerate(text):
        coordinates_frame.append(landmark.x)
        coordinates_frame.append(landmark.y)
        num = num + 1
        coord_file.write(f"{landmark.x:.4f}|{landmark.y:.4f}|")
        coord_file.write(f"{num}")

    coord_file.close()




app = tk.Tk()
app.title("Face and Hand Recognition")
app.geometry("300x300")

tk.Label(app, text="Choose Mode", font=("Arial", 14)).pack(pady=20)
# tk.Button(app, text="Start Combined Recognition", width=25, command=lambda: start_recognition("Multiple")).pack(pady=10)
tk.Button(app, text="Collect Data", width=25, command=record_face_and_hand).pack(pady=10)

app.mainloop()
