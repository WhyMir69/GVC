face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# label_map = {}
# trained = False

# # MediaPipe for hand tracking
# mp_hands = mp.solutions.hands
# hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# # Training
# # def prepare_training_data(dataset_path):
# #     global label_map
# #     faces = []
# #     labels = []
# #     label_id = 0
# #     label_map = {}

# #     for person_name in os.listdir(dataset_path):
# #         person_path = os.path.join(dataset_path, person_name)
# #         if not os.path.isdir(person_path):
# #             continue

# #         label_map[label_id] = person_name
# #         for img_name in os.listdir(person_path):
# #             img_path = os.path.join(person_path, img_name)
# #             img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# #             if img is None:
# #                 continue
# #             faces.append(img)
# #             labels.append(label_id)
# #         label_id += 1

# #     return faces, np.array(labels)

# # def train_model():
# #     global trained
# #     faces, labels = prepare_training_data(DATASET_PATH)
# #     if len(faces) > 0:
# #         face_recognizer.train(faces, labels)
# #         trained = True
# #         print(f"[INFO] Model trained with {len(faces)} face samples.")
# #     else:
# #         trained = False
# #         print("[WARN] No faces found. Model not trained.")




# # def start_recognition(mode):
# #     global detection_mode
# #     detection_mode = mode
# #     Thread(target=run_combined_recognition).start()


# # Combined recognition with output of hand and face data points (only x, y)
# # def run_combined_recognition():
# #     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# #     font = cv2.FONT_HERSHEY_SIMPLEX
# #     known_color = (0, 255, 0)
# #     unknown_color = (0, 0, 255)

# #     while cap.isOpened():
# #         success, frame = cap.read()
# #         if not success:
# #             break

# #         frame = cv2.flip(frame, 1)
# #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# #         # Detect faces
# #         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# #         if detection_mode == "Single" and len(faces) > 0:
# #             faces = [faces[0]]

# #         for (x, y, w, h) in faces:
# #             roi = gray[y:y+h, x:x+w]
# #             name = "Unknown"
# #             confidence_text = ""
# #             color = unknown_color

# #             if trained:
# #                 try:
# #                     label, confidence = face_recognizer.predict(roi)
# #                     confidence_percent = max(0, 100 - confidence)
# #                     if confidence < 60:
# #                         name = label_map.get(label, "Unknown")
# #                         color = known_color
# #                         confidence_text = f"{confidence_percent:.1f}%"
# #                     else:
# #                         confidence_text = f"{confidence_percent:.1f}% (low)"
# #                 except Exception as e:
# #                     confidence_text = str(e)

# #             cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
# #             cv2.putText(frame, name, (x, y-30), font, 0.8, color, 2)
# #             cv2.putText(frame, confidence_text, (x, y-10), font, 0.7, color, 2)

# #             # Print only x, y of face bounding box
# #             print(f"Face Bounding Box: x={x}, y={y}")

# #         # Detect hands
# #         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         results = hands_detector.process(rgb)

# #         if results.multi_hand_landmarks:
# #             for hand_landmarks in results.multi_hand_landmarks:
# #                 # Print x, y coordinates for each hand landmark
# #                 for idx, landmark in enumerate(hand_landmarks.landmark):
# #                     print(f"Hand Landmark {idx}: x={landmark.x}, y={landmark.y}")

# #                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# #         cv2.imshow("Face + Hand Recognition", frame)
# #         if cv2.waitKey(1) & 0xFF == ord('q'):
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()


# def record_face_and_hand():
#     name = simpledialog.askstring("Input", "Enter person's name:")
#     if not name:
#         return

#     face_dir = os.path.join(DATASET_PATH, name)
#     hand_dir = os.path.join(HANDSET_PATH, name)
#     os.makedirs(face_dir, exist_ok=True)
#     os.makedirs(hand_dir, exist_ok=True)

#     cap = cv2.VideoCapture(0)
#     count = 0
#     max_images = 20

#     while count < max_images:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.flip(frame, 1)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#         results = hands_detector.process(rgb)

#         if len(faces) > 0 and results.multi_hand_landmarks:
#             (x, y, w, h) = faces[0]
#             face_img = gray[y:y+h, x:x+w]
#             face_path = os.path.join(face_dir, f'{count}.jpg')
#             cv2.imwrite(face_path, face_img)

#             for hand_landmarks in results.multi_hand_landmarks:
#                 h_img, w_img, _ = frame.shape
#                 x_coords = [lm.x * w_img for lm in hand_landmarks.landmark]
#                 y_coords = [lm.y * h_img for lm in hand_landmarks.landmark]
#                 x_min, x_max = int(min(x_coords)), int(max(x_coords))
#                 y_min, y_max = int(min(y_coords)), int(max(y_coords))

#                 padding = 20
#                 x_min = max(x_min - padding, 0)
#                 y_min = max(y_min - padding, 0)
#                 x_max = min(x_max + padding, w_img)
#                 y_max = min(y_max + padding, h_img)

#                 hand_crop = frame[y_min:y_max, x_min:x_max]
#                 gray_hand = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2GRAY)
#                 hand_path = os.path.join(hand_dir, f'{count}.jpg')
#                 cv2.imwrite(hand_path, gray_hand)

#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # Print x, y coordinates of hand landmarks
#                 for idx, landmark in enumerate(hand_landmarks.landmark):
#                     print(f"Hand Landmark {idx}: x={landmark.x}, y={landmark.y}")

#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 100, 0), 2)
#             cv2.putText(frame, f"Captured: {count+1}/{max_images}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#             count += 1

#         cv2.imshow("Recording Face and Hand", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     # train_model()

# # Train model initially
# # train_model()

# # UI
# app = tk.Tk()
# app.title("Face and Hand Recognition")
# app.geometry("300x300")

# tk.Label(app, text="Choose Mode", font=("Arial", 14)).pack(pady=20)
# # tk.Button(app, text="Start Combined Recognition", width=25, command=lambda: start_recognition("Multiple")).pack(pady=10)
# tk.Button(app, text="Collect Data", width=25, command=record_face_and_hand).pack(pady=10)

# app.mainloop()
