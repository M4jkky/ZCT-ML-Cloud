import mediapipe as mp
import numpy as np
import pickle
import json
import cv2


# Načítanie modelu z pickle súboru
try:
    model_dict = pickle.load(open('misc/model.pkl', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Model file not found. Please ensure the file path is correct.")
    exit(1)

# Načítanie tried z JSON súboru
try:
    with open('misc/labels.json') as f:
        labels_dict = json.load(f)
except FileNotFoundError:
    print("JSON file not found. Please ensure the file path is correct.")
    exit(1)
except json.JSONDecodeError:
    print("JSON file is not properly formatted.")
    exit(1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam is not available or not working properly.")
    exit(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def extract_hand_landmarks(frame):
    data_aux = []
    x_ = []
    y_ = []

    h, w, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

    return data_aux, x_, y_, h, w


while True:
    ret, frame = cap.read()

    data_aux, x_, y_, H, W = extract_hand_landmarks(frame)

    if x_:
        x1 = int(min(x_) * W) - 2
        y1 = int(min(y_) * H) - 2
        x2 = int(max(x_) * W) - 2
        y2 = int(max(y_) * H) - 2

        try:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[str(int(prediction[0]))]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 0), 1)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2,
                        cv2.LINE_AA)
        except ValueError as e:
            print(f"An error occurred: {e}")

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
