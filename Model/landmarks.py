import mediapipe as mp
import pickle
import cv2
import os

DATA_DIR = '../new_data'
PICKLE_FILE = 'misc/data.pickle'
MP_HANDS = mp.solutions.hands
MP_DRAWING = mp.solutions.drawing_utils
MP_DRAWING_STYLES = mp.solutions.drawing_styles

# Inicializácia modelu na detekciu rúk
hands = MP_HANDS.Hands(static_image_mode=True, min_detection_confidence=0.3)


def process_image(img_path):
    """Spracovanie obrázku a extrakcia landmarkov rúk."""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    data_aux = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_, y_ = [], []
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
    return data_aux


def main():
    data = []
    labels = []
    for dir_ in os.listdir(DATA_DIR):
        full_path = os.path.join(DATA_DIR, dir_)
        if os.path.isdir(full_path):
            for img_path in os.listdir(full_path):
                data_aux = process_image(os.path.join(full_path, img_path))
                if data_aux:
                    data.append(data_aux)
                    labels.append(dir_)
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)


if __name__ == "__main__":
    main()
