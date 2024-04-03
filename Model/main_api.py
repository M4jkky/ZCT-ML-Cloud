import mediapipe as mp
import numpy as np
import urllib.request
import json
import cv2
import ssl
import os


def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


allowSelfSignedHttps(True)  # this line is needed if you use self-signed certificate in your scoring service.

model_name = 'final-model-3-11'
url = 'https://zct-zadanie-2-rchbb.westeurope.inference.ml.azure.com/score'
api_key = '6SUlh5UQRVnniyv0SxhI4yNRJ4FINyyK'  # Primárny / sekundárny token
headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key),
           'azureml-model-deployment': model_name}


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam is not available or not working properly.")
    exit(1)

try:
    with open('misc/labels.json') as f:
        labels_dict = json.load(f)
except FileNotFoundError:
    print("JSON file not found. Please ensure the file path is correct.")
    exit(1)
except json.JSONDecodeError:
    print("JSON file is not properly formatted.")
    exit(1)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Inicializácia premenných na zapisovanie textu do súboru
previous_sentence = ""  # Kópia sentence, aby sa neplnila konzola za každým frame-om
sentence = ""  # Premenná, ktorá bude neskôr použitá na front-end predikciu
threshold = 22  # Počet frame-ov, ktoré musia prejsť, aby sa písmeno zapísalo do vety
hold_counter = 0  # Počítadlo, ktoré zabezpečuje, že sa nezmení písmeno príliš rýchlo
no_hand_counter = 0  # Počítadlo, ktoré zapisuje do súboru, neskôr na výslednej stránke ak nebude detekovaná ruka nastane text-to-speech
predicted_character_counter = 0  # Počítadlo, ktoré zabezpečuje, aby sa po určitej dobe na výslednej stránke zapísalo písmeno do vety
last_predicted_character = None


# Extrakcia bodov z ruky a vykreslenie bodov na obrazovku
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

    data_aux_2d = np.array(data_aux).reshape(1, -1)

    if x_:
        no_hand_counter = 0
        x1 = int(min(x_) * W) - 2
        y1 = int(min(y_) * H) - 2
        x2 = int(max(x_) * W) - 2
        y2 = int(max(y_) * H) - 2

        try:
            #  Request pre model na Azure
            data = {"data": data_aux_2d.tolist()}

            body = str.encode(json.dumps(data))  # Enkódovanie dát do JSON formátu

            req = urllib.request.Request(url, body, headers)  # Vytvorenie request-u

            try:
                response = urllib.request.urlopen(req)  # Poslanie request-u

                result = json.loads(response.read())
                prediction = result[0]
                predicted_character = labels_dict[str(int(prediction))]
                displayed_character = predicted_character  # aby sa nezobrazovalo „Space“ / „Delete“ v texte iba na kamere

                if predicted_character == "Space":
                    predicted_character = " "
                    displayed_character = "Space"

                if last_predicted_character != predicted_character:
                    predicted_character_counter = 0

                predicted_character_counter += 1
                hold_counter += 1

                if predicted_character == "Delete" and predicted_character_counter > threshold:
                    predicted_character = ""
                    if sentence.endswith("Hello"):
                        sentence = sentence[:-5]
                    elif sentence.endswith("ILY"):
                        sentence = sentence[:-3]
                    elif sentence.endswith("PEWPEW"):
                        sentence = sentence[:-6]
                    else:
                        sentence = sentence[:-1]
                    displayed_character = "Delete"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 1)
                if predicted_character != " ":
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2,
                                cv2.LINE_AA)
                else:
                    cv2.putText(frame, displayed_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2,
                                cv2.LINE_AA)

                if not sentence and predicted_character_counter > threshold:
                    sentence += predicted_character
                    predicted_character_counter = 0
                elif predicted_character_counter > threshold:
                    sentence += predicted_character
                    predicted_character_counter = 0

                if sentence != previous_sentence:
                    print(sentence)
                    previous_sentence = sentence

                if hold_counter > 10:
                    last_predicted_character = predicted_character
                    hold_counter = 0

                hold_counter += 1

            except urllib.error.HTTPError as error:
                print("The request failed with status code: " + str(error.code))
                print(error.info())
                print(error.read().decode("utf8", 'ignore'))

        except ValueError as e:
            print(f"An error occurred: {e}")
    else:
        no_hand_counter += 1

    if no_hand_counter >= 30 and sentence != "":
        sentence = ""
        no_hand_counter = 0

    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
