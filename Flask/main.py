import mediapipe as mp
import numpy as np
import urllib.request
import json
import cv2
import ssl
import os
import threading
from flask import Flask, Response, render_template, request
from flask_socketio import SocketIO
from gtts import gTTS
from io import BytesIO
import pygame
import pickle


app = Flask(__name__)
socketio = SocketIO(app)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
language = "sk"
status = 'START'
latest_frame = None

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

def speak(text, language):
    tts = gTTS(text=text, lang=language)
    audio_stream = BytesIO()
    tts.write_to_fp(audio_stream)
    audio_stream.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(audio_stream)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()

def generate_frames():
    while True:
        if latest_frame is not None:
            ret, buffer = cv2.imencode('.jpg', latest_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('landing_page.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('sign_up.html')

@app.route('/start_process', methods=['POST'])
def start_process():
    global language
    language = request.json
    language = language.get('language')
    return 'Process started successfully'

@app.route('/main')
def main_web():
    """Video streaming home page."""
    return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle', methods=['POST'])
def toggle():
    global status
    print(status)
    status = request.json.get('status')
    if status == 'START':
        return {'status': 'PAUSE', 'message': 'Process toggled successfully'}
    else:
        return {'status': 'START', 'message': 'Process toggled successfully'}


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

def main():
    global latest_frame
    global language
    global status
    # Open the text file in append mode
    file = open("predicted_sentences.txt", "a")

    # Load the model from pickle file
    try:
        model_dict = pickle.load(open('./misc/model.pkl', 'rb'))
        model = model_dict['model']
    except FileNotFoundError:
        print("Model file not found. Please ensure the file path is correct.")
        exit(1)

    # Load the classes from JSON file
    try:
        with open('./misc/labels.json') as f:
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

    # Inicializácia premenných na zapisovanie textu do súboru
    previous_sentence = ""  # Kópia sentence, aby sa neplnila konzola za každým frame-om
    sentence = ""  # Premenná, ktorá bude neskôr použitá na front-end predikciu
    threshold = 22  # Počet frame-ov, ktoré musia prejsť, aby sa písmeno zapísalo do vety
    hold_counter = 0  # Počítadlo, ktoré zabezpečuje, že sa nezmení písmeno príliš rýchlo
    no_hand_counter = 0  # Počítadlo, ktoré zapisuje do súboru, neskôr na výslednej stránke ak nebude detekovaná ruka nastane text-to-speech
    predicted_character_counter = 0  # Počítadlo, ktoré zabezpečuje, aby sa po určitej dobe na výslednej stránke zapísalo písmeno do vety
    last_predicted_character = None
    while True:
        ret, frame = cap.read()
        data_aux, x_, y_, H, W = extract_hand_landmarks(frame)

        if x_:
            no_hand_counter = 0
            x1 = int(min(x_) * W) - 2
            y1 = int(min(y_) * H) - 2
            x2 = int(max(x_) * W) - 2
            y2 = int(max(y_) * H) - 2

            try:
                data_aux_2d = np.array(data_aux).reshape(1, -1)
                prediction = model.predict(data_aux_2d)
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
                    socketio.emit('character', {'value': predicted_character})
                else:
                    cv2.putText(frame, displayed_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2,
                                cv2.LINE_AA)
                    socketio.emit('character', {'value': displayed_character})

                if not sentence and predicted_character_counter > threshold and displayed_character != "Delete":
                    sentence += predicted_character
                    predicted_character_counter = 0
                elif predicted_character_counter > threshold:
                    sentence += predicted_character
                    predicted_character_counter = 0

                if sentence != previous_sentence:
                    sentence = sentence.lower()
                    socketio.emit('sentence', {'value': sentence})
                    print(sentence)
                    previous_sentence = sentence

                if hold_counter > 10:
                    last_predicted_character = predicted_character
                    hold_counter = 0

            except ValueError as e:
                print(f"An error occurred: {e}")
        else:
            no_hand_counter += 1

        if no_hand_counter >= 30 and sentence != "":
            threading.Thread(target=speak, args=(sentence,language,)).start()
            sentence = ""
            no_hand_counter = 0

        latest_frame = frame

        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()
    main()
