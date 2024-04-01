import cv2
import json
import mediapipe as mp
import numpy as np
import pickle
import threading
from flask import Flask, Response, render_template
from flask_socketio import SocketIO
from gtts import gTTS
from io import BytesIO
import pygame

app = Flask(__name__)
socketio = SocketIO(app)
hands_detector = mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

file = open("predicted_sentences.txt", "a")

def speak(text):
    tts = gTTS(text=text, lang="sk")
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
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def load_model_and_labels():
    try:
        with open('misc/model.pkl', 'rb') as f:
            model_dict = pickle.load(f)
            model = model_dict['model']
    except FileNotFoundError:
        print("Model file not found. Please ensure the file path is correct.")
        exit(1)

    try:
        with open('misc/labels.json') as f:
            labels_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Labels file not found or improperly formatted. Please ensure the file path is correct.")
        exit(1)

    return model, labels_dict

def extract_hand_landmarks(frame):
    data_aux = []
    x_ = []
    y_ = []
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i, landmark in enumerate(hand_landmarks.landmark):
                x = landmark.x
                y = landmark.y
                x_.append(x)
                y_.append(y)
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
            mp.solutions.drawing_utils.draw_landmarks(frame,
                                                      hand_landmarks,
                                                      mp.solutions.hands.HAND_CONNECTIONS,
                                                      mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                                      mp.solutions.drawing_styles.get_default_hand_connections_style())
    return data_aux, x_, y_, h, w

def main():
    global latest_frame
    previous_sentence = ""
    sentence = ""
    hold_counter = 0
    no_hand_counter = 0
    predicted_character_counter = 0
    last_predicted_character = None
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam is not available or not working properly.")
        exit(1)

    model, labels_dict = load_model_and_labels()
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
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[str(int(prediction[0]))]
                displayed_character = predicted_character

                if predicted_character == "Space":
                    predicted_character = " "
                    displayed_character = "Space"

                if last_predicted_character != predicted_character:
                    predicted_character_counter = 0

                predicted_character_counter += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 0), 1)
                if predicted_character != " ":
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
                    socketio.emit('character', {'value': predicted_character})
                else:
                    cv2.putText(frame, displayed_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
                    socketio.emit('character', {'value': displayed_character})

                if not sentence and predicted_character_counter > 20:
                    sentence += predicted_character
                    predicted_character_counter = 0
                elif predicted_character_counter > 20:
                    sentence += predicted_character
                    predicted_character_counter = 0

                if sentence != previous_sentence:
                    sentence = sentence.lower()
                    print(sentence)
                    socketio.emit('sentence', {'value': sentence})
                    previous_sentence = sentence

                if hold_counter > 7:
                    last_predicted_character = predicted_character
                    hold_counter = 0

                hold_counter += 1

            except ValueError as e:
                print(f"An error occurred: {e}")
        else:
            no_hand_counter += 1

        if no_hand_counter >= 40 and sentence != "":
            file.write(sentence + "\n")
            print("written to file")
            threading.Thread(target=speak, args=(sentence,)).start()
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
