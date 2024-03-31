import cv2
import json
import mediapipe as mp
import numpy as np
import pickle
import threading

from flask import Flask, Response, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

hands_detector = mp.solutions.hands.Hands(
    static_image_mode=True, min_detection_confidence=0.3)

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

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

    return data_aux, x_, y_, h, w


def main():
    global latest_frame
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Webcam is not available or not working properly.")
        exit(1)

    model, labels_dict = load_model_and_labels()

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
                socketio.emit('character', {'value': predicted_character})
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 80, 0), 1)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2,
                            cv2.LINE_AA)
            except ValueError as e:
                print(f"An error occurred: {e}")

        latest_frame = frame

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()
    main()
