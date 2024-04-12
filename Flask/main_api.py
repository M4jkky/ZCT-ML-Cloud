import os
import cv2
import ssl
import json
import pygame
import psycopg2
import threading
import numpy as np
import urllib.request
import mediapipe as mp
from gtts import gTTS
from io import BytesIO
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, Response, render_template, request, url_for,flash, redirect
from flask_login import login_user, login_required, logout_user, LoginManager, UserMixin, current_user

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

app = Flask(__name__)
#secret key
app.secret_key = 'balamuta_toto_tu_o_2_v_noci_123'
socketio = SocketIO(app)

# login manager
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

language = "sk"
status = 'START'
latest_frame = None

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

model_name = 'final-model-3-11'
url = 'https://zct-zadanie-2-rchbb.westeurope.inference.ml.azure.com/score'
api_key = '0tqsDiQ1ygBfw88ocgsXO3ebUTYcVEbD'
headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key),
           'azureml-model-deployment': model_name}

# AWS database connection for sign up and login
def connect_to_db():
    conn = psycopg2.connect(
        host="users-zct2.czsaiqieednr.eu-north-1.rds.amazonaws.com",
        database="initial_db",
        user="postgres",
        password="zct2_posunky_stranky_aws_azure_ouje_posrche911_databaza_dobry-den",
        port="5432"
    )
    return conn

@login_manager.user_loader
def load_user(user_id):
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user_data = cur.fetchone()
    cur.close()
    if user_data:
        return User(id=user_data[0],username=user_data[1], password=user_data[2])
    else:
        return None

def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


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


@app.route('/login', methods=['GET', 'POST'])
def login():
    conn = connect_to_db()
    cur = conn.cursor()

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()

        if user:
            if check_password_hash(user[2], password):
                print("Password is correct")
                user_obj = User(id=user[0], username=user[1], password=user[2])
                login_user(user_obj, remember=True)
                return redirect(url_for('main_web'))
            else:
                print("Password is incorrect")
        else:
            print("User does not exist")

        cur.close()

    return render_template("login.html", user=current_user)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        conn = connect_to_db()
        cur = conn.cursor()

        # Check if username already exists
        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()
        if user:
            print("Username address or username already exists.")
            #flash('Username address or username already exists.', category='error')
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            print("Account created!")
            #flash('Account created!', category='success')

            # Fetch the user data from the database
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cur.fetchone()
            print(user)

            # Create a User object
            user_obj = User(id=user[0], username=user[1], password=user[2])

            login_user(user_obj, remember=True)

            return redirect(url_for('main_web'))

        cur.close()

    return render_template('sign_up.html', user=current_user)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

def process_video(video_data):
    # Convert the binary data to a numpy array
    video_data = np.frombuffer(video_data, np.uint8)

    # Load the data as a cv2.VideoCapture object
    video = cv2.imdecode(video_data, cv2.IMREAD_COLOR)

    # Process the video frame by frame
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Process the frame here
        extract_hand_landmarks(frame)

@app.route('/upload_video', methods=['POST'])
@login_required
def upload_video():
    video_file = request.files['video']  # get the video file from the request
    video_data = video_file.read()  # read the video data

    # Process the video
    process_video(video_data)

    return 'Video processed successfully'



@app.route('/start_process', methods=['POST'])
@login_required
def start_process():
    global language
    language = request.json.get('language')
    return 'Process started successfully'


@app.route('/main')
@login_required
def main_web():
    return render_template('main.html', user=current_user)


@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle', methods=['POST'])
@login_required
def toggle():
    global status
    status = request.json.get('status')
    if status == 'START':
        return {'status': 'PAUSE', 'message': 'Process toggled successfully'}
    else:
        return {'status': 'START', 'message': 'Process toggled successfully'}


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

    previous_sentence = ""
    sentence = ""
    threshold = 22
    hold_counter = 0
    no_hand_counter = 0
    predicted_character_counter = 0
    last_predicted_character = None

    allowSelfSignedHttps(True)
    cap = cv2.VideoCapture(0)

    try:
        with open('misc/labels.json') as f:
            labels_dict = json.load(f)
    except FileNotFoundError:
        print("JSON file not found. Please ensure the file path is correct.")
        exit(1)
    except json.JSONDecodeError:
        print("JSON file is not properly formatted.")
        exit(1)

    while True:
        ret, frame = cap.read()
        if status == 'START':
            data_aux, x_, y_, H, W = extract_hand_landmarks(frame)
            data_aux_2d = np.array(data_aux).reshape(1, -1)

            if x_:
                no_hand_counter = 0
                x1 = int(min(x_) * W) - 2
                y1 = int(min(y_) * H) - 2
                x2 = int(max(x_) * W) - 2
                y2 = int(max(y_) * H) - 2

                try:
                    data = {"data": data_aux_2d.tolist()}
                    body = str.encode(json.dumps(data))
                    req = urllib.request.Request(url, body, headers)

                    response = urllib.request.urlopen(req)
                    result = json.loads(response.read())
                    prediction = result[0]
                    predicted_character = labels_dict[str(int(prediction))]
                    displayed_character = predicted_character

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
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0),
                                    2, cv2.LINE_AA)
                        socketio.emit('character', {'value': predicted_character})
                    else:
                        cv2.putText(frame, displayed_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0),
                                    2, cv2.LINE_AA)
                        socketio.emit('character', {'value': displayed_character})

                    if not sentence and predicted_character_counter > threshold:
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

                    hold_counter += 1

                except urllib.error.HTTPError as error:
                    print("The request failed with status code: " + str(error.code))
                    print(error.info())
                    print(error.read().decode("utf8", 'ignore'))

            else:
                no_hand_counter += 1

            if no_hand_counter >= 30 and sentence != "":
                threading.Thread(target=speak, args=(sentence, language,)).start()
                sentence = ""
                no_hand_counter = 0

            latest_frame = frame
        else:
            latest_frame = frame


if __name__ == '__main__':
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}).start()
    main()
