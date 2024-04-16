import os
import ssl
import cv2
import json
import pygame
import base64
import pickle
import psycopg2
import threading
import time
import numpy as np
import urllib.request
from PIL import Image
from gtts import gTTS
import mediapipe as mp
from io import BytesIO
from flask_socketio import SocketIO, emit
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, LoginManager, UserMixin, current_user
from flask import Flask, render_template, send_from_directory, Response, request, url_for, flash, redirect

class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

app = Flask(__name__, static_folder="./templates/static")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, async_mode="eventlet")

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

previous_sentence = ""
sentence = ""
threshold = 22
hold_counter = 0
no_hand_counter = 0
predicted_character_counter = 0
last_predicted_character = None

try:
    with open('misc/labels.json') as f:
        labels_dict = json.load(f)
except FileNotFoundError:
    print("JSON file not found. Please ensure the file path is correct.")
    exit(1)
except json.JSONDecodeError:
    print("JSON file is not properly formatted.")
    exit(1)

# Load the model from pickle file
try:
    model_dict = pickle.load(open('misc/model.pkl', 'rb'))
    model = model_dict['model']
except FileNotFoundError:
    print("Model file not found. Please ensure the file path is correct.")
    exit(1)

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


@app.route('/')
def landing():
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
    return redirect(url_for('landing'))


@app.route('/main')
@login_required
def main_web():
    return render_template('main.html', user=current_user)

def extract_hand_landmarks(frame):
    data_aux = []
    x_ = []
    y_ = []

    frame = np.array(frame)
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

def process_image(input_img):
    # Initialize necessary variables
    no_hand_counter = 0
    predicted_character_counter = 0
    hold_counter = 0
    last_predicted_character = ""
    sentence = ""
    previous_sentence = ""
    threshold = 10  # Adjust threshold as needed

    input_img = np.asarray(input_img)

    # Extract hand landmarks from the input image
    data_aux, x_, y_, H, W = extract_hand_landmarks(input_img)

    if x_:
        x1 = int(min(x_) * W) - 2
        y1 = int(min(y_) * H) - 2
        x2 = int(max(x_) * W) - 2
        y2 = int(max(y_) * H) - 2

        try:
            data_aux_2d = np.array(data_aux).reshape(1, -1)
            # Perform prediction using your model
            # Replace model.predict with your actual prediction code
            prediction = model.predict(data_aux_2d)
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
                # Update sentence based on the prediction
                # Adjust this part according to your actual logic
                if sentence.endswith("Hello"):
                    sentence = sentence[:-5]
                elif sentence.endswith("ILY"):
                    sentence = sentence[:-3]
                elif sentence.endswith("PEWPEW"):
                    sentence = sentence[:-6]
                else:
                    sentence = sentence[:-1]
                displayed_character = "Delete"

            cv2.rectangle(input_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
            if predicted_character != " ":
                cv2.putText(input_img, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2,
                            cv2.LINE_AA)
            else:
                cv2.putText(input_img, displayed_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2,
                            cv2.LINE_AA)

            if not sentence and predicted_character_counter > threshold and displayed_character != "Delete":
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

        except ValueError as e:
            print(f"An error occurred: {e}")
    else:
        no_hand_counter += 1

    if no_hand_counter >= 30 and sentence != "":
        sentence = ""
        no_hand_counter = 0

    return input_img

from flask import render_template_string
import base64

@app.route('/index', methods=['GET', 'POST'])
#@login_required
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded', 400
        file = request.files['file']
        if file.filename == '':
            return 'No file selected', 400
        if file:
            input_image = Image.open(file.stream)
            output_image = process_image(input_image)
            img_io = BytesIO()
            output_image.save(img_io, 'PNG')
            img_io.seek(0)
            img_data = base64.b64encode(img_io.getvalue()).decode('ascii')
            img_data = "data:image/png;base64," + img_data
            return render_template('index.html', img_data=img_data)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(app, debug=True, port=80, host='0.0.0.0')
