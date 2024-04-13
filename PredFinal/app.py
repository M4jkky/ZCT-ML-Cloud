import os
import ssl
import cv2
import json
import pygame
import base64
import psycopg2
import threading
import numpy as np
import urllib.request
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
                return redirect(url_for('index'))
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

            return redirect(url_for('index'))

        cur.close()

    return render_template('sign_up.html', user=current_user)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))


def base64_to_image(base64_string):
    """
    The base64_to_image function accepts a base64 encoded string and returns an image.
    The function extracts the base64 binary data from the input string, decodes it, converts 
    the bytes to numpy array, and then decodes the numpy array as an image using OpenCV.
    
    :param base64_string: Pass the base64 encoded image string to the function
    :return: An image
    """
    base64_data = base64_string.split(",")[1]
    image_bytes = base64.b64decode(base64_data)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


@socketio.on("connect")
def test_connect():
    """
    The test_connect function is used to test the connection between the client and server.
    It sends a message to the client letting it know that it has successfully connected.
    
    :return: A 'connected' string
    """
    print("Connected")
    emit("my response", {"data": "Connected"})




@socketio.on("image")
def receive_image(image):
    """
    The receive_image function takes in an image from the webcam, converts it to grayscale, and then emits
    the processed image back to the client.


    :param image: Pass the image data to the receive_image function
    :return: The image that was received from the client
    """
    # Decode the base64-encoded image data
    image = base64_to_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(gray, (640, 360))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame_encoded = cv2.imencode(".jpg", frame_resized, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()
    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data
    emit("processed_image", processed_img_data)


@app.route("/index") # toto potom treba prerobiť na našu main stránku, ale opatrne - pozor zitra
def index():
    """
    The index function returns the index.html template, which is a landing page for users.
    
    :return: The index
    """
    return render_template("index.html")


if __name__ == "__main__":
    socketio.run(app, debug=True, port=5000, host='0.0.0.0')
