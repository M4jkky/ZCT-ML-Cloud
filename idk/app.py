import os
import ssl
import cv2
import json
import pickle
import psycopg2
import numpy as np
import urllib.request
from PIL import Image
import mediapipe as mp
from flask_socketio import SocketIO
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, LoginManager, UserMixin, current_user
from flask import Flask, render_template,request, url_for, redirect
import base64

# Initialize Flask app
app = Flask(__name__, static_folder="./templates/static")
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Azure ML model deployment details
model_name = 'final-model-3-11'
url = 'https://zct-zadanie-2-rchbb.westeurope.inference.ml.azure.com/score'
api_key = '0tqsDiQ1ygBfw88ocgsXO3ebUTYcVEbD'
headers = {'Content-Type': 'application/json', 'Authorization': ('Bearer ' + api_key), 'azureml-model-deployment': model_name}


class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# Load labels dictionary
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

# AWS database connection details
def connect_to_db():
    conn = psycopg2.connect(
        host="users-zct2.czsaiqieednr.eu-north-1.rds.amazonaws.com",
        database="initial_db",
        user="postgres",
        password="zct2_posunky_stranky_aws_azure_ouje_posrche911_databaza_dobry-den",
        port="5432"
    )
    return conn

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id):
    conn = connect_to_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user_data = cur.fetchone()
    cur.close()
    if user_data:
        return User(id=user_data[0], username=user_data[1], password=user_data[2])
    else:
        return None

# Helper function to allow self-signed HTTPS
def allowSelfSignedHttps(allowed):
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

# Landing page route
@app.route('/')
def landing():
    return render_template('landing_page.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    conn = connect_to_db()
    cur = conn.cursor()

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        cur.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cur.fetchone()

        if user and check_password_hash(user[2], password):
            print("Password is correct")
            user_obj = User(id=user[0], username=user[1], password=user[2])
            login_user(user_obj, remember=True)
            return redirect(url_for('main_web'))
        else:
            print("Invalid username or password")

        cur.close()

    return render_template("login.html", user=current_user)

# Signup route
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
            print("Username or email already exists.")
        else:
            hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
            cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
            conn.commit()
            print("Account created!")

            # Fetch the user data from the database
            cur.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cur.fetchone()

            # Create a User object
            user_obj = User(id=user[0], username=user[1], password=user[2])

            login_user(user_obj, remember=True)

            return redirect(url_for('main_web'))

        cur.close()

    return render_template('sign_up.html', user=current_user)

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('landing'))

# Main page route
@app.route('/main')
@login_required
def main_web():
    return render_template('main.html', user=current_user)

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

def process_image1(input_img):
    data_aux, x_, y_, H, W = extract_hand_landmarks(input_img)
    data_aux_2d = np.array(data_aux).reshape(1, -1)

    if x_:
        x1 = int(min(x_) * W) - 2
        y1 = int(min(y_) * H) - 2
        x2 = int(max(x_) * W) - 2
        y2 = int(max(y_) * H) - 2

        try:
            data = {"data": data_aux_2d.tolist()}
            body = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(url, body, headers)

            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read())
                prediction = result[0]
                predicted_character = labels_dict.get(str(int(prediction)), '')
                displayed_character = predicted_character

            cv2.rectangle(input_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
            cv2.putText(input_img, displayed_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0),
                        2, cv2.LINE_AA)

        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))
            print(error.info())
            print(error.read().decode("utf8", 'ignore'))

    return input_img

@app.route('/index')
def index():
    return render_template('index.html', img_data=None)


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    # Read image as PIL image
    pil_img = Image.open(file)

    # Convert PIL image to OpenCV format (numpy array)
    opencv_img = np.array(pil_img)
    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)

    processed_img = process_image1(opencv_img)

    # Convert NumPy array to base64 string
    _, img_encoded = cv2.imencode('.jpg', processed_img)
    img_str = base64.b64encode(img_encoded).decode()

    return render_template('index.html', img_data="data:image/jpeg;base64," + img_str)


if __name__ == "__main__":
    app.run(debug=True, port=80, host='0.0.0.0')
