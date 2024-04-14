var socket = io.connect(window.location.protocol + '//' + document.domain + ':' + location.port, {
    transports: ['websocket']
});
socket.on('connect', function () {
    console.log("Connected...!", socket.connected)
});


var canvas = document.getElementById('canvas');
var context = canvas.getContext('2d');
const video = document.querySelector("#videoElement");

video.width = 400;
video.height = 300;


if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({
        video: true
    })
        .then(function (stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function (err0r) {

        });
}

const FPS = 10;
setInterval(() => {
    width = video.width;
    height = video.height;
    context.drawImage(video, 0, 0, width, height);
    var data = canvas.toDataURL('image/jpeg', 0.5);
    context.clearRect(0, 0, width, height);
    socket.emit('image', data);
}, 1000 / FPS);

socket.on('processed_image', function (image) {
    photo.setAttribute('src', image);

});

const startPauseButton = document.getElementById('startPauseButton');

// Set the initial text content of the start button to "Start"
startPauseButton.textContent = 'START';

// Function to toggle the text of the start button between "Start" and "Pause"
function toggleStartButton() {
    if (startPauseButton.textContent === 'START') {
        startPauseButton.textContent = 'PAUSE';
    } else {
        startPauseButton.textContent = 'START';
    }

    // Send a request to your Flask backend with the updated status
    fetch('/toggle', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status: startPauseButton.textContent })
    })
    .then(response => response.json())
    .then(data => {
        // Handle response if needed
        if (data.status === 'PAUSE') {
            startPauseButton.classList.add('started');
        } else {
            startPauseButton.classList.remove('started');
        }
    })
    .catch(error => console.error('Error:', error));
}

startPauseButton.addEventListener('click', toggleStartButton);

let language = 'sk'; // Default language

function toggleLanguage() {
    // Toggle language between English and Slovak
    if (language === 'en') {
        language = 'sk';
    } else {
        language = 'en';
    }
    // Update button text based on the selected language
    document.getElementById("LanguageButton").textContent = (language === "en" ? "English" : "Slovak");
    // Send a request to your Flask backend with the selected language
    fetch('/start_process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ language: language }), // Pass the selected language
    })
    .then(response => {
        // Handle response if needed
        console.log(response);
    })
    .catch(error => {
        // Handle errors if any
        console.error('Error:', error);
    });
}