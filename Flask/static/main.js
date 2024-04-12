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
