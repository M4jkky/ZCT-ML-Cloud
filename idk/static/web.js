const startPauseButton = document.getElementById('startPauseButton');

startPauseButton.addEventListener('click', function() {
    if (startPauseButton.innerText === 'START') {
        startPauseButton.innerText = 'PAUSE';
        startPauseButton.classList.add('started')
    } else {
        startPauseButton.innerText = 'START';
        startPauseButton.classList.remove('started');
    }
});

let language = 'sk'; // Default language
function toggleLanguage() {
    // Toggle language between English and Slovak
    if (language === 'en') {
        language = 'sk';
    } else {
        language = 'en';
    }
    // Update button text based on the selected language
    document.getElementById("languageButton").textContent = "Change Language (" + (language === "en" ? "English" : "Slovak") + ")";
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
