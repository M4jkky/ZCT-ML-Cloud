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

const languageButton = document.getElementById('languageButton');

languageButton.addEventListener('click', function(){
    if(languageButton.innerText==='Slovensky'){
        languageButton.innerText = 'English';
        languageButton.classList.add('slovensky');
    } else {
        languageButton.innerText = 'Slovensky';
        languageButton.classList.remove('slovensky')
    }
});
