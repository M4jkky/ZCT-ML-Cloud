function hideEllips() {
    var ellipses = document.querySelectorAll('.ellipse-1, .ellipse-2, .ellipse-3, .ellipse-4, .ellipse-5, .ellipse-6, .ellipse-7');
    ellipses.forEach(function(ellipse) {
        ellipse.classList.add('hide-ellipse');
    });
}


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
