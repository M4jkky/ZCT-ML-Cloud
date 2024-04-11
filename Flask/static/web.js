function hideEllips() {
    event.preventDefault();
    var ellipses = document.querySelectorAll('.ellipse-1, .ellipse-2, .ellipse-3, .ellipse-4, .ellipse-5, .ellipse-6, .ellipse-7');

    // Použijeme index na sledovanie aktuálneho času oneskorenia
    var index = 0;

    // Definujeme funkciu na skrytie prvku s oneskorením
    function hideEllipseWithDelay() {
        // Ak sme dosiahli koniec zoznamu prvkov, skončíme
        if (index >= ellipses.length) {

            window.location.href = "/login";
            return;
        }

        // Vyberieme aktuálny prvok
        var currentEllipse = ellipses[index];

        // Pridáme triedu hide-ellipse, aby sa prvok skryl
        currentEllipse.classList.add('hide-ellipse');

        // Zvýšime index pre ďalší prvok
        index++;

        // Zavoláme túto funkciu znova po oneskorení
        setTimeout(hideEllipseWithDelay, 100); // Nastavíme oneskorenie na 100ms (0.1 sekundy)
    }

    // Spustíme funkciu pre skrytie prvku s oneskorením
    hideEllipseWithDelay();
}

// Nájdeme tlačidlo "CONTINUE" podľa jeho ID
var continueButton = document.getElementById('continueButton');

// Pridáme event listener, ktorý zavolá funkciu hideEllips() po kliknutí na tlačidlo
continueButton.addEventListener('click', function() {
    hideEllips();
});