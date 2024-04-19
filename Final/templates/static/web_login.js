// Definujeme funkciu hideEllips s oneskorením animácie
function hideEllips(event) {
    // Zastavíme predvolené správanie kliknutia na tlačidlo (prechod na inú stránku)
    event.preventDefault();

    // Vyberieme všetky elementy s triedami .ellipse-1 až .ellipse45
    var ellipses_right = document.querySelectorAll('.ellipse28, .ellipse29, .ellipse30, .ellipse31, .ellipse32, .ellipse33, .ellipse34,  .ellipse39, .ellipse45');
    var ellipses_left = document.querySelectorAll('.ellipse35, .ellipse36, .ellipse37, .ellipse38,.ellipse40, .ellipse41, .ellipse42, .ellipse43, .ellipse44')

    // Použijeme index na sledovanie aktuálneho času oneskorenia
    var index = 0;

    // Definujeme funkciu na skrytie prvku s oneskorením
    function hideEllipseWithDelay() {
        // Ak sme dosiahli koniec zoznamu prvkov vpravo aj vľavo, skončíme
        if (index >= ellipses_right.length && index >= ellipses_left.length) {
            // Akonáhle je animácia dokončená, prejdeme na tretiu stránku
            window.location.href = "/main";
            return;
        }

        // Vyberieme aktuálny prvok vpravo a vľavo
        var currentEllipseRight = ellipses_right[index];
        var currentEllipseLeft = ellipses_left[index];

        // Pridáme triedu hide-ellipse, aby sa prvok skryl vpravo aj vľavo
        currentEllipseRight.classList.add('hide-ellipse-right');
        currentEllipseLeft.classList.add('hide-ellipse-left');

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
continueButton.addEventListener('click', function(event) {
    hideEllips(event);
});