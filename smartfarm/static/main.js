
function refreshIcons() {
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
    }
}


document.addEventListener('DOMContentLoaded', () => {
    refreshIcons();
    console.log("AI Smart Farm UI Initialized");
});


function toggleMenu() {
    const menu = document.getElementById('mobile-menu');
    const icon = document.getElementById('menu-icon');

    menu.classList.toggle('hidden');

    if (menu.classList.contains('hidden')) {
        icon.setAttribute('data-lucide', 'menu');
    } else {
        icon.setAttribute('data-lucide', 'x');
    }

    lucide.createIcons();
}