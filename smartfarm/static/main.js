
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


document.addEventListener('DOMContentLoaded', function() {
    const cultSelects = ['cultSelect', 'cult_id'];

    cultSelects.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', function() {
                localStorage.setItem('selectedCultId', this.value);
            });
        }
    });

    const saved = localStorage.getItem('selectedCultId');
    if (saved) {
        cultSelects.forEach(id => {
            const el = document.getElementById(id);
            if (el && [...el.options].some(o => o.value === saved)) {
                el.value = saved;
                if (id === 'cultSelect') {
                    onCultChange && onCultChange(saved);
                }
            }
        });
    }
});