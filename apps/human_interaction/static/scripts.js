function type_text(elementId, text, hold_duration = 3000) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    const bubble = element.parentElement;
    bubble.classList.add('visible');
    
    let i = 0;
    element.innerHTML = '';
    const typing = setInterval(() => {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
        } else {
            clearInterval(typing);
            setTimeout(() => {
                bubble.classList.remove('visible');
            }, hold_duration);
        }
    }, 30);
}

function initCardHoverEffects() {
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        if (card.dataset.hoverInitialized) return;
        card.dataset.hoverInitialized = 'true';

        card.addEventListener('mousemove', (e) => {
            const container = card.closest('.card-container');
            const isHovered = container.matches(':hover');
            
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = (y - centerY) / 10;
            const rotateY = (centerX - x) / 10;
            
            // hover 시 scale 추가
            const scaleValue = isHovered ? 'scale(1.15)' : '';
            card.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg) ${scaleValue}`;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = 'rotateX(0) rotateY(0)';
        });
    });
}

// Add this function for the shake effect
function shake_card(playerId) {
    const card = document.getElementById(`player-card-${playerId}`);
    if (card) {
        card.classList.add('shake');
        setTimeout(() => {
            card.classList.remove('shake');
        }, 500); // Duration of the shake animation
    }
}

function set_theme(theme) {
    const bg = document.getElementById('background-div');
    if (!bg) return;

    if (theme === 'night') {
        bg.classList.remove('theme-day');
        bg.classList.add('theme-night');
    } else {
        bg.classList.remove('theme-night');
        bg.classList.add('theme-day');
    }
}

