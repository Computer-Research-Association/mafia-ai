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

        // 카드 클릭 이벤트
        card.addEventListener('click', () => {
            const playerId = card.id.replace('player-card-', '');
            showCardDetail(playerId);
        });
    });
}

let currentDetailPlayerId = null;

function showCardDetail(playerId) {
    // 이미 열려있는 카드를 다시 클릭하면 닫기
    if (currentDetailPlayerId === playerId) {
        closeCardDetail();
        return;
    }

    // 오버레이가 없으면 생성
    let overlay = document.getElementById('card-detail-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'card-detail-overlay';
        document.body.appendChild(overlay);

        // 카드 컨테이너
        const cardContainer = document.createElement('div');
        cardContainer.className = 'detail-card-container';
        cardContainer.onclick = () => closeCardDetail();
        overlay.appendChild(cardContainer);

        // 정보 패널
        const infoPanel = document.createElement('div');
        infoPanel.className = 'detail-info-panel';
        overlay.appendChild(infoPanel);

        // 배경 클릭으로 닫기
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                closeCardDetail();
            }
        });
    }

    // 이전 활성 카드 클래스 제거
    document.querySelectorAll('.card-container.detail-active').forEach(c => {
        c.classList.remove('detail-active');
    });

    // 현재 카드에 활성 클래스 추가
    const playerContainer = document.querySelector(`.card-container-${playerId}`);
    if (playerContainer) {
        playerContainer.classList.add('detail-active');
    }

    // 카드 정보 가져오기
    const playerRole = document.getElementById(`player-role-${playerId}`).innerText;
    
    // 카드 렌더링
    const cardContainer = overlay.querySelector('.detail-card-container');
    cardContainer.innerHTML = `
        <div class="detail-card">
            <p class="player-id">Player ${playerId}</p>
            <p>${playerRole}</p>
        </div>
    `;

    // 발언 기록 가져오기
    window.getPlayerStatements(playerId).then(statements => {
        const infoPanel = overlay.querySelector('.detail-info-panel');
        infoPanel.innerHTML = `<h2>Player ${playerId} 발언 기록</h2>`;
        
        if (statements.length === 0) {
            infoPanel.innerHTML += '<p style="color: #aaaaaa; font-size: 1.2em;">아직 발언 기록이 없습니다.</p>';
        } else {
            statements.forEach(stmt => {
                const stmtDiv = document.createElement('div');
                stmtDiv.className = 'statement-item';
                stmtDiv.innerHTML = `
                    <div class="day-info">Day ${stmt.day} - ${stmt.phase}</div>
                    <div class="statement-text">${stmt.text}</div>
                `;
                infoPanel.appendChild(stmtDiv);
            });
        }
    });

    currentDetailPlayerId = playerId;
    overlay.classList.add('visible');
}

function closeCardDetail() {
    const overlay = document.getElementById('card-detail-overlay');
    if (overlay) {
        overlay.classList.remove('visible');
    }
    
    // 활성 카드 클래스 제거
    document.querySelectorAll('.card-container.detail-active').forEach(c => {
        c.classList.remove('detail-active');
    });
    
    currentDetailPlayerId = null;
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

