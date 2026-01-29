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
    console.log('showCardDetail called with playerId:', playerId);
    
    // 이미 열려있는 카드를 다시 클릭하면 닫기
    if (currentDetailPlayerId === playerId) {
        closeCardDetail();
        return;
    }

    // 이전 카드가 열려있으면 닫기
    if (currentDetailPlayerId !== null) {
        closeCardDetail();
    }

    // 오버레이가 없으면 생성
    let overlay = document.getElementById('card-detail-overlay');
    if (!overlay) {
        console.log('Creating new overlay');
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

    // 원래 카드 숨기기
    const originalCard = document.querySelector(`.card-container-${playerId}`);
    if (originalCard) {
        originalCard.style.opacity = '0';
        originalCard.dataset.hidden = 'true';
    }

    // 원래 카드 위치 가져오기
    const rect = originalCard.getBoundingClientRect();
    const startLeft = rect.left + rect.width / 2;
    const startTop = rect.top + rect.height / 2;

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

    // detail-card를 원래 위치에서 시작
    cardContainer.style.left = startLeft + 'px';
    cardContainer.style.top = startTop + 'px';
    cardContainer.style.transform = 'translate(-50%, -50%) scale(0.333)';

    // 오버레이 표시
    overlay.classList.add('visible');

    // 애니메이션 시작: 목표 위치로 이동 및 회전
    requestAnimationFrame(() => {
        const detailCard = cardContainer.querySelector('.detail-card');
        
        requestAnimationFrame(() => {
            cardContainer.style.left = '15%';
            cardContainer.style.top = '50%';
            cardContainer.style.transform = 'translate(-50%, -50%) scale(1)';
            detailCard.classList.add('spinning');
        });
    });

    // 발언 기록 가져오기
    window.getPlayerStatements(playerId).then(statements => {
        const infoPanel = overlay.querySelector('.detail-info-panel');
        if (infoPanel) {
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
        }
    }).catch(err => {
        console.error('Failed to get player statements:', err);
    });

    currentDetailPlayerId = playerId;
    console.log('Overlay visible, currentDetailPlayerId:', currentDetailPlayerId);
}

function closeCardDetail() {
    console.log('closeCardDetail called');
    const overlay = document.getElementById('card-detail-overlay');
    if (overlay) {
        overlay.classList.remove('visible');
    }
    
    // 원래 카드 다시 보이기
    if (currentDetailPlayerId !== null) {
        const originalCard = document.querySelector(`.card-container-${currentDetailPlayerId}`);
        if (originalCard && originalCard.dataset.hidden === 'true') {
            originalCard.style.opacity = '1';
            delete originalCard.dataset.hidden;
        }
    }
    
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

