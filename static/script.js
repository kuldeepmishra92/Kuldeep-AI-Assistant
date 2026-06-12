// --- State ---
let sessionId   = Math.random().toString(36).substring(2, 10);
let messages    = [];
let isWelcome   = true;
// Default is light-theme. isDark = true only if user explicitly chose dark.
let isDarkTheme = localStorage.getItem('theme') === 'dark';

// --- Selectors ---
const body            = document.body;
const themeToggle     = document.getElementById('theme-toggle');
const chatMessages    = document.getElementById('chat-messages');
const userInput       = document.getElementById('user-input');
const sendBtn         = document.getElementById('send-btn');
const welcomeScreen   = document.getElementById('welcome-screen');
const inputWrapper    = document.getElementById('input-container-wrapper');
const newChatBtn      = document.getElementById('new-chat-btn');
const statMessages    = document.getElementById('stat-messages');
const downloadBtn     = document.getElementById('download-chat-btn');
const mobileMenuBtn   = document.getElementById('mobile-menu-btn');
const sidebarOverlay  = document.getElementById('sidebar-overlay');
const sidebar         = document.querySelector('.sidebar');
const desktopToggle   = document.getElementById('desktop-sidebar-toggle');

// --- Init ---
function init() {
    // Apply saved theme preference
    if (isDarkTheme) {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
        themeToggle.querySelector('i').setAttribute('data-lucide', 'sun');
    } else {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
        themeToggle.querySelector('i').setAttribute('data-lucide', 'moon');
    }
    lucide.createIcons();

    if (window.marked) {
        marked.setOptions({ breaks: true, gfm: true, headerIds: false, mangle: false });
    }
}

// --- Theme toggle ---
themeToggle.addEventListener('click', () => {
    isDarkTheme = !isDarkTheme;
    if (isDarkTheme) {
        body.classList.remove('light-theme');
        body.classList.add('dark-theme');
        themeToggle.querySelector('i').setAttribute('data-lucide', 'sun');
    } else {
        body.classList.remove('dark-theme');
        body.classList.add('light-theme');
        themeToggle.querySelector('i').setAttribute('data-lucide', 'moon');
    }
    localStorage.setItem('theme', isDarkTheme ? 'dark' : 'light');
    lucide.createIcons();
});

// --- Mobile sidebar ---
if (mobileMenuBtn && sidebarOverlay && sidebar) {
    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.add('open');
        sidebarOverlay.classList.add('active');
    });
    sidebarOverlay.addEventListener('click', () => {
        sidebar.classList.remove('open');
        sidebarOverlay.classList.remove('active');
    });
}

if (desktopToggle) {
    desktopToggle.addEventListener('click', () => sidebar.classList.toggle('collapsed'));
}

// --- Welcome transition ---
function exitWelcome() {
    if (!isWelcome) return;
    isWelcome = false;
    welcomeScreen.style.opacity    = '0';
    welcomeScreen.style.visibility = 'hidden';
    inputWrapper.classList.remove('welcome-mode');
}

// --- Input auto-resize ---
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = userInput.scrollHeight + 'px';
    sendBtn.disabled = userInput.value.trim() === '';
});

userInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

sendBtn.addEventListener('click', sendMessage);

newChatBtn.addEventListener('click', () => {
    sessionId  = Math.random().toString(36).substring(2, 10);
    messages   = [];
    chatMessages.innerHTML = '';
    isWelcome  = true;
    welcomeScreen.style.opacity    = '1';
    welcomeScreen.style.visibility = 'visible';
    inputWrapper.classList.add('welcome-mode');
    statMessages.textContent = '0';
    downloadBtn.disabled     = true;
});

// --- Chat ---
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    exitWelcome();
    userInput.value      = '';
    userInput.style.height = 'auto';
    sendBtn.disabled     = true;

    addMessage('user', text);

    const typingId = 'typing-' + Date.now();
    addTypingIndicator(typingId);

    try {
        const response = await fetch('/api/chat', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ message: text, session_id: sessionId }),
        });

        if (!response.ok) throw new Error('Network error');

        removeTypingIndicator(typingId);
        const msgId     = 'msg-' + Date.now();
        const contentEl = initAssistantMessage(msgId);

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let fullMarkdown = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            fullMarkdown += decoder.decode(value);
            const processed = processMarkdown(fullMarkdown);
            contentEl.innerHTML = window.marked ? marked.parse(processed) : processed;
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        messages.push({ role: 'assistant', content: fullMarkdown });
        statMessages.textContent = Math.ceil(messages.length / 2);
        downloadBtn.disabled     = false;

    } catch (err) {
        console.error('Chat error:', err);
        removeTypingIndicator(typingId);
        addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
    }
}

function processMarkdown(md) {
    // Just return as-is — marked.js handles everything correctly
    return md;
}

function addMessage(role, content) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;
    const processed = processMarkdown(content);
    msgDiv.innerHTML = `
        <div class="avatar">${role === 'user' ? 'You' : '<img src="/static/ai-avatar.png" alt="Kuldeep AI">'}</div>
        <div class="message-content">
            <div class="bubble">${role === 'user' ? processed : (window.marked ? marked.parse(processed) : processed)}</div>
            <div class="message-meta">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
        </div>`;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    if (role === 'user') messages.push({ role, content });
}

function addTypingIndicator(id) {
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="avatar"><img src="/static/ai-avatar.png" alt="Kuldeep AI"></div>
        <div class="message-content">
            <div class="typing-bubble">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function initAssistantMessage(id) {
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="avatar"><img src="/static/ai-avatar.png" alt="Kuldeep AI"></div>
        <div class="message-content">
            <div class="bubble assistant-bubble"></div>
            <div class="message-meta">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
        </div>`;
    chatMessages.appendChild(div);
    return div.querySelector('.assistant-bubble');
}

// --- Download ---
downloadBtn.addEventListener('click', () => {
    let md = `# Chat Session ${sessionId}\n\n`;
    messages.forEach(m => {
        const role = m.role === 'user' ? 'You' : 'Kuldeep AI';
        md += `### ${role}\n${m.content}\n\n---\n\n`;
    });
    const blob = new Blob([md], { type: 'text/markdown' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `chat_${sessionId}.md`; a.click();
});

init();
