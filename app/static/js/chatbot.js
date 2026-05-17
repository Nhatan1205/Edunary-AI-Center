/**
 * Edunary Chatbot — WebSocket Chat Logic
 * Extracted from chat_new.html for maintainability.
 *
 * Reads optional JWT token from URL query param: ?token=<jwt>
 * Token is sent as an auth handshake message after WebSocket connects
 * (never in the WS URL, so it never appears in server access logs).
 */

// Configure marked
marked.setOptions({
    breaks: true,
    gfm: true,
    highlight: function (code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try { return hljs.highlight(code, { language: lang }).value; } catch (e) {}
        }
        return hljs.highlightAuto(code).value;
    }
});


class ChatBot {
    constructor() {
        this.sessionId = 'session_' + Math.random().toString(36).substr(2, 12) + '_' + Date.now();
        this.websocket = null;
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.chatWidget = document.getElementById('chatWidget');
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.currentStreamingMessage = null;
        this.streamingContent = '';
        this.typingIndicator = null;

        // Read JWT token from URL query param (?token=...)
        this.jwtToken = this._readTokenFromUrl();

        this.init();
    }

    // ── Initialisation ──────────────────────────────────────────

    _readTokenFromUrl() {
        const params = new URLSearchParams(window.location.search);
        return params.get('token') || null;
    }

    init() {
        this.setupEventListeners();
        this.connectWebSocket();
        this.addMessage('Hello! How can I help you? 😊', 'bot');
        setTimeout(() => this.messageInput.focus(), 300);
    }

    // ── Event Listeners ─────────────────────────────────────────

    setupEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());

        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        this.messageInput.addEventListener('input', () => {
            this.sendButton.disabled = !this.messageInput.value.trim();
        });
    }

    // ── WebSocket ───────────────────────────────────────────────

    connectWebSocket() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            // Token is NOT in the URL — sent as first message after connect.
            const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;

            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                this.reconnectAttempts = 0;

                // Auth handshake — send token as first message (never in URL)
                if (this.jwtToken) {
                    this.websocket.send(JSON.stringify({
                        type: 'auth',
                        token: this.jwtToken,
                    }));
                }
            };

            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this._handleWsMessage(data);
                } catch (_) {}
            };

            this.websocket.onclose = (event) => {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
                    setTimeout(() => this.connectWebSocket(), delay);
                }
            };

            this.websocket.onerror = () => {};
        } catch (_) {}
    }

    _handleWsMessage(data) {
        if (data.type === 'stream_start') {
            this.hideTyping();
            this.streamingContent = '';
            this.currentStreamingMessage = this.addStreamingMessage();
        } else if (data.type === 'stream_chunk') {
            this.streamingContent += data.content;
            this.updateStreamingMessage();
        } else if (data.type === 'stream_end') {
            this.finalizeStreamingMessage();
            this.enableInput();
        } else if (data.type === 'course_cards') {
            this.addCourseCards(data.courses);
        } else if (data.type === 'bot_message') {
            this.hideTyping();
            this.addMessage(data.message, 'bot');
            this.enableInput();
        }
    }

    // ── Sending ─────────────────────────────────────────────────

    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || !this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            return;
        }

        this.addMessage(message, 'user');
        this.showTyping();
        this.disableInput();

        this.websocket.send(JSON.stringify({
            message: message,
            session_id: this.sessionId,
        }));

        this.messageInput.value = '';
        this.sendButton.disabled = true;
    }

    // ── Message Rendering ────────────────────────────────────────

    addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        if (type === 'bot') {
            const htmlContent = marked.parse(content);
            messageDiv.innerHTML = `<div class="message-content">${htmlContent}</div>`;
        } else {
            const escapedContent = this.escapeHtml(content);
            messageDiv.innerHTML = `<div class="message-content">${escapedContent}</div>`;
        }

        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
    }

    addStreamingMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message bot';
        messageDiv.innerHTML = '<div class="message-content"><span class="streaming-content"></span><span class="cursor">|</span></div>';
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        return messageDiv;
    }

    updateStreamingMessage() {
        if (this.currentStreamingMessage) {
            const el = this.currentStreamingMessage.querySelector('.streaming-content');
            if (el) {
                el.innerHTML = marked.parse(this.streamingContent);
                this.scrollToBottom();
            }
        }
    }

    finalizeStreamingMessage() {
        if (this.currentStreamingMessage) {
            this.currentStreamingMessage.querySelector('.cursor')?.remove();
            const el = this.currentStreamingMessage.querySelector('.streaming-content');
            if (el) {
                el.innerHTML = marked.parse(this.streamingContent);
            }
            this.currentStreamingMessage = null;
            this.streamingContent = '';
        }
    }

    // ── Typing Indicator ─────────────────────────────────────────

    showTyping() {
        if (!this.typingIndicator) {
            this.typingIndicator = document.createElement('div');
            this.typingIndicator.className = 'typing-indicator';
            this.typingIndicator.innerHTML = `
                <div class="typing-indicator-content">
                    <div class="typing-dots"><div class="dot"></div></div>
                </div>
            `;
        }
        this.chatMessages.appendChild(this.typingIndicator);
        this.typingIndicator.style.display = 'block';
        this.scrollToBottom();
    }

    hideTyping() {
        if (this.typingIndicator?.parentNode) {
            this.typingIndicator.remove();
            this.typingIndicator = null;
        }
    }

    // ── Input State ──────────────────────────────────────────────

    disableInput() {
        this.messageInput.disabled = true;
        this.sendButton.disabled = true;
    }

    enableInput() {
        this.messageInput.disabled = false;
        this.sendButton.disabled = !this.messageInput.value.trim();
        this.messageInput.focus();
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }

    // ── Course Cards ─────────────────────────────────────────────

    addCourseCards(courses) {
        if (!courses || courses.length === 0) return;

        const cardsContainer = document.createElement('div');
        cardsContainer.className = 'course-cards-container';

        courses.forEach(course => {
            const card = this.createCourseCard(course);
            cardsContainer.appendChild(card);
        });

        this.chatMessages.appendChild(cardsContainer);
        this.scrollToBottom();
    }

    createCourseCard(course) {
        const card = document.createElement('a');
        card.className = 'course-card';
        card.href = course.courseUrl;
        card.target = '_blank';

        const imageUrl = course.imageUrl || 'https://via.placeholder.com/80?text=Course';

        card.innerHTML = `
            <div class="course-card-header">
                <img src="${imageUrl}" alt="${this.escapeHtml(course.title)}" class="course-image" />
                <div class="course-info">
                    <div class="course-title">${this.escapeHtml(course.title)}</div>
                    ${course.subtitle ? `<div class="course-subtitle">${this.escapeHtml(course.subtitle)}</div>` : ''}
                    <div class="course-instructor">👨‍🏫 ${this.escapeHtml(course.instructorName)}</div>
                </div>
            </div>
        `;

        return card;
    }

    // ── Utilities ────────────────────────────────────────────────

    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }
}

// Bootstrap when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ChatBot();
});
