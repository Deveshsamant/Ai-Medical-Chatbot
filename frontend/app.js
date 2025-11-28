/**
 * Medical Chatbot Frontend Application
 * Handles UI interactions and API communication
 */

// ============================================================================
// CONFIGURATION
// ============================================================================
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000,
    TYPING_DELAY: 50,
};

// ============================================================================
// STATE MANAGEMENT
// ============================================================================
const state = {
    conversationHistory: [],
    isProcessing: false,
    isConnected: false,
};

// ============================================================================
// DOM ELEMENTS
// ============================================================================
let elements = {};

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    console.log('🚀 Initializing Medical Chatbot...');

    // Initialize DOM elements
    initializeElements();

    // Setup event listeners
    setupEventListeners();

    // Check backend connection
    checkBackendHealth();

    // Load conversation history from localStorage
    loadConversationHistory();

    // Auto-resize textarea
    autoResizeTextarea();

    console.log('✅ Initialization complete');
}

function initializeElements() {
    elements = {
        chatMessages: document.getElementById('chat-messages'),
        chatForm: document.getElementById('chat-form'),
        userInput: document.getElementById('user-input'),
        sendBtn: document.getElementById('send-btn'),
        clearBtn: document.getElementById('clear-chat-btn'),
        typingIndicator: document.getElementById('typing-indicator'),
        statusIndicator: document.getElementById('status-indicator'),
        charCount: document.getElementById('char-count'),
        devProfileBtn: document.getElementById('dev-profile-btn'),
        devModal: document.getElementById('dev-modal'),
        closeModalBtn: document.getElementById('close-modal-btn'),
    };

    console.log('Elements initialized:', elements);
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================
function setupEventListeners() {
    // Form submission
    elements.chatForm.addEventListener('submit', handleFormSubmit);

    // Clear chat button
    elements.clearBtn.addEventListener('click', clearConversation);

    // Input character count
    elements.userInput.addEventListener('input', updateCharCount);

    // Auto-resize textarea
    elements.userInput.addEventListener('input', autoResizeTextarea);

    // Enter key to send (Shift+Enter for new line)
    elements.userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            elements.chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Suggestion chips
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('suggestion-chip')) {
            const question = e.target.dataset.question;
            elements.userInput.value = question;
            elements.userInput.focus();
            updateCharCount();
        }
    });

    // Developer Profile Modal
    if (elements.devProfileBtn) {
        console.log('Attaching click listener to Developer button');
        elements.devProfileBtn.addEventListener('click', (e) => {
            console.log('Developer button clicked');
            e.preventDefault(); // Prevent any default behavior
            openDevModal();
        });
    } else {
        console.error('Developer button not found in DOM');
    }

    if (elements.closeModalBtn) {
        elements.closeModalBtn.addEventListener('click', closeDevModal);
    }

    // Close modal when clicking outside
    if (elements.devModal) {
        elements.devModal.addEventListener('click', (e) => {
            if (e.target === elements.devModal) {
                closeDevModal();
            }
        });
    }

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && elements.devModal.classList.contains('active')) {
            closeDevModal();
        }
    });
}

// ============================================================================
// MODAL HANDLING
// ============================================================================
// Make openDevModal global for debugging
window.openDevModal = function () {
    console.log('openDevModal called');
    const modal = document.getElementById('dev-modal');
    if (modal) {
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
        console.log('Modal active class added');
    } else {
        console.error('Modal element not found by ID');
    }
};

function closeDevModal() {
    elements.devModal.classList.remove('active');
    document.body.style.overflow = ''; // Restore scrolling
}

// ============================================================================
// FORM HANDLING
// ============================================================================
async function handleFormSubmit(e) {
    e.preventDefault();

    const message = elements.userInput.value.trim();

    if (!message || state.isProcessing) {
        return;
    }

    // Clear input
    elements.userInput.value = '';
    updateCharCount();
    autoResizeTextarea();

    // Remove welcome message if present
    removeWelcomeMessage();

    // Add user message to UI
    addMessage('user', message);

    // Add to conversation history
    state.conversationHistory.push({
        role: 'user',
        content: message,
        timestamp: new Date().toISOString(),
    });

    // Show typing indicator
    showTypingIndicator();

    // Disable input
    setProcessingState(true);

    try {
        // Send message to backend
        const response = await sendChatMessage(message);

        // Hide typing indicator
        hideTypingIndicator();

        // Add assistant response
        addMessage('assistant', response.message, response.sources, response.confidence);

        // Add to conversation history
        state.conversationHistory.push({
            role: 'assistant',
            content: response.message,
            timestamp: new Date().toISOString(),
        });

        // Save to localStorage
        saveConversationHistory();

    } catch (error) {
        console.error('Error sending message:', error);
        hideTypingIndicator();
        addMessage('assistant',
            '❌ Sorry, I encountered an error processing your request. Please check if the backend server is running and try again.',
            [],
            null
        );
    } finally {
        setProcessingState(false);
    }
}

// ============================================================================
// API COMMUNICATION
// ============================================================================
async function sendChatMessage(message, retries = 0) {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                conversation_history: state.conversationHistory.slice(-6), // Last 6 messages
                max_sources: 3,
                temperature: 0.7,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;

    } catch (error) {
        console.error(`API request failed (attempt ${retries + 1}/${CONFIG.MAX_RETRIES}):`, error);

        if (retries < CONFIG.MAX_RETRIES - 1) {
            await sleep(CONFIG.RETRY_DELAY);
            return sendChatMessage(message, retries + 1);
        }

        throw error;
    }
}

async function checkBackendHealth() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            setConnectionStatus(true);
            console.log('✅ Backend connected:', data);
        } else {
            setConnectionStatus(false);
            console.warn('⚠️ Backend degraded:', data);
        }
    } catch (error) {
        setConnectionStatus(false);
        console.error('❌ Backend connection failed:', error);
    }
}

// ============================================================================
// UI UPDATES
// ============================================================================
function addMessage(role, content, sources = [], confidence = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    // Avatar
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'assistant'
        ? `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
             <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2"/>
             <path d="M2 17L12 22L22 17M2 12L12 17L22 12" stroke="currentColor" stroke-width="2"/>
           </svg>`
        : `<svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
             <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2M12 11a4 4 0 100-8 4 4 0 000 8z" stroke="currentColor" stroke-width="2"/>
           </svg>`;

    // Content
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    const text = document.createElement('div');
    text.className = 'message-text';
    text.textContent = content;

    bubble.appendChild(text);

    // Add sources if available
    if (sources && sources.length > 0) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';

        const sourcesTitle = document.createElement('div');
        sourcesTitle.className = 'sources-title';
        sourcesTitle.textContent = '📚 Sources:';
        sourcesDiv.appendChild(sourcesTitle);

        sources.forEach((source, index) => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            sourceItem.textContent = `${index + 1}. ${source.content}`;
            if (source.similarity_score) {
                sourceItem.textContent += ` (${(source.similarity_score * 100).toFixed(0)}% relevant)`;
            }
            sourcesDiv.appendChild(sourceItem);
        });

        bubble.appendChild(sourcesDiv);
    }

    // Add metadata
    const meta = document.createElement('div');
    meta.className = 'message-meta';

    const time = document.createElement('span');
    time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    meta.appendChild(time);

    if (confidence !== null) {
        const confidenceSpan = document.createElement('span');
        confidenceSpan.textContent = `Confidence: ${(confidence * 100).toFixed(0)}%`;
        meta.appendChild(confidenceSpan);
    }

    contentDiv.appendChild(bubble);
    contentDiv.appendChild(meta);

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);

    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function removeWelcomeMessage() {
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.style.animation = 'fadeOut 0.3s ease-out';
        setTimeout(() => welcomeMsg.remove(), 300);
    }
}

function showTypingIndicator() {
    elements.typingIndicator.style.display = 'flex';
    scrollToBottom();
}

function hideTypingIndicator() {
    elements.typingIndicator.style.display = 'none';
}

function setProcessingState(isProcessing) {
    state.isProcessing = isProcessing;
    elements.sendBtn.disabled = isProcessing;
    elements.userInput.disabled = isProcessing;
}

function setConnectionStatus(isConnected) {
    state.isConnected = isConnected;

    if (isConnected) {
        elements.statusIndicator.classList.remove('disconnected');
        elements.statusIndicator.title = 'Connected';
    } else {
        elements.statusIndicator.classList.add('disconnected');
        elements.statusIndicator.title = 'Disconnected - Check if backend is running';
    }
}

function updateCharCount() {
    const count = elements.userInput.value.length;
    elements.charCount.textContent = `${count} / 2000`;

    if (count > 1800) {
        elements.charCount.style.color = 'var(--color-warning)';
    } else {
        elements.charCount.style.color = 'var(--color-text-muted)';
    }
}

function autoResizeTextarea() {
    elements.userInput.style.height = 'auto';
    elements.userInput.style.height = elements.userInput.scrollHeight + 'px';
}

function scrollToBottom() {
    setTimeout(() => {
        elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
    }, 100);
}

function clearConversation() {
    if (!confirm('Are you sure you want to clear the conversation?')) {
        return;
    }

    // Clear messages except welcome
    const messages = elements.chatMessages.querySelectorAll('.message');
    messages.forEach(msg => msg.remove());

    // Clear history
    state.conversationHistory = [];
    localStorage.removeItem('chatHistory');

    // Show welcome message again
    location.reload();
}

// ============================================================================
// LOCAL STORAGE
// ============================================================================
function saveConversationHistory() {
    try {
        localStorage.setItem('chatHistory', JSON.stringify(state.conversationHistory));
    } catch (error) {
        console.error('Error saving conversation history:', error);
    }
}

function loadConversationHistory() {
    try {
        const saved = localStorage.getItem('chatHistory');
        if (saved) {
            state.conversationHistory = JSON.parse(saved);

            // Restore messages to UI
            if (state.conversationHistory.length > 0) {
                removeWelcomeMessage();
                state.conversationHistory.forEach(msg => {
                    addMessage(msg.role, msg.content);
                });
            }
        }
    } catch (error) {
        console.error('Error loading conversation history:', error);
    }
}

// ============================================================================
// UTILITIES
// ============================================================================
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Add fadeOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-20px); }
    }
`;
document.head.appendChild(style);

// ============================================================================
// PERIODIC HEALTH CHECK
// ============================================================================
setInterval(() => {
    checkBackendHealth();
}, 30000); // Check every 30 seconds

console.log('💬 Medical Chatbot Frontend Loaded');
