/**
 * Lecture Transcriber - Common Utilities
 * Provides shared functionality for all frontend pages
 */

// ===========================
// API Configuration
// ===========================
const API_BASE_URL = '/api';

const API_ENDPOINTS = {
    upload: `${API_BASE_URL}/upload`,
    transcribe: `${API_BASE_URL}/transcribe`,
    status: (taskId) => `${API_BASE_URL}/status/${taskId}`,
    result: (taskId) => `${API_BASE_URL}/result/${taskId}`,
    download: (taskId, format) => `${API_BASE_URL}/download/${taskId}?format=${format}`
};

// ===========================
// Toast Notifications
// ===========================
class ToastManager {
    constructor() {
        this.container = document.getElementById('toast-container');
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.id = 'toast-container';
            this.container.className = 'toast-container';
            document.body.appendChild(this.container);
        }
    }

    /**
     * Show a toast notification
     * @param {string} message - The message to display
     * @param {string} type - Type of toast: 'success', 'error', 'warning', 'info'
     * @param {number} duration - Duration in milliseconds (default: 3000)
     */
    show(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const messageEl = document.createElement('p');
        messageEl.className = 'toast-message';
        messageEl.textContent = message;
        
        toast.appendChild(messageEl);
        this.container.appendChild(toast);

        // Auto-remove after duration
        setTimeout(() => {
            toast.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => {
                if (toast.parentNode) {
                    this.container.removeChild(toast);
                }
            }, 300);
        }, duration);
    }

    success(message, duration) {
        this.show(message, 'success', duration);
    }

    error(message, duration) {
        this.show(message, 'error', duration);
    }

    warning(message, duration) {
        this.show(message, 'warning', duration);
    }

    info(message, duration) {
        this.show(message, 'info', duration);
    }
}

// Global toast instance
const toast = new ToastManager();

// ===========================
// Loading Overlay
// ===========================
class LoadingOverlay {
    constructor() {
        this.overlay = document.getElementById('loading-overlay');
        if (!this.overlay) {
            this.overlay = document.createElement('div');
            this.overlay.id = 'loading-overlay';
            this.overlay.className = 'loading-overlay hidden';
            this.overlay.innerHTML = `
                <div class="loading-spinner"></div>
                <p class="loading-text">Обработка...</p>
                <div class="loading-progress hidden">
                    <div class="loading-progress-container">
                        <div class="loading-progress-bar"></div>
                    </div>
                    <span class="loading-progress-text">0%</span>
                </div>
            `;
            document.body.appendChild(this.overlay);
        }
        this.textElement = this.overlay.querySelector('.loading-text');
        this.progressContainer = this.overlay.querySelector('.loading-progress');
        this.progressBar = this.overlay.querySelector('.loading-progress-bar');
        this.progressText = this.overlay.querySelector('.loading-progress-text');
    }

    /**
     * Show the loading overlay
     * @param {string} message - Optional message to display
     * @param {number} progress - Optional progress percentage (0-100)
     */
    show(message = 'Обработка...', progress = null) {
        this.textElement.textContent = message;
        
        if (progress !== null && progress !== undefined) {
            this.progressContainer.classList.remove('hidden');
            this.updateProgress(progress);
        } else {
            this.progressContainer.classList.add('hidden');
        }
        
        this.overlay.classList.remove('hidden');
    }

    /**
     * Update progress bar
     * @param {number} progress - Progress percentage (0-100)
     */
    updateProgress(progress) {
        const percent = Math.max(0, Math.min(100, progress));
        this.progressBar.style.width = `${percent}%`;
        this.progressText.textContent = `${Math.round(percent)}%`;
    }

    /**
     * Hide the loading overlay
     */
    hide() {
        this.overlay.classList.add('hidden');
        this.progressContainer.classList.add('hidden');
    }
}

// Global loading instance
const loading = new LoadingOverlay();

// ===========================
// API Client
// ===========================
class APIClient {
    /**
     * Make an API request
     * @param {string} url - The URL to request
     * @param {object} options - Fetch options
     * @returns {Promise<object>} - The response data
     */
    async request(url, options = {}) {
        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error?.message || `HTTP error! status: ${response.status}`);
            }

            return data;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    /**
     * Upload a video file
     * @param {File} file - The file to upload
     * @param {Function} onProgress - Progress callback
     * @returns {Promise<object>} - Upload response
     */
    async uploadFile(file, onProgress) {
        return new Promise((resolve, reject) => {
            const formData = new FormData();
            formData.append('file', file);

            const xhr = new XMLHttpRequest();

            // Track upload progress
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable && onProgress) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    onProgress(percentComplete);
                }
            });

            // Handle completion
            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        const data = JSON.parse(xhr.responseText);
                        resolve(data);
                    } catch (error) {
                        reject(new Error('Failed to parse response'));
                    }
                } else {
                    try {
                        const error = JSON.parse(xhr.responseText);
                        reject(new Error(error.error?.message || 'Upload failed'));
                    } catch {
                        reject(new Error(`Upload failed with status ${xhr.status}`));
                    }
                }
            });

            // Handle errors
            xhr.addEventListener('error', () => {
                reject(new Error('Network error during upload'));
            });

            xhr.open('POST', API_ENDPOINTS.upload);
            xhr.send(formData);
        });
    }

    /**
     * Start transcription
     * @param {object} params - Transcription parameters
     * @returns {Promise<object>} - Task response
     */
    async startTranscription(params) {
        return this.request(API_ENDPOINTS.transcribe, {
            method: 'POST',
            body: JSON.stringify(params)
        });
    }

    /**
     * Get task status
     * @param {string} taskId - The task ID
     * @returns {Promise<object>} - Status response
     */
    async getStatus(taskId) {
        return this.request(API_ENDPOINTS.status(taskId));
    }

    /**
     * Get task result
     * @param {string} taskId - The task ID
     * @returns {Promise<object>} - Result response
     */
    async getResult(taskId) {
        return this.request(API_ENDPOINTS.result(taskId));
    }

    /**
     * Get download URL
     * @param {string} taskId - The task ID
     * @param {string} format - File format ('md' or 'json')
     * @returns {string} - Download URL
     */
    getDownloadUrl(taskId, format = 'md') {
        return API_ENDPOINTS.download(taskId, format);
    }
}

// Global API client instance
const api = new APIClient();

// ===========================
// Local Storage Manager
// ===========================
class StorageManager {
    /**
     * Get item from localStorage
     * @param {string} key - Storage key
     * @param {*} defaultValue - Default value if key doesn't exist
     * @returns {*} - Stored value or default
     */
    get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error('Error reading from localStorage:', error);
            return defaultValue;
        }
    }

    /**
     * Set item in localStorage
     * @param {string} key - Storage key
     * @param {*} value - Value to store
     */
    set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
        } catch (error) {
            console.error('Error writing to localStorage:', error);
        }
    }

    /**
     * Remove item from localStorage
     * @param {string} key - Storage key
     */
    remove(key) {
        try {
            localStorage.removeItem(key);
        } catch (error) {
            console.error('Error removing from localStorage:', error);
        }
    }

    /**
     * Clear all items from localStorage
     */
    clear() {
        try {
            localStorage.clear();
        } catch (error) {
            console.error('Error clearing localStorage:', error);
        }
    }
}

// Global storage instance
const storage = new StorageManager();

// ===========================
// Utility Functions
// ===========================

/**
 * Format file size to human-readable string
 * @param {number} bytes - File size in bytes
 * @returns {string} - Formatted size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Format duration to human-readable string
 * @param {number} seconds - Duration in seconds
 * @returns {string} - Formatted duration
 */
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }
    return `${minutes}:${String(secs).padStart(2, '0')}`;
}

/**
 * Format timestamp to readable format
 * @param {string|Date} timestamp - ISO timestamp or Date object
 * @returns {string} - Formatted timestamp
 */
function formatTimestamp(timestamp) {
    const date = new Date(timestamp);
    return date.toLocaleString('ru-RU', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Validate video file
 * @param {File} file - File to validate
 * @returns {object} - Validation result {valid: boolean, error: string}
 */
function validateVideoFile(file) {
    const maxSize = 500 * 1024 * 1024; // 500 MB
    const allowedTypes = ['video/mp4', 'video/webm', 'video/x-matroska'];
    const allowedExtensions = ['.mp4', '.webm', '.mkv'];
    
    if (!file) {
        return { valid: false, error: 'Файл не выбран' };
    }
    
    // Check file size
    if (file.size > maxSize) {
        return { valid: false, error: `Размер файла превышает ${formatFileSize(maxSize)}` };
    }
    
    // Check file type
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
        return { valid: false, error: 'Неподдерживаемый формат файла. Используйте MP4, WebM или MKV' };
    }
    
    return { valid: true };
}

/**
 * Debounce function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} - Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Poll for task status
 * @param {string} taskId - Task ID to poll
 * @param {Function} onUpdate - Callback for status updates
 * @param {number} interval - Polling interval in milliseconds
 * @returns {Promise<object>} - Final result
 */
async function pollTaskStatus(taskId, onUpdate, interval = 2000) {
    return new Promise((resolve, reject) => {
        const poll = async () => {
            try {
                const status = await api.getStatus(taskId);
                
                if (onUpdate) {
                    onUpdate(status);
                }
                
                if (status.status === 'completed') {
                    resolve(status);
                } else if (status.status === 'failed') {
                    reject(new Error(status.message || 'Task failed'));
                } else {
                    setTimeout(poll, interval);
                }
            } catch (error) {
                reject(error);
            }
        };
        
        poll();
    });
}

// ===========================
// Initialize on DOM Ready
// ===========================
document.addEventListener('DOMContentLoaded', () => {
    // Set active navigation item
    const currentPath = window.location.pathname;
    const navItems = document.querySelectorAll('.navbar-item');
    
    navItems.forEach(item => {
        const href = item.getAttribute('href');
        if (href === currentPath || (currentPath === '/' && href === '/')) {
            item.classList.add('active');
        }
    });
});

// ===========================
// Export for use in other scripts
// ===========================
window.LectureTranscriber = {
    toast,
    loading,
    api,
    storage,
    formatFileSize,
    formatDuration,
    formatTimestamp,
    validateVideoFile,
    debounce,
    pollTaskStatus,
    API_ENDPOINTS
};
