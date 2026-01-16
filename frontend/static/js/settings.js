/**
 * Lecture Transcriber - Settings Page
 * Manages user settings for transcription
 */

// ===========================
// Constants
// ===========================
const STORAGE_KEY = 'lecture_transcriber_settings';

const DEFAULT_SETTINGS = {
    model: 'openai/whisper-medium',
    language: 'ru',
    cleaning_intensity: 2
};

const MODEL_INFO = {
    'openai/whisper-tiny': {
        size: '~75 MB',
        speed: 'Очень быстрая',
        quality: 'Низкое'
    },
    'openai/whisper-base': {
        size: '~150 MB',
        speed: 'Быстрая',
        quality: 'Среднее'
    },
    'openai/whisper-small': {
        size: '~500 MB',
        speed: 'Средняя',
        quality: 'Хорошее'
    },
    'openai/whisper-medium': {
        size: '~1.5 GB',
        speed: 'Средняя',
        quality: 'Высокое'
    },
    'openai/whisper-large': {
        size: '~3 GB',
        speed: 'Медленная',
        quality: 'Максимальное'
    }
};

// ===========================
// Settings Manager Class
// ===========================
class SettingsManager {
    constructor() {
        this.form = document.getElementById('settings-form');
        this.modelSelect = document.getElementById('model-select');
        this.languageSelect = document.getElementById('language-select');
        this.cleaningIntensity = document.getElementById('cleaning-intensity');
        this.intensityValue = document.getElementById('intensity-value');
        this.saveBtn = document.getElementById('save-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.statusMessage = document.getElementById('status-message');
        
        // Model info elements
        this.modelSize = document.getElementById('model-size');
        this.modelSpeed = document.getElementById('model-speed');
        this.modelQuality = document.getElementById('model-quality');
        
        this.init();
    }

    /**
     * Initialize the settings manager
     */
    init() {
        // Load saved settings
        this.loadSettings();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Update model info display
        this.updateModelInfo();
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Form submission
        this.form.addEventListener('submit', (e) => {
            e.preventDefault();
            this.saveSettings();
        });
        
        // Reset button
        this.resetBtn.addEventListener('click', () => {
            this.resetSettings();
        });
        
        // Model selection change
        this.modelSelect.addEventListener('change', () => {
            this.updateModelInfo();
        });
        
        // Cleaning intensity slider
        this.cleaningIntensity.addEventListener('input', (e) => {
            this.intensityValue.textContent = e.target.value;
        });
    }

    /**
     * Load settings from localStorage
     */
    loadSettings() {
        const { storage } = window.LectureTranscriber;
        const settings = storage.get(STORAGE_KEY, DEFAULT_SETTINGS);
        
        // Apply settings to form
        this.modelSelect.value = settings.model;
        this.languageSelect.value = settings.language;
        this.cleaningIntensity.value = settings.cleaning_intensity;
        this.intensityValue.textContent = settings.cleaning_intensity;
    }

    /**
     * Save settings to localStorage
     */
    saveSettings() {
        const { storage, toast } = window.LectureTranscriber;
        
        const settings = {
            model: this.modelSelect.value,
            language: this.languageSelect.value,
            cleaning_intensity: parseInt(this.cleaningIntensity.value, 10)
        };
        
        try {
            storage.set(STORAGE_KEY, settings);
            this.showStatus('Настройки успешно сохранены!', 'success');
            toast.success('Настройки сохранены');
        } catch (error) {
            console.error('Error saving settings:', error);
            this.showStatus('Ошибка при сохранении настроек', 'error');
            toast.error('Не удалось сохранить настройки');
        }
    }

    /**
     * Reset settings to defaults
     */
    resetSettings() {
        const { toast } = window.LectureTranscriber;
        
        // Apply default settings to form
        this.modelSelect.value = DEFAULT_SETTINGS.model;
        this.languageSelect.value = DEFAULT_SETTINGS.language;
        this.cleaningIntensity.value = DEFAULT_SETTINGS.cleaning_intensity;
        this.intensityValue.textContent = DEFAULT_SETTINGS.cleaning_intensity;
        
        // Update model info
        this.updateModelInfo();
        
        // Save defaults
        this.saveSettings();
        
        toast.info('Настройки сброшены по умолчанию');
    }

    /**
     * Update model information display
     */
    updateModelInfo() {
        const selectedModel = this.modelSelect.value;
        const info = MODEL_INFO[selectedModel];
        
        if (info) {
            this.modelSize.textContent = info.size;
            this.modelSpeed.textContent = info.speed;
            this.modelQuality.textContent = info.quality;
        }
    }

    /**
     * Show status message
     * @param {string} message - Message to display
     * @param {string} type - Message type ('success' or 'error')
     */
    showStatus(message, type = 'success') {
        this.statusMessage.textContent = message;
        this.statusMessage.className = `status-message visible ${type}`;
        
        // Hide after 5 seconds
        setTimeout(() => {
            this.statusMessage.classList.remove('visible');
        }, 5000);
    }

    /**
     * Get current settings
     * @returns {object} - Current settings
     */
    getCurrentSettings() {
        return {
            model: this.modelSelect.value,
            language: this.languageSelect.value,
            cleaning_intensity: parseInt(this.cleaningIntensity.value, 10)
        };
    }
}

// ===========================
// Public API
// ===========================

/**
 * Get saved settings from localStorage
 * @returns {object} - Saved settings or defaults
 */
function getSettings() {
    const { storage } = window.LectureTranscriber;
    return storage.get(STORAGE_KEY, DEFAULT_SETTINGS);
}

/**
 * Update specific setting
 * @param {string} key - Setting key
 * @param {*} value - Setting value
 */
function updateSetting(key, value) {
    const { storage } = window.LectureTranscriber;
    const settings = storage.get(STORAGE_KEY, DEFAULT_SETTINGS);
    settings[key] = value;
    storage.set(STORAGE_KEY, settings);
}

// ===========================
// Initialize on DOM Ready
// ===========================
document.addEventListener('DOMContentLoaded', () => {
    const settingsManager = new SettingsManager();
    
    // Export to window for external access
    window.LectureTranscriber.settings = {
        getSettings,
        updateSetting,
        DEFAULT_SETTINGS,
        manager: settingsManager
    };
});
