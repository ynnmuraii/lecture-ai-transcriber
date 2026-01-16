/**
 * Lecture Transcriber - Upload Page
 * Handles video file upload and transcription initiation
 */

// ===========================
// State Management
// ===========================
class UploadState {
    constructor() {
        this.selectedFile = null;
        this.uploadedFileId = null;
        this.isUploading = false;
        this.isProcessing = false;
    }

    setFile(file) {
        this.selectedFile = file;
    }

    clearFile() {
        this.selectedFile = null;
        this.uploadedFileId = null;
    }

    setUploadedFileId(fileId) {
        this.uploadedFileId = fileId;
    }

    setUploading(status) {
        this.isUploading = status;
    }

    setProcessing(status) {
        this.isProcessing = status;
    }

    hasFile() {
        return this.selectedFile !== null;
    }

    isReady() {
        return this.uploadedFileId !== null && !this.isUploading && !this.isProcessing;
    }
}

const uploadState = new UploadState();

// ===========================
// DOM Elements (initialized on DOMContentLoaded)
// ===========================
let elements = {};

// ===========================
// Utility accessors
// ===========================
function getUtils() {
    return window.LectureTranscriber || {};
}

// ===========================
// File Selection & Validation
// ===========================

/**
 * Handle file selection
 * @param {File} file - Selected file
 */
function handleFileSelect(file) {
    if (!file) return;

    const utils = getUtils();
    
    // Validate file
    const validation = utils.validateVideoFile(file);
    if (!validation.valid) {
        utils.toast.error(validation.error);
        return;
    }

    // Update state
    uploadState.setFile(file);

    // Update UI
    updateFileInfo(file);
    elements.uploadZone.classList.add('has-file');
    elements.fileInfo.classList.add('visible');
    elements.resetBtn.disabled = false;

    // Auto-upload file
    uploadFile(file);
}

/**
 * Update file info display
 * @param {File} file - File to display info for
 */
function updateFileInfo(file) {
    const utils = getUtils();
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = utils.formatFileSize(file.size);
}

/**
 * Clear file selection
 */
function clearFileSelection() {
    uploadState.clearFile();
    elements.fileInput.value = '';
    elements.uploadZone.classList.remove('has-file');
    elements.fileInfo.classList.remove('visible');
    elements.uploadProgress.classList.remove('visible');
    elements.resetBtn.disabled = true;
    elements.startBtn.disabled = true;
    updateProgress(0);
}

// ===========================
// Drag and Drop
// ===========================

/**
 * Initialize drag and drop functionality
 */
function initDragAndDrop() {
    // Prevent default drag behaviors on document
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when dragging over it
    ['dragenter', 'dragover'].forEach(eventName => {
        elements.uploadZone.addEventListener(eventName, () => {
            elements.uploadZone.classList.add('drag-over');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        elements.uploadZone.addEventListener(eventName, () => {
            elements.uploadZone.classList.remove('drag-over');
        }, false);
    });

    // Handle dropped files
    elements.uploadZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    }, false);
}

// ===========================
// File Upload
// ===========================

/**
 * Upload file to server
 * @param {File} file - File to upload
 */
async function uploadFile(file) {
    if (uploadState.isUploading) return;

    const utils = getUtils();
    
    uploadState.setUploading(true);
    elements.uploadProgress.classList.add('visible');
    elements.startBtn.disabled = true;

    try {
        const response = await utils.api.uploadFile(file, (progress) => {
            updateProgress(progress);
        });

        // Store uploaded file ID
        uploadState.setUploadedFileId(response.file_id);

        // Update UI
        updateProgress(100);
        utils.toast.success('Файл успешно загружен!');
        elements.startBtn.disabled = false;

        // Save file info to storage for later use
        utils.storage.set('lastUpload', {
            fileId: response.file_id,
            fileName: response.filename,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Upload failed:', error);
        utils.toast.error(`Ошибка загрузки: ${error.message}`);
        clearFileSelection();
    } finally {
        uploadState.setUploading(false);
    }
}

/**
 * Update upload progress
 * @param {number} percent - Progress percentage (0-100)
 */
function updateProgress(percent) {
    const rounded = Math.round(percent);
    elements.progressBar.style.width = `${rounded}%`;
    elements.progressPercent.textContent = `${rounded}%`;
}

// ===========================
// Transcription
// ===========================

/**
 * Start transcription process
 */
async function startTranscription() {
    const utils = getUtils();
    
    if (!uploadState.isReady()) {
        utils.toast.warning('Пожалуйста, дождитесь завершения загрузки файла');
        return;
    }

    // Get form values
    const formData = new FormData(elements.settingsForm);
    const params = {
        file_id: uploadState.uploadedFileId,
        model: formData.get('model'),
        language: formData.get('language'),
        cleaning_intensity: parseInt(formData.get('cleaning_intensity'))
    };

    uploadState.setProcessing(true);
    elements.startBtn.disabled = true;
    elements.resetBtn.disabled = true;

    try {
        utils.loading.show('Запуск транскрипции...');

        // Start transcription
        const response = await utils.api.startTranscription(params);
        const taskId = response.task_id;

        utils.toast.success('Транскрипция запущена!');

        // Save task ID and settings
        utils.storage.set('currentTask', {
            taskId: taskId,
            fileName: uploadState.selectedFile.name,
            settings: params,
            startedAt: new Date().toISOString()
        });

        // Poll for status updates
        utils.loading.show('Обработка видео...', 0);

        await utils.pollTaskStatus(taskId, (status) => {
            // Update loading message with progress
            if (status.progress !== undefined) {
                const percent = Math.round(status.progress);
                const message = status.message || `Обработка видео... ${percent}%`;
                utils.loading.show(message, percent);
            }
            if (status.message) {
                console.log('Status update:', status.message);
            }
        });

        // Transcription completed
        utils.loading.hide();
        utils.toast.success('Транскрипция завершена!');

        // Redirect to results page
        setTimeout(() => {
            window.location.href = `/results?task_id=${taskId}`;
        }, 1000);

    } catch (error) {
        console.error('Transcription failed:', error);
        utils.loading.hide();
        utils.toast.error(`Ошибка транскрипции: ${error.message}`);
        elements.startBtn.disabled = false;
        elements.resetBtn.disabled = false;
    } finally {
        uploadState.setProcessing(false);
    }
}

/**
 * Reset form and clear selection
 */
function resetForm() {
    const utils = getUtils();
    
    if (uploadState.isUploading || uploadState.isProcessing) {
        utils.toast.warning('Дождитесь завершения текущей операции');
        return;
    }

    clearFileSelection();
    elements.settingsForm.reset();
    elements.intensityValue.textContent = '2';
    utils.toast.info('Форма сброшена');
}

// ===========================
// Settings Management
// ===========================

/**
 * Load saved settings from storage
 */
function loadSavedSettings() {
    const utils = getUtils();
    
    // Try to load from settings page first, then fall back to upload-specific settings
    let savedSettings = utils.storage.get('lecture_transcriber_settings');
    if (!savedSettings) {
        savedSettings = utils.storage.get('uploadSettings');
    }
    
    if (savedSettings) {
        if (savedSettings.model) {
            elements.modelSelect.value = savedSettings.model;
        }
        if (savedSettings.language) {
            elements.languageSelect.value = savedSettings.language;
        }
        if (savedSettings.cleaning_intensity !== undefined) {
            elements.cleaningIntensity.value = savedSettings.cleaning_intensity;
            elements.intensityValue.textContent = savedSettings.cleaning_intensity;
        }
    }
}

/**
 * Save settings to storage
 */
function saveSettings() {
    const utils = getUtils();
    const settings = {
        model: elements.modelSelect.value,
        language: elements.languageSelect.value,
        cleaning_intensity: parseInt(elements.cleaningIntensity.value)
    };
    utils.storage.set('uploadSettings', settings);
}

// ===========================
// Event Listeners
// ===========================

/**
 * Initialize event listeners
 */
function initEventListeners() {
    // File selection button
    elements.selectFileBtn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.fileInput.click();
    });

    // File input change
    elements.fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });

    // Remove file button
    elements.fileRemove.addEventListener('click', () => {
        clearFileSelection();
    });

    // Cleaning intensity slider
    elements.cleaningIntensity.addEventListener('input', (e) => {
        elements.intensityValue.textContent = e.target.value;
    });

    // Save settings on change
    elements.modelSelect.addEventListener('change', saveSettings);
    elements.languageSelect.addEventListener('change', saveSettings);
    elements.cleaningIntensity.addEventListener('change', saveSettings);

    // Reset button
    elements.resetBtn.addEventListener('click', resetForm);

    // Start transcription button
    elements.startBtn.addEventListener('click', startTranscription);

    // Click on upload zone to select file
    elements.uploadZone.addEventListener('click', (e) => {
        // Don't trigger if clicking the button or inside it
        if (e.target === elements.selectFileBtn || 
            elements.selectFileBtn.contains(e.target) ||
            e.target === elements.fileInput) {
            return;
        }
        elements.fileInput.click();
    });
}

// ===========================
// Initialization
// ===========================

/**
 * Initialize upload page
 */
function initUploadPage() {
    console.log('Initializing upload page...');

    // Check if utilities are loaded
    if (!window.LectureTranscriber) {
        console.error('LectureTranscriber utilities not loaded!');
        return;
    }

    // Initialize DOM elements
    elements = {
        uploadZone: document.getElementById('upload-zone'),
        fileInput: document.getElementById('file-input'),
        selectFileBtn: document.getElementById('select-file-btn'),
        fileInfo: document.getElementById('file-info'),
        fileName: document.getElementById('file-name'),
        fileSize: document.getElementById('file-size'),
        fileRemove: document.getElementById('file-remove'),
        uploadProgress: document.getElementById('upload-progress'),
        progressBar: document.getElementById('progress-bar'),
        progressPercent: document.getElementById('progress-percent'),
        settingsForm: document.getElementById('settings-form'),
        modelSelect: document.getElementById('model-select'),
        languageSelect: document.getElementById('language-select'),
        cleaningIntensity: document.getElementById('cleaning-intensity'),
        intensityValue: document.getElementById('intensity-value'),
        resetBtn: document.getElementById('reset-btn'),
        startBtn: document.getElementById('start-transcription-btn')
    };

    // Verify all elements exist
    for (const [key, element] of Object.entries(elements)) {
        if (!element) {
            console.error(`Element not found: ${key}`);
        }
    }

    // Initialize drag and drop
    initDragAndDrop();

    // Initialize event listeners
    initEventListeners();

    // Load saved settings
    loadSavedSettings();

    // Check for pending task
    const utils = getUtils();
    const currentTask = utils.storage.get('currentTask');
    if (currentTask && currentTask.taskId) {
        // Show notification about pending task
        utils.toast.info('У вас есть незавершённая транскрипция', 5000);
    }

    console.log('Upload page initialized successfully');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initUploadPage);
} else {
    initUploadPage();
}
