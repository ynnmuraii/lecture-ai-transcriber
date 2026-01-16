/**
 * Lecture Transcriber - Results Page
 * Handles displaying transcription results, markdown rendering, and navigation
 */

// Import utilities from app.js
const { toast, loading, api, storage, formatDuration, formatTimestamp, pollTaskStatus } = window.LectureTranscriber;

// ===========================
// Results Manager
// ===========================
class ResultsManager {
    constructor() {
        this.taskId = null;
        this.resultData = null;
        this.segments = [];
        this.currentSegmentIndex = 0;
        
        // DOM elements
        this.statusBanner = document.getElementById('status-banner');
        this.statusText = document.getElementById('status-text');
        this.statusMessage = document.getElementById('status-message');
        this.statusProgressBar = document.getElementById('status-progress-bar');
        this.taskIdDisplay = document.getElementById('task-id-display');
        
        this.actionBar = document.getElementById('action-bar');
        this.metadataDate = document.getElementById('metadata-date');
        this.metadataDuration = document.getElementById('metadata-duration');
        this.metadataSegments = document.getElementById('metadata-segments');
        
        this.contentLayout = document.getElementById('content-layout');
        this.contentEmpty = document.getElementById('content-empty');
        this.markdownContent = document.getElementById('markdown-content');
        this.timestampList = document.getElementById('timestamp-list');
        
        this.downloadJsonBtn = document.getElementById('download-json-btn');
        this.downloadMdBtn = document.getElementById('download-md-btn');
        
        this.init();
    }

    /**
     * Initialize the results page
     */
    init() {
        // Get task ID from URL
        this.taskId = this.getTaskIdFromUrl();
        
        if (!this.taskId) {
            this.showError('Task ID не найден в URL');
            return;
        }
        
        this.taskIdDisplay.textContent = `Task ID: ${this.taskId}`;
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Start loading results
        this.loadResults();
    }

    /**
     * Get task ID from URL parameters
     * @returns {string|null} - Task ID or null
     */
    getTaskIdFromUrl() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('task_id');
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Download buttons
        this.downloadJsonBtn.addEventListener('click', () => this.downloadFile('json'));
        this.downloadMdBtn.addEventListener('click', () => this.downloadFile('md'));
    }

    /**
     * Load results and poll for status
     */
    async loadResults() {
        try {
            // Poll for task status
            await pollTaskStatus(
                this.taskId,
                (status) => this.updateStatus(status),
                2000
            );
            
            // Task completed, load full results
            await this.loadFullResults();
            
        } catch (error) {
            console.error('Error loading results:', error);
            this.showError(error.message);
        }
    }

    /**
     * Update status display
     * @param {object} status - Status response
     */
    updateStatus(status) {
        // Update progress bar
        this.statusProgressBar.style.width = `${status.progress}%`;
        
        // Update status text
        this.statusText.textContent = this.getStatusText(status.status);
        this.statusMessage.textContent = status.message || '';
        
        // Update banner style
        this.statusBanner.className = `status-banner ${status.status}`;
        
        // Update icon
        const icon = this.statusBanner.querySelector('.status-icon');
        icon.textContent = this.getStatusIcon(status.status);
    }

    /**
     * Get status text
     * @param {string} status - Status code
     * @returns {string} - Human-readable status
     */
    getStatusText(status) {
        const statusMap = {
            'pending': 'Ожидание',
            'processing': 'Обработка',
            'completed': 'Завершено',
            'failed': 'Ошибка'
        };
        return statusMap[status] || status;
    }

    /**
     * Get status icon
     * @param {string} status - Status code
     * @returns {string} - Icon emoji
     */
    getStatusIcon(status) {
        const iconMap = {
            'pending': '⏳',
            'processing': '⚙️',
            'completed': '✅',
            'failed': '❌'
        };
        return iconMap[status] || '❓';
    }

    /**
     * Load full results
     */
    async loadFullResults() {
        try {
            loading.show('Загрузка результатов...');
            
            const result = await api.getResult(this.taskId);
            this.resultData = result;
            
            if (result.status === 'completed' && result.content) {
                this.displayResults(result);
            } else {
                this.showError('Результаты не готовы');
            }
            
        } catch (error) {
            console.error('Error loading full results:', error);
            toast.error('Не удалось загрузить результаты');
        } finally {
            loading.hide();
        }
    }

    /**
     * Display results
     * @param {object} result - Result response
     */
    displayResults(result) {
        // Hide status banner progress
        const progressDiv = this.statusBanner.querySelector('.status-progress');
        if (progressDiv) {
            progressDiv.style.display = 'none';
        }
        
        // Show action bar
        this.actionBar.classList.remove('hidden');
        
        // Update metadata
        if (result.metadata) {
            this.updateMetadata(result.metadata);
        }
        
        // Show content layout
        this.contentLayout.classList.remove('hidden');
        
        // Extract segments from content or use provided segments
        if (result.segments && result.segments.length > 0) {
            this.segments = result.segments;
        } else {
            // Try to extract segments from markdown content
            this.segments = this.extractSegmentsFromMarkdown(result.content);
        }
        
        // Render markdown content
        this.renderMarkdown(result.content);
        
        // Build timestamp navigation
        if (this.segments.length > 0) {
            this.buildTimestampNavigation();
        }
        
        toast.success('Результаты загружены');
    }

    /**
     * Update metadata display
     * @param {object} metadata - Metadata object
     */
    updateMetadata(metadata) {
        if (metadata.created_at) {
            this.metadataDate.textContent = formatTimestamp(metadata.created_at);
        }
        
        if (metadata.duration) {
            this.metadataDuration.textContent = formatDuration(metadata.duration);
        }
        
        if (metadata.segment_count) {
            this.metadataSegments.textContent = `${metadata.segment_count} сегментов`;
        } else if (this.segments.length > 0) {
            this.metadataSegments.textContent = `${this.segments.length} сегментов`;
        }
    }

    /**
     * Extract segments from markdown content
     * @param {string} content - Markdown content
     * @returns {Array} - Array of segments
     */
    extractSegmentsFromMarkdown(content) {
        const segments = [];
        
        // Look for timestamp markers in format [HH:MM:SS] or [MM:SS]
        const timestampRegex = /\[(\d{1,2}):(\d{2})(?::(\d{2}))?\]/g;
        let match;
        let lastIndex = 0;
        
        while ((match = timestampRegex.exec(content)) !== null) {
            const hours = match[3] ? parseInt(match[1]) : 0;
            const minutes = match[3] ? parseInt(match[2]) : parseInt(match[1]);
            const seconds = match[3] ? parseInt(match[3]) : parseInt(match[2]);
            
            const startTime = hours * 3600 + minutes * 60 + seconds;
            
            // Extract text until next timestamp or end
            const startPos = match.index + match[0].length;
            const nextMatch = timestampRegex.exec(content);
            const endPos = nextMatch ? nextMatch.index : content.length;
            
            // Reset regex for next iteration
            if (nextMatch) {
                timestampRegex.lastIndex = nextMatch.index;
            }
            
            const text = content.substring(startPos, endPos).trim();
            
            if (text) {
                segments.push({
                    text: text.substring(0, 100), // Preview
                    start_time: startTime,
                    end_time: startTime + 60 // Estimate
                });
            }
        }
        
        return segments;
    }

    /**
     * Render markdown content
     * @param {string} content - Markdown content
     */
    renderMarkdown(content) {
        if (!content) {
            return;
        }
        
        // Hide empty state
        this.contentEmpty.classList.add('hidden');
        this.markdownContent.classList.remove('hidden');
        
        // Convert markdown to HTML
        const html = this.markdownToHtml(content);
        this.markdownContent.innerHTML = html;
        
        // Add click handlers to timestamp markers
        this.addTimestampClickHandlers();
    }

    /**
     * Convert markdown to HTML
     * @param {string} markdown - Markdown text
     * @returns {string} - HTML string
     */
    markdownToHtml(markdown) {
        let html = markdown;
        
        // Headers
        html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
        html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
        html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');
        
        // Bold
        html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic
        html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
        
        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Code blocks
        html = html.replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>');
        
        // Blockquotes
        html = html.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');
        
        // Unordered lists
        html = html.replace(/^\* (.*$)/gim, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        
        // Ordered lists
        html = html.replace(/^\d+\. (.*$)/gim, '<li>$1</li>');
        
        // Timestamps - convert to clickable markers
        html = html.replace(/\[(\d{1,2}:\d{2}(?::\d{2})?)\]/g, 
            '<span class="timestamp-marker" data-timestamp="$1">$1</span>');
        
        // Paragraphs
        html = html.split('\n\n').map(para => {
            if (!para.match(/^<[h|u|o|l|b|p]/)) {
                return `<p>${para}</p>`;
            }
            return para;
        }).join('\n');
        
        return html;
    }

    /**
     * Add click handlers to timestamp markers
     */
    addTimestampClickHandlers() {
        const markers = this.markdownContent.querySelectorAll('.timestamp-marker');
        markers.forEach(marker => {
            marker.addEventListener('click', (e) => {
                const timestamp = e.target.dataset.timestamp;
                this.scrollToTimestamp(timestamp);
            });
        });
    }

    /**
     * Build timestamp navigation
     */
    buildTimestampNavigation() {
        this.timestampList.innerHTML = '';
        
        this.segments.forEach((segment, index) => {
            const li = document.createElement('li');
            li.className = 'timestamp-item';
            li.dataset.index = index;
            
            const timeSpan = document.createElement('span');
            timeSpan.className = 'timestamp-time';
            timeSpan.textContent = formatDuration(segment.start_time);
            
            const previewSpan = document.createElement('span');
            previewSpan.className = 'timestamp-preview';
            previewSpan.textContent = segment.text.substring(0, 50) + '...';
            
            li.appendChild(timeSpan);
            li.appendChild(previewSpan);
            
            // Click handler
            li.addEventListener('click', () => {
                this.navigateToSegment(index);
            });
            
            this.timestampList.appendChild(li);
        });
    }

    /**
     * Navigate to a specific segment
     * @param {number} index - Segment index
     */
    navigateToSegment(index) {
        this.currentSegmentIndex = index;
        
        // Update active state in navigation
        const items = this.timestampList.querySelectorAll('.timestamp-item');
        items.forEach((item, i) => {
            if (i === index) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        
        // Scroll to corresponding content
        const segment = this.segments[index];
        if (segment) {
            const timestamp = formatDuration(segment.start_time);
            this.scrollToTimestamp(timestamp);
        }
    }

    /**
     * Scroll to timestamp in content
     * @param {string} timestamp - Timestamp string
     */
    scrollToTimestamp(timestamp) {
        const markers = this.markdownContent.querySelectorAll('.timestamp-marker');
        
        for (const marker of markers) {
            if (marker.dataset.timestamp === timestamp || marker.textContent === timestamp) {
                marker.scrollIntoView({ behavior: 'smooth', block: 'center' });
                
                // Highlight temporarily
                marker.style.backgroundColor = 'var(--warning-color)';
                setTimeout(() => {
                    marker.style.backgroundColor = 'var(--primary-color)';
                }, 1000);
                
                break;
            }
        }
    }

    /**
     * Download file
     * @param {string} format - File format ('md' or 'json')
     */
    downloadFile(format) {
        if (!this.taskId) {
            toast.error('Task ID не найден');
            return;
        }
        
        const url = api.getDownloadUrl(this.taskId, format);
        
        // Create temporary link and trigger download
        const link = document.createElement('a');
        link.href = url;
        link.download = `transcription_${this.taskId}.${format}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        toast.success(`Скачивание ${format.toUpperCase()} файла начато`);
    }

    /**
     * Show error state
     * @param {string} message - Error message
     */
    showError(message) {
        this.statusBanner.className = 'status-banner failed';
        this.statusText.textContent = 'Ошибка';
        this.statusMessage.textContent = message;
        
        const icon = this.statusBanner.querySelector('.status-icon');
        icon.textContent = '❌';
        
        const progressDiv = this.statusBanner.querySelector('.status-progress');
        if (progressDiv) {
            progressDiv.style.display = 'none';
        }
        
        toast.error(message);
    }
}

// ===========================
// Initialize on DOM Ready
// ===========================
document.addEventListener('DOMContentLoaded', () => {
    new ResultsManager();
});
