// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');

// API base URL - adjust if needed
const API_BASE_URL = '';

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    setupEventListeners();
    updateConfidenceDisplay();
});

function setupEventListeners() {
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Confidence slider
    confidenceSlider.addEventListener('input', updateConfidenceDisplay);
    
    // Prevent default drag behaviors on the whole document
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        processImage(files[0]);
    } else {
        alert('Please drop a valid image file');
    }
}

// File selection handler
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processImage(files[0]);
    }
}

// Update confidence display
function updateConfidenceDisplay() {
    confidenceValue.textContent = confidenceSlider.value;
}

// Main image processing function
async function processImage(file) {
    // Validate file
    if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
    }
    
    // Show loading state
    showLoading();
    
    try {
        // Display original image
        displayOriginalImage(file);
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        // Get confidence threshold
        const confidence = parseFloat(confidenceSlider.value);
        
        // Make API call
        const response = await fetch(`${API_BASE_URL}/predict?confidence=${confidence}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to process image');
        }
        
        const data = await response.json();
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Error processing image:', error);
        hideLoading();
        alert(`Error processing image: ${error.message}`);
    }
}

// Display original image
function displayOriginalImage(file) {
    const originalImg = document.getElementById('originalImage');
    originalImg.src = URL.createObjectURL(file);
}

// Show loading state
function showLoading() {
    loading.style.display = 'block';
    results.style.display = 'none';
}

// Hide loading state
function hideLoading() {
    loading.style.display = 'none';
}

// Display detection results
function displayResults(data) {
    hideLoading();
    results.style.display = 'block';
    
    // Display annotated image
    const annotatedImg = document.getElementById('annotatedImage');
    annotatedImg.src = `data:image/jpeg;base64,${data.annotated_image}`;
    
    // Update summary title
    const summaryTitle = document.getElementById('summaryTitle');
    if (data.total_defects > 0) {
        summaryTitle.textContent = `🚨 ${data.total_defects} Defect${data.total_defects > 1 ? 's' : ''} Found`;
        summaryTitle.style.color = '#dc3545';
    } else {
        summaryTitle.textContent = '✅ No Defects Detected';
        summaryTitle.style.color = '#28a745';
    }
    
    // Display detection statistics
    displayDetectionStats(data);
    
    // Display detection list
    displayDetectionList(data.detections);
    
    // Display model info
    displayModelInfo(data.model_info);
}

// Display detection statistics
function displayDetectionStats(data) {
    const statsContainer = document.getElementById('detectionStats');
    
    const stats = [
        { label: 'Total Defects', value: data.total_defects, color: '#dc3545' },
        { label: 'High Confidence', value: data.detections.filter(d => d.confidence > 0.7).length, color: '#ffc107' },
        { label: 'Model Confidence', value: data.model_info ? 'Loaded' : 'Unknown', color: '#28a745' }
    ];
    
    statsContainer.innerHTML = stats.map(stat => `
        <div class="stat-card">
            <div class="stat-value" style="color: ${stat.color}">${stat.value}</div>
            <div class="stat-label">${stat.label}</div>
        </div>
    `).join('');
}

// Display detection list
function displayDetectionList(detections) {
    const listContainer = document.getElementById('detectionList');
    
    if (detections.length === 0) {
        listContainer.innerHTML = `
            <div class="no-defects">
                <h4>🎉 PCB Quality Check Passed!</h4>
                <p>No manufacturing defects detected in this PCB image.</p>
            </div>
        `;
        return;
    }
    
    // Sort by confidence (highest first)
    detections.sort((a, b) => b.confidence - a.confidence);
    
    const detectionsHTML = detections.map((detection, index) => {
        const confidence = detection.confidence;
        const confidenceClass = confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low';
        const badgeClass = confidence > 0.7 ? 'confidence-high' : confidence > 0.4 ? 'confidence-medium' : 'confidence-low';
        
        const bbox = detection.bbox.map(coord => Math.round(coord));
        
        return `
            <div class="detection-item ${confidenceClass}-confidence">
                <div class="detection-header">
                    <span class="detection-label">${formatDefectLabel(detection.label)}</span>
                    <span class="confidence-badge ${badgeClass}">
                        ${(confidence * 100).toFixed(1)}%
                    </span>
                </div>
                <div class="detection-details">
                    <strong>Location:</strong> (${bbox[0]}, ${bbox[1]}) → (${bbox[2]}, ${bbox[3]})<br>
                    <strong>Size:</strong> ${bbox[2] - bbox[0]} × ${bbox[3] - bbox[1]} pixels<br>
                    <strong>Severity:</strong> ${getSeverityLabel(confidence)}
                </div>
            </div>
        `;
    }).join('');
    
    listContainer.innerHTML = detectionsHTML;
}

// Display model information
function displayModelInfo(modelInfo) {
    const modelContainer = document.getElementById('modelInfo');
    
    if (!modelInfo) {
        modelContainer.innerHTML = '<p>Model information not available</p>';
        return;
    }
    
    const infoHTML = `
        <p><strong>Model:</strong> ${modelInfo.model_name || 'YOLOv8'}</p>
        <p><strong>Classes:</strong> ${modelInfo.num_classes || 'Unknown'} defect types</p>
        <p><strong>Status:</strong> ${modelInfo.is_custom ? 'Custom Trained' : 'Pretrained'}</p>
        <p><strong>Confidence Threshold:</strong> ${confidenceSlider.value}</p>
    `;
    
    modelContainer.innerHTML = infoHTML;
}

// Utility functions
function formatDefectLabel(label) {
    return label
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

function getSeverityLabel(confidence) {
    if (confidence > 0.8) return 'Critical';
    if (confidence > 0.6) return 'High';
    if (confidence > 0.4) return 'Medium';
    return 'Low';
}

// Export functions for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        formatDefectLabel,
        getSeverityLabel,
        processImage
    };
}