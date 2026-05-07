let suspicionChart = null;

document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('video-upload');
    const queueList = document.getElementById('queue-list');
    const queueCount = document.getElementById('queue-count');
    const videoPlayer = document.getElementById('video-player');
    const loader = document.getElementById('loader');
    const loaderFilename = document.getElementById('loader-filename');
    
    // Stats Elements
    const progressCircle = document.getElementById('progress-circle');
    const probText = document.getElementById('prob-text');
    const probLabel = document.getElementById('prob-label');
    const overallStatus = document.getElementById('overall-status');
    const statusText = document.getElementById('status-text');
    const fakeVal = document.getElementById('fake-val');
    const realVal = document.getElementById('real-val');
    const framesVal = document.getElementById('frames-val');
    const sizeVal = document.getElementById('size-val');
    const heatmapContainer = document.getElementById('heatmap-container');

    let videoQueue = [];
    let isProcessing = false;
    window._lastResult = null;
    window._currentFilename = '';

    // Initialization
    initChart();

    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.background = 'rgba(0,217,255,0.1)'; });
    dropZone.addEventListener('dragleave', () => { dropZone.style.background = 'rgba(0,0,0,0.2)'; });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.background = 'rgba(0,0,0,0.2)';
        addFilesToQueue(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', (e) => {
        addFilesToQueue(e.target.files);
    });

    function addFilesToQueue(files) {
        let added = 0;
        for(let i=0; i<files.length; i++){
            if(files[i].type.startsWith('video/')){
                videoQueue.push({ file: files[i], status: 'pending' });
                added++;
            }
        }
        if(added > 0) updateQueueUI();
        if(!isProcessing) processNextInQueue();
    }

    function updateQueueUI() {
        queueCount.textContent = videoQueue.length;
        if(videoQueue.length === 0){
            queueList.innerHTML = '<li class="placeholder-text">No videos in queue</li>';
            return;
        }
        queueList.innerHTML = '';
        videoQueue.forEach((item, index) => {
            const li = document.createElement('li');
            li.className = `queue-item ${item.status}`;
            li.innerHTML = `<span>${item.file.name}</span> <span>${item.status.toUpperCase()}</span>`;
            queueList.appendChild(li);
        });
    }

    async function processNextInQueue() {
        const pendingIndex = videoQueue.findIndex(v => v.status === 'pending');
        if(pendingIndex === -1) {
            isProcessing = false;
            return;
        }

        isProcessing = true;
        const item = videoQueue[pendingIndex];
        item.status = 'active';
        updateQueueUI();

        // Show Video in Player
        const fileURL = URL.createObjectURL(item.file);
        videoPlayer.src = fileURL;

        loader.classList.remove('hidden');
        loaderFilename.textContent = `Processing: ${item.file.name}`;

        const formData = new FormData();
        formData.append('video', item.file);

        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            if (!response.ok) throw new Error("Server Error");
            const data = await response.json();
            if (data.error) throw new Error(data.error);

            updateDashboard(data, item.file.name);
            item.status = 'done';
            window._lastResult = data;
            window._currentFilename = item.file.name;
            
        } catch (error) {
            console.error(error);
            item.status = 'error';
            overallStatus.style.borderColor = 'var(--accent-pink)';
            statusText.textContent = 'ANALYSIS FAILED';
        } finally {
            loader.classList.add('hidden');
            updateQueueUI();
            processNextInQueue(); // Next item
        }
    }

    function updateDashboard(data, filename) {
        const fakeThresh    = data.threshold_used    || 0.55;
        const suspectThresh = data.suspect_threshold || 0.38;
        const fakePct       = (data.fake_prob * 100).toFixed(1);
        const realPct       = (data.real_prob * 100).toFixed(1);

        // ── 3-tier classification ──────────────────────────────
        let verdict, color, bgColor, icon;

        if (data.fake_prob >= fakeThresh) {
            verdict = 'DEEPFAKE DETECTED';
            color   = '#ff006e';
            bgColor = 'rgba(255,0,110,0.1)';
            icon    = '🎭';
        } else if (data.fake_prob >= suspectThresh) {
            verdict = 'SUSPICIOUS — Possible Filter/Overlay';
            color   = '#ffaa00';
            bgColor = 'rgba(255,170,0,0.1)';
            icon    = '⚠️';
        } else {
            verdict = 'REAL VIDEO';
            color   = '#22c55e';
            bgColor = 'rgba(34,197,94,0.1)';
            icon    = '✅';
        }

        // Circular Progress
        progressCircle.style.background = `conic-gradient(${color} ${fakePct * 3.6}deg, #333 0deg)`;
        probText.textContent = `${fakePct}%`;
        probText.style.color = color;
        probLabel.textContent = data.fake_prob >= fakeThresh ? 'FAKE' :
                                data.fake_prob >= suspectThresh ? 'SUSPECT' : 'REAL';

        // Status badge
        overallStatus.style.borderColor = color;
        overallStatus.style.background  = bgColor;
        statusText.textContent = `${icon} ${verdict}`;
        statusText.style.color = color;

        fakeVal.textContent   = `${fakePct}%`;
        realVal.textContent   = `${realPct}%`;
        framesVal.textContent = `${data.frames_analyzed} frames`;
        sizeVal.textContent   = `${data.file_size_mb} MB`;

        // Heatmap
        if (data.overlay_b64) {
            heatmapContainer.innerHTML = `<img src="data:image/png;base64,${data.overlay_b64}" alt="Heatmap">`;
        }

        updateChart(data.frame_scores);

        // Show feedback card
        const feedbackCard = document.getElementById('feedback-card');
        feedbackCard.style.display      = 'flex';
        feedbackCard.style.flexDirection = 'column';
        document.getElementById('feedback-label-picker').style.display = 'none';
        document.getElementById('feedback-thankyou').style.display     = 'none';
        document.getElementById('btn-correct').disabled = false;
        document.getElementById('btn-wrong').disabled   = false;
    }

    function initChart() {
        const ctx = document.getElementById('suspicionChart').getContext('2d');
        suspicionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Array.from({length: 16}, (_, i) => `F${i+1}`),
                datasets: [{
                    label: 'Suspicion Score',
                    data: Array(16).fill(0),
                    backgroundColor: [], // Set dynamically
                    borderRadius: 4,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                yMin: 50,
                                yMax: 50,
                                borderColor: 'rgba(255, 255, 255, 0.2)',
                                borderWidth: 1,
                                borderDash: [4, 4],
                            }
                        }
                    }
                },
                scales: {
                    y: { 
                        beginAtZero: true, 
                        max: 100,
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: 'rgba(255,255,255,0.5)', stepSize: 25 }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: 'rgba(255,255,255,0.5)' }
                    }
                }
            }
        });
    }

    function updateChart(scores) {
        const colors = scores.map(score => {
            if (score > 85) return '#ff006e';
            if (score > 50) return '#ffaa00';
            return '#00d9ff';
        });
        suspicionChart.data.datasets[0].data = scores;
        suspicionChart.data.datasets[0].backgroundColor = colors;
        suspicionChart.update();
    }
});

// ── Global Feedback Functions ────────────────────────────────────────────────
function submitFeedback(isCorrect) {
    document.getElementById('btn-correct').disabled = true;
    document.getElementById('btn-wrong').disabled = true;

    if (isCorrect) {
        confirmFeedback(null, true);
    } else {
        // Show label picker
        document.getElementById('feedback-label-picker').style.display = 'block';
    }
}

async function confirmFeedback(actualLabel, isCorrect = false) {
    const lastResult = window._lastResult;
    const filename = window._currentFilename;
    if (!lastResult) return;

    const threshold = lastResult.threshold_used || 0.35;
    const modelPrediction = lastResult.fake_prob > threshold ? 'FAKE' : 'REAL';
    const trueLabel = isCorrect ? modelPrediction : actualLabel;

    try {
        const res = await fetch('/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: filename || 'unknown',
                model_prediction: modelPrediction,
                actual_label: trueLabel,
                fake_prob: lastResult.fake_prob,
                real_prob: lastResult.real_prob,
                is_correct: isCorrect || (modelPrediction === trueLabel),
            })
        });
        const stats = await res.json();

        document.getElementById('feedback-label-picker').style.display = 'none';
        document.getElementById('feedback-thankyou').style.display = 'block';
        document.getElementById('feedback-stats-text').textContent =
            `Total feedback: ${stats.total_feedback} | Model accuracy (user-rated): ${stats.model_accuracy_so_far}%`;
    } catch(e) {
        console.error('Feedback failed:', e);
    }
}
