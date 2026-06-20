// --- APP STATE ---
let isTraining = false;
let socket = null;
let currentChart = null;
let activeTab = 'acc'; // 'acc' or 'loss'
let selectedImagePath = '';
let metricsHistory = [];
let clientDistributions = {};

// Canvas animation state
const canvas = document.getElementById('network-canvas');
const ctx = canvas.getContext('2d');
let animationFrameId = null;
let topologyNodes = {};
let particles = [];
let serverPulseRadius = 0;
let serverPulseActive = false;

// --- CONFIGURATION MANAGEMENT ---
const configInputs = {
    rounds: document.getElementById('input-rounds'),
    epochs: document.getElementById('input-epochs'),
    lr: document.getElementById('input-lr'),
    distribution: document.getElementById('input-distribution'),
    dpEnabled: document.getElementById('toggle-dp'),
    noise: document.getElementById('input-noise'),
    clip: document.getElementById('input-clip')
};

const valueDisplays = {
    rounds: document.getElementById('val-rounds'),
    epochs: document.getElementById('val-epochs'),
    lr: document.getElementById('val-lr'),
    noise: document.getElementById('val-noise'),
    clip: document.getElementById('val-clip')
};

// Bind sliders to value displays
Object.keys(valueDisplays).forEach(key => {
    configInputs[key].addEventListener('input', (e) => {
        valueDisplays[key].textContent = e.target.value;
        saveConfig();
    });
});

configInputs.distribution.addEventListener('change', saveConfig);
configInputs.dpEnabled.addEventListener('change', (e) => {
    const dpPanel = document.getElementById('dp-config-panel');
    if (e.target.checked) {
        dpPanel.classList.remove('collapsed');
    } else {
        dpPanel.classList.add('collapsed');
    }
    
    // Update local client cards privacy badges
    const shieldStatus = e.target.checked ? "DP Enabled" : "DP Disabled";
    const shieldFill = document.getElementById('stat-ha-shield-fill');
    
    ['ha', 'hb', 'hc'].forEach(id => {
        const shieldEl = document.getElementById(`stat-${id}-shield`);
        const fillEl = document.getElementById(`stat-${id}-shield-fill`);
        shieldEl.textContent = shieldStatus;
        if (e.target.checked) {
            fillEl.className = "meter-fill fill-secured";
            fillEl.style.width = "100%";
        } else {
            fillEl.className = "meter-fill fill-leak";
            fillEl.style.width = "100%";
        }
    });
    
    saveConfig();
});

// Save config to FastAPI server
function saveConfig() {
    const payload = {
        num_rounds: parseInt(configInputs.rounds.value),
        local_epochs: parseInt(configInputs.epochs.value),
        learning_rate: parseFloat(configInputs.lr.value),
        data_distribution: configInputs.distribution.value,
        dp_enabled: configInputs.dpEnabled.checked,
        dp_noise_multiplier: parseFloat(configInputs.noise.value),
        dp_clip_norm: parseFloat(configInputs.clip.value)
    };
    
    fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .catch(err => console.error("Error saving config:", err));
}

// --- CONSOLE LOGGER ---
const consoleLogs = document.getElementById('console-logs');
document.getElementById('btn-clear-console').addEventListener('click', () => {
    consoleLogs.innerHTML = '';
});

function logToConsole(message, type = 'info') {
    const timeString = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = `log-line ${type}-line`;
    
    let prefix = `[${timeString}]`;
    if (type === 'round') prefix += ` [Round]`;
    else if (type === 'system') prefix += ` [System]`;
    else if (type === 'error') prefix += ` [ERROR]`;
    
    line.textContent = `${prefix} ${message}`;
    consoleLogs.appendChild(line);
    consoleLogs.scrollTop = consoleLogs.scrollHeight;
}

// --- TAB NAV FOR CHARTS ---
document.getElementById('tab-acc').addEventListener('click', (e) => {
    switchTab('acc');
});
document.getElementById('tab-loss').addEventListener('click', (e) => {
    switchTab('loss');
});

function switchTab(tab) {
    activeTab = tab;
    document.getElementById('tab-acc').classList.toggle('active', tab === 'acc');
    document.getElementById('tab-loss').classList.toggle('active', tab === 'loss');
    updateCharts();
}

// --- WEBSOCKET CONNECTION ---
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    socket = new WebSocket(wsUrl);
    
    socket.onopen = () => {
        logToConsole("Connected to central coordination server.", "system");
    };
    
    socket.onclose = () => {
        logToConsole("Disconnected from server. Reconnecting in 3s...", "error");
        setTimeout(connectWebSocket, 3000);
    };
    
    socket.onerror = (err) => {
        console.error("WS Error:", err);
    };
    
    socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };
}

function handleServerMessage(data) {
    switch (data.type) {
        case 'status':
            // Set initial state
            updateSystemStatus(data.status);
            metricsHistory = data.metrics_history || [];
            updateClientCards(data.clients);
            
            // Populate slider values from server
            if (data.params) {
                configInputs.rounds.value = data.params.num_rounds;
                valueDisplays.rounds.textContent = data.params.num_rounds;
                configInputs.epochs.value = data.params.local_epochs;
                valueDisplays.epochs.textContent = data.params.local_epochs;
                configInputs.lr.value = data.params.learning_rate;
                valueDisplays.lr.textContent = data.params.learning_rate;
                configInputs.distribution.value = data.params.data_distribution;
                configInputs.dpEnabled.checked = data.params.dp_enabled;
                configInputs.noise.value = data.params.dp_noise_multiplier;
                valueDisplays.noise.textContent = data.params.dp_noise_multiplier;
                configInputs.clip.value = data.params.dp_clip_norm;
                valueDisplays.clip.textContent = data.params.dp_clip_norm;
                
                // Toggle DP collapse
                const dpPanel = document.getElementById('dp-config-panel');
                if (data.params.dp_enabled) dpPanel.classList.remove('collapsed');
                else dpPanel.classList.add('collapsed');
            }
            updateCharts();
            break;
            
        case 'data_generated':
            logToConsole(data.message, "system");
            updateClientCards(data.clients);
            loadSampleGallery();
            updateSystemStatus("idle");
            break;
            
        case 'info':
            logToConsole(data.message, "info");
            break;
            
        case 'init':
            logToConsole("Federated Simulation Initialized.", "system");
            metricsHistory = [];
            // Push initial round 0 evaluation
            if (data.metrics) {
                metricsHistory.push({
                    round: 0,
                    metrics: data.metrics,
                    clients: {}
                });
            }
            updateCharts();
            updateSystemStatus("training");
            break;
            
        case 'round_start':
            logToConsole(data.message, "round");
            serverPulseActive = false;
            particles = [];
            
            // Animate weights broadcast: Server -> Clients
            sendPackets('server', 'clients');
            break;
            
        case 'client_start':
            logToConsole(data.message, "info");
            setClientIndicator(data.client, "training");
            break;
            
        case 'client_complete':
            logToConsole(`${data.client.replace('_', ' ').toUpperCase()} finished training. Update Norm: ${data.metrics.update_norm.toFixed(4)}`, "info");
            setClientIndicator(data.client, "online");
            updateSingleClientStats(data.client, data.metrics);
            
            // Animate upload: Client -> Server
            sendPackets(data.client, 'server');
            break;
            
        case 'aggregation_start':
            logToConsole(data.message, "info");
            serverPulseActive = true;
            serverPulseRadius = 0;
            break;
            
        case 'round_complete':
            logToConsole(data.message, "round");
            serverPulseActive = false;
            metricsHistory.push({
                round: data.round,
                metrics: data.metrics,
                clients: data.clients
            });
            updateCharts();
            
            // Enable inference test if accuracy is set
            document.getElementById('btn-predict').disabled = false;
            break;
            
        case 'error':
            logToConsole(data.message, "error");
            if (data.detail) console.error(data.detail);
            updateSystemStatus("idle");
            break;
    }
}

// --- SYSTEM CONTROLS ---
document.getElementById('btn-start').addEventListener('click', () => {
    fetch('/api/start-training', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'started') {
                updateSystemStatus("training");
            }
        });
});

document.getElementById('btn-stop').addEventListener('click', () => {
    fetch('/api/stop-training', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'stopped') {
                updateSystemStatus("stopped");
            }
        });
});

document.getElementById('btn-generate-data').addEventListener('click', () => {
    fetch('/api/generate-data', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'started') {
                updateSystemStatus("generating_data");
            }
        });
});

function updateSystemStatus(status) {
    isTraining = (status === 'training');
    
    // Update badge UI
    const badge = document.getElementById('system-status-badge');
    badge.className = `status-badge status-${status}`;
    badge.textContent = status.replace('_', ' ').toUpperCase();
    
    // Toggle button availabilities
    document.getElementById('btn-start').disabled = isTraining || status === 'generating_data';
    document.getElementById('btn-stop').disabled = !isTraining;
    document.getElementById('btn-generate-data').disabled = isTraining || status === 'generating_data';
    
    // Disable/Enable configurations
    Object.values(configInputs).forEach(input => {
        input.disabled = isTraining || status === 'generating_data';
    });
    
    if (status === 'completed' || status === 'stopped' || status === 'idle') {
        // Reset client indicators to standard online
        ['hospital_a', 'hospital_b', 'hospital_c'].forEach(client => {
            setClientIndicator(client, "online");
        });
    }
}

function setClientIndicator(client, state) {
    const card = document.getElementById(`card-${client}`);
    if (!card) return;
    const indicator = card.querySelector('.client-indicator');
    if (indicator) {
        indicator.className = `client-indicator ${state}`;
    }
}

function updateClientCards(clients) {
    if (!clients) return;
    clientDistributions = clients;
    
    const mapping = {
        "hospital_a": "ha",
        "hospital_b": "hb",
        "hospital_c": "hc"
    };
    
    Object.keys(clients).forEach(client => {
        const id = mapping[client];
        const dist = clients[client].distribution;
        if (!dist) return;
        
        document.getElementById(`stat-${id}-total`).textContent = `${dist.total} Slices`;
        document.getElementById(`stat-${id}-dist`).textContent = `${(dist.tumor_ratio * 100).toFixed(0)}% Tumor / ${((1 - dist.tumor_ratio)*100).toFixed(0)}% Normal`;
    });
}

function updateSingleClientStats(client, metrics) {
    const mapping = {
        "hospital_a": "ha",
        "hospital_b": "hb",
        "hospital_c": "hc"
    };
    const id = mapping[client];
    if (!id) return;
    
    document.getElementById(`stat-${id}-acc`).textContent = `${(metrics.val_accuracy * 100).toFixed(1)}%`;
    document.getElementById(`stat-${id}-norm`).textContent = metrics.update_norm.toFixed(4);
}

// --- CHART.JS VISUALIZATION ---
function updateCharts() {
    const ctxChart = document.getElementById('metrics-chart').getContext('2d');
    
    const rounds = metricsHistory.map(h => h.round);
    
    let datasets = [];
    let title = "";
    
    if (activeTab === 'acc') {
        title = "Accuracy Over Rounds";
        
        // Global Test Accuracy
        datasets.push({
            label: 'Global Model (Test Set)',
            data: metricsHistory.map(h => h.metrics.accuracy * 100),
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            borderWidth: 3,
            tension: 0.1,
            fill: true
        });
        
        // Client Val Accuracies (only available from round 1 onwards)
        if (metricsHistory.length > 1) {
            const hasHistory = metricsHistory.some(h => h.round > 0 && h.clients && Object.keys(h.clients).length > 0);
            if (hasHistory) {
                const clients = ['hospital_a', 'hospital_b', 'hospital_c'];
                const clientLabels = { 'hospital_a': 'Hospital A Val', 'hospital_b': 'Hospital B Val', 'hospital_c': 'Hospital C Val' };
                const clientColors = { 'hospital_a': '#06b6d4', 'hospital_b': '#f59e0b', 'hospital_c': '#e11d48' };
                
                clients.forEach(c => {
                    const clientData = metricsHistory.map(h => {
                        if (h.round === 0) return null;
                        return h.clients[c] ? h.clients[c].val_accuracy * 100 : null;
                    });
                    
                    datasets.push({
                        label: clientLabels[c],
                        data: clientData,
                        borderColor: clientColors[c],
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        tension: 0.1,
                        fill: false
                    });
                });
            }
        }
    } else {
        title = "Loss Over Rounds";
        
        // Global Test Loss
        datasets.push({
            label: 'Global Model Loss (Test Set)',
            data: metricsHistory.map(h => h.metrics.loss),
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            borderWidth: 3,
            tension: 0.1,
            fill: true
        });
        
        // Client Train Losses
        if (metricsHistory.length > 1) {
            const hasHistory = metricsHistory.some(h => h.round > 0 && h.clients && Object.keys(h.clients).length > 0);
            if (hasHistory) {
                const clients = ['hospital_a', 'hospital_b', 'hospital_c'];
                const clientLabels = { 'hospital_a': 'Hospital A Train Loss', 'hospital_b': 'Hospital B Train Loss', 'hospital_c': 'Hospital C Train Loss' };
                const clientColors = { 'hospital_a': '#06b6d4', 'hospital_b': '#f59e0b', 'hospital_c': '#e11d48' };
                
                clients.forEach(c => {
                    const clientData = metricsHistory.map(h => {
                        if (h.round === 0) return null;
                        return h.clients[c] ? h.clients[c].local_train_loss : null;
                    });
                    
                    datasets.push({
                        label: clientLabels[c],
                        data: clientData,
                        borderColor: clientColors[c],
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        tension: 0.1,
                        fill: false
                    });
                });
            }
        }
    }
    
    if (currentChart) {
        currentChart.destroy();
    }
    
    currentChart = new Chart(ctxChart, {
        type: 'line',
        data: {
            labels: rounds.map(r => `Round ${r}`),
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        font: { size: 10, family: 'Inter' },
                        color: '#94a3b8'
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#64748b', font: { size: 9 } }
                },
                y: {
                    min: activeTab === 'acc' ? 40 : undefined,
                    max: activeTab === 'acc' ? 100 : undefined,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#64748b',
                        font: { size: 9 },
                        callback: function(value) {
                            return activeTab === 'acc' ? value + '%' : value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// --- CANVASES TOPOLOGY ANIMATION ---
function resizeCanvas() {
    const parent = canvas.parentElement;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;
    
    // Define topology node positions based on dimensions
    const w = canvas.width;
    const h = canvas.height;
    
    topologyNodes = {
        server: { x: w / 2, y: h / 2 - 10, name: "Aggregator Server", color: '#6366f1', radius: 20 },
        hospital_a: { x: w * 0.18, y: h * 0.25, name: "Hospital A", color: '#06b6d4', radius: 12 },
        hospital_b: { x: w * 0.18, y: h * 0.70, name: "Hospital B", color: '#06b6d4', radius: 12 },
        hospital_c: { x: w * 0.82, y: h * 0.48, name: "Hospital C", color: '#06b6d4', radius: 12 }
    };
}

function sendPackets(fromKey, toKey) {
    const speed = 1.8;
    
    if (fromKey === 'server' && toKey === 'clients') {
        const dests = ['hospital_a', 'hospital_b', 'hospital_c'];
        dests.forEach(dest => {
            const start = topologyNodes.server;
            const end = topologyNodes[dest];
            // Spawn 5 packets along the line
            for (let i = 0; i < 6; i++) {
                particles.push({
                    x: start.x,
                    y: start.y,
                    tx: end.x,
                    ty: end.y,
                    progress: -i * 0.15, // staggered starts
                    speed: speed / 100,
                    color: '#10b981' // green download packets
                });
            }
        });
    } else if (toKey === 'server') {
        const start = topologyNodes[fromKey];
        const end = topologyNodes.server;
        for (let i = 0; i < 6; i++) {
            particles.push({
                x: start.x,
                y: start.y,
                tx: end.x,
                ty: end.y,
                progress: -i * 0.15,
                speed: speed / 100,
                color: '#6366f1' // indigo upload packets
            });
        }
    }
}

function drawTopology() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const server = topologyNodes.server;
    if (!server) return;
    
    // Draw connection lines
    const clients = ['hospital_a', 'hospital_b', 'hospital_c'];
    clients.forEach(c => {
        const node = topologyNodes[c];
        ctx.beginPath();
        ctx.moveTo(server.x, server.y);
        ctx.lineTo(node.x, node.y);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);
    });
    
    // Update and draw flowing packets
    particles = particles.filter(p => {
        p.progress += p.speed;
        if (p.progress < 0) return true; // not spawned yet
        if (p.progress >= 1.0) return false; // reached target
        
        // Linear interpolation
        p.x = p.tx * p.progress + (1 - p.progress) * (p.tx === server.x ? p.x : server.x); // dynamic start fix
        
        // Actual coordinates
        const startX = p.tx === server.x ? topologyNodes[Object.keys(topologyNodes).find(k => topologyNodes[k].x === p.x && k !== 'server') || 'hospital_a'].x : server.x;
        const startY = p.tx === server.x ? topologyNodes[Object.keys(topologyNodes).find(k => topologyNodes[k].y === p.y && k !== 'server') || 'hospital_a'].y : server.y;
        
        p.x = startX + (p.tx - startX) * p.progress;
        p.y = startY + (p.ty - startY) * p.progress;
        
        // Draw particle glow
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
        
        return true;
    });
    
    // Server aggregation pulse animation
    if (serverPulseActive) {
        serverPulseRadius += 0.8;
        if (serverPulseRadius > 45) serverPulseRadius = 0;
        
        ctx.beginPath();
        ctx.arc(server.x, server.y, server.radius + serverPulseRadius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(99, 102, 241, ${1 - serverPulseRadius / 45})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
    
    // Draw client nodes
    clients.forEach(c => {
        const node = topologyNodes[c];
        
        // Glow effect if training
        const card = document.getElementById(`card-${c}`);
        const isClientTraining = card && card.querySelector('.client-indicator').classList.contains('training');
        
        if (isClientTraining) {
            ctx.beginPath();
            ctx.arc(node.x, node.y, node.radius + 6, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(6, 182, 212, 0.08)';
            ctx.strokeStyle = 'rgba(6, 182, 212, 0.2)';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fill();
        }
        
        // Draw node
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = isClientTraining ? '#06b6d4' : '#1e293b';
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fill();
        
        // Draw node name
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(c.replace('_', ' ').toUpperCase(), node.x, node.y + node.radius + 14);
    });
    
    // Draw server node
    ctx.beginPath();
    ctx.arc(server.x, server.y, server.radius, 0, Math.PI * 2);
    const gradient = ctx.createRadialGradient(server.x, server.y, 2, server.x, server.y, server.radius);
    gradient.addColorStop(0, '#818cf8');
    gradient.addColorStop(1, '#4f46e5');
    ctx.fillStyle = gradient;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.fill();
    
    // Server Label
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 10px Outfit';
    ctx.textAlign = 'center';
    ctx.fillText("AGGREGATOR", server.x, server.y + server.radius + 15);
    
    animationFrameId = requestAnimationFrame(drawTopology);
}

// --- DIAGNOSTIC INFERENCE SANDBOX INTERACTION ---
function loadSampleGallery() {
    const gallery = document.getElementById('gallery-samples');
    gallery.innerHTML = '';
    
    fetch('/api/samples')
        .then(res => res.json())
        .then(data => {
            if (data.tumor.length === 0 && data.normal.length === 0) {
                gallery.innerHTML = '<p class="loading-placeholder">No samples. Run "Generate Dataset" first!</p>';
                return;
            }
            
            // Populate gallery
            const allSamples = [];
            data.normal.forEach(path => allSamples.push({ path, label: 'Normal' }));
            data.tumor.forEach(path => allSamples.push({ path, label: 'Tumor' }));
            
            // Shuffle list for variety
            allSamples.sort(() => Math.random() - 0.5);
            
            allSamples.forEach((item, index) => {
                const img = document.createElement('img');
                img.src = item.path;
                img.className = 'gallery-item';
                img.title = `${item.label} MRI Sample`;
                img.addEventListener('click', () => {
                    selectImage(item.path, `Sample: ${item.label} MRI`);
                    document.querySelectorAll('.gallery-item').forEach(el => el.classList.remove('selected'));
                    img.classList.add('selected');
                });
                gallery.appendChild(img);
            });
        })
        .catch(err => {
            gallery.innerHTML = '<p class="loading-placeholder">Failed to fetch samples</p>';
        });
}

function selectImage(path, labelText) {
    selectedImagePath = path;
    const preview = document.getElementById('mri-preview');
    preview.src = path;
    
    document.getElementById('mri-meta-info').textContent = `128 x 128 | Grayscale | ${labelText}`;
    
    // Enable predict button (only if model is trained at least round 1 or training finished)
    // For local evaluation, we can allow prediction at any time, but default to enabling it
    document.getElementById('btn-predict').disabled = false;
    
    // Reset prediction outcome card states
    const card = document.getElementById('prediction-outcome-card');
    card.className = "prediction-card-glow";
    document.getElementById('diagnosis-outcome').textContent = "Ready to Analyze";
    document.getElementById('diagnosis-statement').textContent = "Click 'Analyze MRI Slice' to run deep learning inference.";
    
    updateRadialDial(0, '#6366f1');
}

// Generate random MRI slices
document.getElementById('btn-gen-normal').addEventListener('click', () => generateRandomMRI(false));
document.getElementById('btn-gen-tumor').addEventListener('click', () => generateRandomMRI(true));

function generateRandomMRI(hasTumor) {
    fetch(`/api/generate-random-mri?has_tumor=${hasTumor}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            selectImage(data.url, `Generated: ${hasTumor ? 'Tumor' : 'Normal'} MRI`);
            document.querySelectorAll('.gallery-item').forEach(el => el.classList.remove('selected'));
        });
}

// Upload custom files
const fileUpload = document.getElementById('file-upload');
document.getElementById('btn-upload').addEventListener('click', () => {
    fileUpload.click();
});

fileUpload.addEventListener('change', (e) => {
    if (e.target.files.length === 0) return;
    const file = e.target.files[0];
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Disable interface while uploading and predicting
    document.getElementById('btn-predict').disabled = true;
    const frame = document.getElementById('mri-frame');
    frame.classList.add('scanning');
    
    fetch('/api/upload-inference', {
        method: 'POST',
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        // Stop scanning, select image, and present result
        frame.classList.remove('scanning');
        selectedImagePath = data.url;
        document.getElementById('mri-preview').src = data.url;
        document.getElementById('mri-meta-info').textContent = `Uploaded File: ${file.name}`;
        
        displayPrediction(data);
    })
    .catch(err => {
        frame.classList.remove('scanning');
        logToConsole("Failed to upload custom image", "error");
    });
});

// Predict button listener
document.getElementById('btn-predict').addEventListener('click', () => {
    if (!selectedImagePath) return;
    
    // Add scanner effect
    const frame = document.querySelector('.mri-frame');
    frame.classList.add('scanning');
    document.getElementById('btn-predict').disabled = true;
    
    // Call server to predict
    fetch(`/api/predict-image?image_path=${encodeURIComponent(selectedImagePath)}`, { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            // Simulated delay of 1.2s to make animation satisfying
            setTimeout(() => {
                frame.classList.remove('scanning');
                document.getElementById('btn-predict').disabled = false;
                displayPrediction(data);
            }, 1200);
        })
        .catch(err => {
            frame.classList.remove('scanning');
            document.getElementById('btn-predict').disabled = false;
            logToConsole("Prediction request failed.", "error");
        });
});

function displayPrediction(data) {
    const card = document.getElementById('prediction-outcome-card');
    const outcomeVal = document.getElementById('diagnosis-outcome');
    const statement = document.getElementById('diagnosis-statement');
    
    const probability = data.probability;
    const percentage = Math.round(probability * 100);
    
    outcomeVal.textContent = data.prediction;
    
    if (data.prediction === 'Tumor') {
        card.className = "prediction-card-glow tumor-prediction";
        outcomeVal.style.color = "var(--color-danger)";
        statement.textContent = `CRITICAL: High-intensity neural mass detected with confidence score ${percentage}%. Simulated pathological structures are consistent with glial brain tumors. Recommended follow-up diagnostic MR scans.`;
        updateRadialDial(percentage, 'var(--color-danger)');
    } else {
        card.className = "prediction-card-glow normal-prediction";
        outcomeVal.style.color = "var(--color-success)";
        statement.textContent = `NORMAL: Brain tissue features are within expected physiological variations. Confidence score: ${(100 - percentage)}% normal. No signs of high-intensity growths or tumor edema identified.`;
        updateRadialDial(percentage, 'var(--color-success)');
    }
}

function updateRadialDial(percentage, color) {
    const radial = document.getElementById('radial-progress');
    const text = document.getElementById('radial-pct');
    const chart = document.querySelector('.circular-chart');
    
    // Set circle class for colors
    chart.className = "circular-chart";
    if (color.includes('danger')) chart.classList.add('tumor-color');
    if (color.includes('success')) chart.classList.add('normal-color');
    
    radial.style.strokeDasharray = `${percentage}, 100`;
    text.textContent = `${percentage}%`;
}

// --- INIT APP ---
window.addEventListener('load', () => {
    // 1. Establish web sockets
    connectWebSocket();
    
    // 2. Initialize Canvas Topology
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    drawTopology();
    
    // 3. Load test set gallery
    loadSampleGallery();
});
