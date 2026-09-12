let metricsChart;

function showSection(sectionId) {
    document.querySelectorAll('section').forEach(sec => {
        sec.classList.remove('active');
        sec.classList.add('hidden');
    });
    const target = document.getElementById(sectionId);
    if (target) {
        target.classList.remove('hidden');
        target.classList.add('active');
    }
}

async function runModel(type) {
    const statusDiv = document.getElementById('training-status');
    const resultBox = document.getElementById('prediction-result');
    const spinner = document.getElementById('loading-spinner');
    
    resultBox.innerHTML = "";
    statusDiv.innerText = "Status: Executing " + type + " pipeline... (This may take a few seconds)";
    spinner.style.display = 'block';

    const endpoint = type === 'cascade' ? 'http://localhost:8080/api/ml/train-full-cascade' : 'http://localhost:8080/api/ml/test-baseline';

    try {
        const response = await fetch(endpoint);
        const data = await response.json();
        
        spinner.style.display = 'none';
        statusDiv.innerText = "Status: Completed Successfully!";
        
        let html = `<h3>Run Details</h3>`;
        html += `<p><strong>ATS Alpha:</strong> ${data.atsCalculatedAlpha || 'Fixed 0.9'}</p>`;
        html += `<p><strong>H1 Early Blocks:</strong> ${data.cascadeMetrics.h1EarlyBlocks || 0}</p>`;
        html += `<p><strong>Total Tested:</strong> ${data.cascadeMetrics.totalTested}</p>`;
        html += `<p><strong>Correct Known:</strong> ${data.cascadeMetrics.correctKnown}</p>`;
        html += `<p><strong>Correct New:</strong> ${data.cascadeMetrics.correctNew}</p>`;
        html += `<p><strong>False Positives (Known):</strong> ${data.cascadeMetrics.falseKnown}</p>`;
        html += `<p><strong>False Negatives (New):</strong> ${data.cascadeMetrics.falseNew}</p>`;
        resultBox.innerHTML = html;

        document.getElementById('stat-na').innerText = (data.cascadeMetrics.normalizedAccuracy * 100).toFixed(2) + "%";
        document.getElementById('stat-known-acc').innerText = (data.cascadeMetrics.knownAccuracy * 100).toFixed(2) + "%";
        document.getElementById('stat-new-acc').innerText = (data.cascadeMetrics.unknownAccuracy * 100).toFixed(2) + "%";
        document.getElementById('stat-alpha').innerText = (data.atsCalculatedAlpha || 0.9).toFixed(3);

        updateChart(data.cascadeMetrics.knownAccuracy * 100, data.cascadeMetrics.unknownAccuracy * 100, data.cascadeMetrics.normalizedAccuracy * 100);

        showSection('dashboard');
    } catch (error) {
        spinner.style.display = 'none';
        statusDiv.innerText = "Status: Error running pipeline. Is the Java server running?";
        resultBox.innerHTML = `<p style="color: red;">${error.message}</p>`;
    }
}

function renderPredictionResult(data) {
    let html = `<h3>Single Packet Analysis</h3>`;
    
    if (data.pcapTotalPackets) {
        html += `<p style="color: #66fcf1;"><strong>File Parsed:</strong> ${data.fileName}</p>`;
        html += `<p style="color: #66fcf1;"><strong>Packets Scanned:</strong> ${data.pcapTotalPackets}</p>`;
        html += `<p style="color: #66fcf1;"><strong>Total Bytes:</strong> ${data.pcapTotalBytes}</p><hr>`;
    }

    html += `<p style="color: #666;"><strong>Ground Truth (Actual):</strong> ${data.trueClass || 'UNKNOWN'} (${data.trueType || 'UNKNOWN'})</p><hr>`;
    html += `<p><strong>Step 1 (H1 Bouncer):</strong> ${data.h1BouncerResult === 'NEW_CLASS_DETECTED' ? '<span style="color:red;">BLOCKED (UNKNOWN THREAT)</span>' : '<span style="color:#39ff14;">PASSED to H2</span>'}</p>`;
    html += `<p><strong>Step 2 (H2 Confidence):</strong> ${(data.h2HighestConfidence * 100).toFixed(2)}% (Threshold: ${(data.atsThresholdUsed * 100).toFixed(2)}%)</p>`;
    
    let decisionColor = data.finalDecisionType === 'NEW' ? '#00ffcc' : '#39ff14';
    html += `<h4 style="color: ${decisionColor}; font-size: 1.4em; text-shadow: 0 0 5px ${decisionColor};">Final AI Decision: ${data.finalDecisionType} TRAFFIC -> [${data.finalDecisionClass}]</h4>`;
    
    if (data.finalDecisionType === 'NEW') {
        html += `<div style="background-color: rgba(255, 0, 0, 0.1); border-left: 4px solid #ff4444; padding: 10px; margin-top: 15px;">
                    <strong style="color: #ff4444;">🧠 Explainable AI (XAI) Engine:</strong><br>
                    <span style="color: #ddd;">${data.xaiReason}</span>
                 </div>`;
    }
    
    html += `<hr style="border-color: #333;"><p style="font-size:0.9em; color:#00ffcc;">[Packet Features Extracted]:<br>${JSON.stringify(data.sampleFeatures, null, 2)}</p>`;
    return html;
}

async function runSinglePredict(type) {
    const statusDiv = document.getElementById('single-status');
    const resultBox = document.getElementById('single-result');
    const spinner = document.getElementById('single-loading-spinner');
    
    resultBox.innerHTML = "";
    statusDiv.innerText = `Status: Intercepting a live ${type.toUpperCase()} packet...`;
    spinner.style.display = 'block';

    try {
        const response = await fetch(`http://localhost:8080/api/ml/predict-single?type=${type}`);
        const data = await response.json();
        spinner.style.display = 'none';
        statusDiv.innerText = "Status: Packet Analyzed!";
        resultBox.innerHTML = renderPredictionResult(data);
    } catch (error) {
        spinner.style.display = 'none';
        statusDiv.innerText = "Status: Error. Is the Java server running?";
        resultBox.innerHTML = `<p style="color: red;">${error.message}</p>`;
    }
}

async function uploadPcap() {
    const statusDiv = document.getElementById('single-status');
    const resultBox = document.getElementById('single-result');
    const spinner = document.getElementById('single-loading-spinner');
    const fileInput = document.getElementById('pcapFile');

    if (!fileInput.files.length) {
        alert("Please select a PCAP file first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultBox.innerHTML = "";
    statusDiv.innerText = `Status: Parsing PCAP and extracting flows...`;
    spinner.style.display = 'block';

    try {
        const response = await fetch(`http://localhost:8080/api/ml/upload-pcap`, {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        spinner.style.display = 'none';
        statusDiv.innerText = "Status: PCAP Analyzed Successfully!";
        resultBox.innerHTML = renderPredictionResult(data);
    } catch (error) {
        spinner.style.display = 'none';
        statusDiv.innerText = "Status: Error. Is the Java server running?";
        resultBox.innerHTML = `<p style="color: red;">${error.message}</p>`;
    }
}

function updateChart(knownAcc, unknownAcc, normAcc) {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    if (metricsChart) { metricsChart.destroy(); }
    metricsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Known Accuracy', 'Unknown Accuracy', 'Normalized Accuracy'],
            datasets: [{
                label: 'Accuracy (%)',
                data: [knownAcc, unknownAcc, normAcc],
                backgroundColor: [ 'rgba(54, 162, 235, 0.7)', 'rgba(255, 99, 132, 0.7)', 'rgba(75, 192, 192, 0.7)' ],
                borderColor: [ 'rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)', 'rgba(75, 192, 192, 1)' ],
                borderWidth: 1
            }]
        },
        options: { 
            scales: { 
                y: { 
                    beginAtZero: true, 
                    max: 100,
                    ticks: { color: '#c5c6c7' },
                    grid: { color: '#333' }
                },
                x: {
                    ticks: { color: '#c5c6c7' },
                    grid: { color: '#333' }
                }
            },
            plugins: {
                legend: { labels: { color: '#c5c6c7' } }
            }
        }
    });
}

async function loadHistory() {
    const historyBox = document.getElementById('history-content');
    historyBox.innerHTML = '<p style="color: #66fcf1;">Fetching database logs...</p>';
    try {
        const response = await fetch('http://localhost:8080/api/ml/history');
        const data = await response.json();
        
        let html = `<table>
            <thead>
                <tr>
                    <th>Run Date</th>
                    <th>Norm. Accuracy</th>
                    <th>Unknown Accuracy</th>
                    <th>H1 Early Blocks</th>
                    <th>ATS Alpha</th>
                </tr>
            </thead>
            <tbody>`;
            
        data.forEach(run => {
            html += `<tr>
                <td>${new Date(run.runDate).toLocaleString()}</td>
                <td>${(run.normalizedAccuracy * 100).toFixed(2)}%</td>
                <td>${(run.unknownAccuracy * 100).toFixed(2)}%</td>
                <td>${run.h1EarlyBlocks}</td>
                <td>${run.atsCalculatedAlpha.toFixed(2)}</td>
            </tr>`;
        });
        
        html += `</tbody></table>`;
        historyBox.innerHTML = html;
    } catch (error) {
        historyBox.innerHTML = `<p style="color: red;">Error fetching history: ${error.message}</p>`;
    }
}

function downloadPDF() {
    const accuracyText = document.getElementById('stat-na').innerText;
    if (accuracyText === "0%") {
        alert("⚠️ Please run the Machine Learning Pipeline first so there is data to export!");
        return;
    }

    const element = document.getElementById('report-content');
    const opt = {
      margin:       0.5,
      filename:     'CCS-UTD-Security-Report.pdf',
      image:        { type: 'jpeg', quality: 1.0 },
      html2canvas:  { scale: 2, useCORS: true, backgroundColor: '#0b0c10' },
      jsPDF:        { unit: 'in', format: 'letter', orientation: 'landscape' }
    };
    
    // Add a temporary class to ensure dark mode text renders perfectly in PDF
    element.style.color = '#c5c6c7';
    html2pdf().set(opt).from(element).save();
}
