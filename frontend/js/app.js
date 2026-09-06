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
    
    // Clear previous
    resultBox.innerHTML = "";
    statusDiv.innerText = "Status: Executing " + type + " pipeline... (This may take a few seconds)";
    spinner.style.display = 'block';

    const endpoint = type === 'cascade' ? 'http://localhost:8080/api/ml/train-full-cascade' : 'http://localhost:8080/api/ml/test-baseline';

    try {
        const response = await fetch(endpoint);
        const data = await response.json();
        
        spinner.style.display = 'none';
        statusDiv.innerText = "Status: Completed Successfully!";
        
        // Render detailed text
        let html = `<h3>Run Details</h3>`;
        html += `<p><strong>ATS Alpha:</strong> ${data.atsCalculatedAlpha || 'Fixed 0.9'}</p>`;
        html += `<p><strong>H1 Early Blocks:</strong> ${data.cascadeMetrics.h1EarlyBlocks || 0}</p>`;
        html += `<p><strong>Total Tested:</strong> ${data.cascadeMetrics.totalTested}</p>`;
        html += `<p><strong>Correct Known:</strong> ${data.cascadeMetrics.correctKnown}</p>`;
        html += `<p><strong>Correct New:</strong> ${data.cascadeMetrics.correctNew}</p>`;
        html += `<p><strong>False Positives (Known):</strong> ${data.cascadeMetrics.falseKnown}</p>`;
        html += `<p><strong>False Negatives (New):</strong> ${data.cascadeMetrics.falseNew}</p>`;
        resultBox.innerHTML = html;

        // Update dashboard metrics
        document.getElementById('stat-na').innerText = (data.cascadeMetrics.normalizedAccuracy * 100).toFixed(2) + "%";
        document.getElementById('stat-known-acc').innerText = (data.cascadeMetrics.knownAccuracy * 100).toFixed(2) + "%";
        document.getElementById('stat-new-acc').innerText = (data.cascadeMetrics.unknownAccuracy * 100).toFixed(2) + "%";
        document.getElementById('stat-alpha').innerText = (data.atsCalculatedAlpha || 0.9).toFixed(3);

        // Update Chart
        updateChart(data.cascadeMetrics.knownAccuracy * 100, data.cascadeMetrics.unknownAccuracy * 100, data.cascadeMetrics.normalizedAccuracy * 100);

        // Switch to dashboard to show graph
        showSection('dashboard');

    } catch (error) {
        spinner.style.display = 'none';
        statusDiv.innerText = "Status: Error running pipeline. Is the Java server running?";
        resultBox.innerHTML = `<p style="color: red;">${error.message}</p>`;
    }
}

function updateChart(knownAcc, unknownAcc, normAcc) {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    
    if (metricsChart) {
        metricsChart.destroy();
    }

    metricsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Known Accuracy', 'Unknown Accuracy', 'Normalized Accuracy'],
            datasets: [{
                label: 'Accuracy (%)',
                data: [knownAcc, unknownAcc, normAcc],
                backgroundColor: [
                    'rgba(54, 162, 235, 0.7)',
                    'rgba(255, 99, 132, 0.7)',
                    'rgba(75, 192, 192, 0.7)'
                ],
                borderColor: [
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 99, 132, 1)',
                    'rgba(75, 192, 192, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}
