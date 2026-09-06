function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('section').forEach(sec => {
        sec.classList.remove('active');
        sec.classList.add('hidden');
    });

    // Show selected section
    const target = document.getElementById(sectionId);
    if (target) {
        target.classList.remove('hidden');
        target.classList.add('active');
    }
}

function startTraining() {
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerText = "Status: Training in progress... (Simulated)";
    // In future phases, this will call the Spring Boot API
    setTimeout(() => {
        statusDiv.innerText = "Status: Training Completed Successfully!";
    }, 2000);
}

function runPrediction() {
    const resultBox = document.getElementById('prediction-result');
    resultBox.innerHTML = "<p>Running inference...</p>";
    // In future phases, this will call the Spring Boot API
    setTimeout(() => {
        resultBox.innerHTML = `
            <p><strong>Predicted Class:</strong> BROWSING</p>
            <p><strong>Class Type:</strong> KNOWN</p>
            <p><strong>CfDmax:</strong> 0.95</p>
            <p><strong>Result:</strong> (Simulated placeholder)</p>
        `;
    }, 1000);
}
