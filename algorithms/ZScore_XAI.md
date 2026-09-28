# Z-Score XAI (Explainable AI) Algorithm

**File Location:** `backend/src/main/java/com/ccsutd/miniproject/controller/MLController.java`
**Line Numbers:** Approximately Lines 179 - 198 (Inside the `/upload-pcap` route)

### Code Snippet
```java
double maxZScore = -1.0;
String anomalousFeature = "";
double anomalousValue = 0;
double expectedMean = 0;

for (int j = 0; j < reducedKnownTraining.numAttributes() - 1; j++) {
    double mean = reducedKnownTraining.meanOrMode(j);
    double stdDev = Math.sqrt(reducedKnownTraining.variance(j));
    double val = targetInstance.value(j);
    
    if (stdDev > 0) {
        double zScore = Math.abs((val - mean) / stdDev);
        if (zScore > maxZScore) {
            maxZScore = zScore;
            anomalousFeature = reducedKnownTraining.attribute(j).name();
            anomalousValue = val;
            expectedMean = mean;
        }
    }
}
xaiReason = String.format("XAI ALERT: Feature '%s' spiked to %.2f. (Normal known traffic average is only %.2f).", anomalousFeature, anomalousValue, expectedMean);
```

### Notes to Explain to Professor
- **What it does:** This algorithm mathematically explains *why* the AI blocked a Zero-Day packet, rather than just saying "Threat Detected" (which is known as a black-box problem).
- **How it works:** It loops through all 23 features of the packet. For each feature, it calculates the Z-Score by subtracting the *Average Normal Mean* from the *Current Packet Value* and dividing by the *Standard Deviation*.
- **Why it matters:** It finds the single feature that spiked the highest (Max Z-Score) and puts it in the Email Alert so the IT Administrator instantly knows what anomalous behavior triggered the block.
