# H1/H2 Cascade Algorithm (Using Random Forest)

**File Location:** `backend/src/main/java/com/ccsutd/miniproject/service/MachineLearningService.java`
**Line Numbers:** Lines 130 - 171 (`predictCascade` method)

### Code Snippet
```java
public PredictionResult predictCascade(Instance instance, Instances dataHeader) throws Exception {
    // 1. Convert incoming network packet to H1 Binary format
    DenseInstance h1Instance = new DenseInstance(1.0, h1Values);

    // 2. Evaluate using H1 (Random Forest)
    double h1PredictionIndex = h1Classifier.classifyInstance(h1Instance);
    String h1PredictionLabel = h1DatasetHeader.classAttribute().value((int) h1PredictionIndex);

    // 3. Early Exit if H1 catches a difficult Zero-Day threat
    if ("NEW".equals(h1PredictionLabel)) {
        // H1 blocked it! We bypass H2 entirely.
        return new PredictionResult("NEW_CLASS", "NEW", confidenceResult, betaThreshold, "NEW_CLASS_DETECTED");
    }

    // 4. If H1 thinks it's normal, pass it to H2 (Stage 2)
    ConfidenceResult confidenceResult = getH2Confidence(instance, dataHeader);

    // 5. Apply the ATS Alpha/Beta Threshold
    if (confidenceResult.getCfDMax() > betaThreshold) {
        classType = "KNOWN";
        finalPrediction = confidenceResult.getPredictedClass(); // Allow traffic
    } else {
        classType = "NEW";
        finalPrediction = "NEW_CLASS"; // Block traffic (Zero-Day detected)
    }

    return new PredictionResult(finalPrediction, classType, confidenceResult, betaThreshold, h1Result);
}
```

### Notes to Explain to Professor
- **What it does:** This is the main "Cascade" architecture. Standard models only use one algorithm. We use TWO algorithms chained together (H1 and H2). 
- **How it works:** 
  - **H1 (The Bouncer):** It looks at the packet first. If it looks even slightly like an unknown anomaly, it immediately drops the packet and triggers the XAI alert.
  - **H2 (The Deep Inspector):** If H1 thinks it's fine, it passes it to H2. H2 calculates the mathematical confidence of the packet. 
  - **ATS (The Threshold):** If H2's confidence is lower than our calculated ATS threshold (e.g., 90%), we still block it. 
- **Why it matters:** By chaining them together, our accuracy goes through the roof, especially for unseen Zero-Day attacks, outperforming single-model baselines.
