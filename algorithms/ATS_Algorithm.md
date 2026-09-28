# ATS (Adaptive Threshold Selection) Algorithm

**File Location:** `backend/src/main/java/com/ccsutd/miniproject/service/AtsService.java`
**Line Numbers:** Lines 27 - 81 (`calculateAdaptiveThreshold` method)

### Code Snippet
```java
public double calculateAdaptiveThreshold(Instances knownTrainingData, Instances pseudoNegatives, Instances likelyKnowns) throws Exception {
    List<Double> percentileValues = new ArrayList<>();
    
    // Divide data into S subsets
    int chunkSize = likelyKnowns.numInstances() / numberOfSubsetsS;
    
    for (int s = 0; s < numberOfSubsetsS; s++) {
        // 1. Train progressive Random Forest
        RandomForest progressiveRf = new RandomForest();
        progressiveRf.buildClassifier(currentTraining);

        // 2. Test against Pseudo-Negatives to find Confidence differences (CfDmax)
        List<Double> currentCfdMaxList = new ArrayList<>();
        for (int i = 0; i < pseudoNegatives.numInstances(); i++) {
            Instance pn = pseudoNegatives.instance(i);
            double[] probs = progressiveRf.distributionForInstance(pn);
            // Calculate CfDmax (Max Prob - Min Prob)
            // ...
        }

        // 3. Find 10th percentile
        Collections.sort(currentCfdMaxList);
        int pIndex = (int) Math.ceil((10 / 100.0) * currentCfdMaxList.size()) - 1;
        percentileValues.add(currentCfdMaxList.get(pIndex));
    }

    // 4. Calculate Final ATS Alpha = Mean - Standard Deviation
    double mean = sum / percentileValues.size();
    double std = Math.sqrt(sqSum / percentileValues.size());
    double alpha = mean - std;
    return alpha;
}
```

### Notes to Explain to Professor
- **What it does:** This algorithm dynamically calculates the `Alpha` threshold (usually ~0.90) required to reject a packet as a Zero-Day Threat. Instead of hardcoding "90%", the AI calculates the exact perfect threshold for the current network environment.
- **How it works:** It trains `S=15` mini-Random Forests on chunks of the data. For each chunk, it tests against known bad data ("pseudo-negatives") and finds the 10th percentile of confidence scores. Finally, it averages them out and subtracts the standard deviation (`Alpha = Mean - StdDev`).
- **Why it matters:** Hardcoded thresholds fail when network traffic changes (like during a weekend vs a weekday). ATS allows the model to self-adjust its strictness dynamically.
