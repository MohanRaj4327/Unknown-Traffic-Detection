# One-Class SVM (Pseudo-Negative Generation) Algorithm

**File Location:** `backend/src/main/java/com/ccsutd/miniproject/service/OcSvmService.java`
**Line Numbers:** Inside the `selectPseudoNegatives` method.

### Code Snippet Explanation
```java
// We use a One-Class Support Vector Machine (OC-SVM) algorithm 
// to automatically find data that looks "weird" but we aren't 
// sure what it is yet. We call these "Pseudo-Negatives".

LibSVM ocSvm = new LibSVM();
ocSvm.setSVMType(new SelectedTag(LibSVM.SVMTYPE_ONE_CLASS_SVM, LibSVM.TAGS_SVMTYPE));
ocSvm.buildClassifier(knownTrainingData);

// Test unlabelled data against the OC-SVM
for (int i = 0; i < unlabelledData.numInstances(); i++) {
    Instance inst = unlabelledData.instance(i);
    double prediction = ocSvm.classifyInstance(inst);
    
    // If OC-SVM rejects it (-1.0), it is highly likely to be a Zero-Day Threat
    if (prediction < 0) {
        pseudoNegatives.add(inst);
    } else {
        likelyKnowns.add(inst);
    }
}
```

### Notes to Explain to Professor
- **What it does:** Standard AIs need both "Good" and "Bad" data to train. But we don't have Zero-Day bad data (because it's a zero-day!). So we use an unsupervised OC-SVM to generate fake bad data called **Pseudo-Negatives**.
- **How it works:** OC-SVM draws a mathematical boundary around all our known normal traffic. Any unlabelled data that falls *outside* that boundary is labeled as a "Pseudo-Negative" (a simulated zero-day threat).
- **Why it matters:** We feed these Pseudo-Negatives into the ATS and H1 algorithms so they can learn what a zero-day threat *might* look like, without ever needing actual zero-day training data.
