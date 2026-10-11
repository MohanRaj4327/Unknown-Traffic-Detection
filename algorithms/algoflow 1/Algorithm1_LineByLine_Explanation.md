# Algorithm 1: Adaptive Threshold Selection (ATS) - Complete Flow & Code Explanation

This document explains **Algorithm 1** from the research paper:
> *"Unknown Network Traffic Detection via Two-Stage Cascade Classifier with Adaptive Threshold"*

The code implementation for this algorithm is saved in:
👉 `algorithms/algoflow 1/AdaptiveThresholdSelection_Algorithm1.java`

---

## 1. Algorithm Overview & Mathematical Goal
In standard network intrusion detection systems, confidence thresholds are fixed (e.g., hardcoded at 0.90). However, network conditions change over time.
**Algorithm 1 dynamically computes the optimal decision threshold (\(\alpha\))** by evaluating how confidence distributions shift across incremental memory subsets.

---

## 2. Input / Output / Requirements

| Parameter | Type / Source | Meaning & Role |
| :--- | :--- | :--- |
| **Require: \(S\)** | Integer (Default: 15) | **Size of memory**: Number of progressive data subsets. |
| **Require: \(\text{RF}_0\)** | Random Forest Model | Initial model trained on known traffic classes. |
| **Require: \(\text{prediction memory}\)** | Array / List | Memory buffer to store 10th percentile scores from each subset iteration. |
| **Require: \(\alpha_0\)** | Float (Default: 0.90) | Initial baseline threshold. |
| **Input: \(U\)** | Dataset | The unlabelled network traffic flow dataset. |
| **Output: \(\alpha\)** | Float (0.0 to 1.0) | The calculated **Adaptive Threshold** value. |

---

## 3. Step-by-Step Line Mapping (Pseudocode vs Java Code)

### Line 1 & Line 2: Obtain M2 and M from Algorithm 2
* **Pseudocode:**
  ```text
  1: Obtain known flows M2 by Algorithm 2
  2: Obtain subset M of unlabelled data with unknown classes whose CfDmax > alpha_0 by Algorithm 2
  ```
* **Java Implementation:**
  ```java
  // M_2 and M are passed from Algorithm 2 (Pseudo-Negative & Likely-Known extraction)
  if (M_2 == null || M_2.numInstances() == 0) return alpha_0;
  if (M == null || M.numInstances() == 0) return alpha_0;
  ```
* **Explanation:**
  * \(M_2\) contains unlabelled traffic samples that look like known traffic.
  * \(M\) contains candidate unknown / pseudo-negative samples that initially had high confidence scores.

---

### Line 3 & Line 4: Divide \(M_2\) into \(S\) Subsets
* **Pseudocode:**
  ```text
  3: # Update RF_0 to recalculate CfDmax for M
  4: Divide M2 into S subsets: M2 = {Sub_1, Sub_2, ..., Sub_S}
  ```
* **Java Implementation:**
  ```java
  List<Instances> subsets = new ArrayList<>();
  int chunkSize = totalInstancesM2 / S;
  for (int i = 0; i < S; i++) {
      Instances sub = new Instances(M_2, startIndex, chunkSize);
      subsets.add(sub);
  }
  ```
* **Explanation:**
  * Rather than retraining all at once, \(M_2\) is split into \(S\) progressive chunks (\(Sub_1, Sub_2, \dots, Sub_S\)) to simulate streaming network flows.

---

### Lines 5 to 9: Progressive Model Retraining
* **Pseudocode:**
  ```text
  5: for i = 1 to S do
  6:    Define Training set Set_i = Sub_1 U Sub_2 U ... U Sub_i
  7:    Retrain RF_i using RF_{i-1} as the initial model and Training set Set_i
  8:    Obtain and store updated model RF_i
  9: end for
  ```
* **Java Implementation:**
  ```java
  for (int i = 0; i < S; i++) {
      Instances Set_i = new Instances(subsets.get(0));
      for (int k = 1; k <= i; k++) {
          Set_i.addAll(subsets.get(k));
      }
      RandomForest RF_i = new RandomForest();
      RF_i.buildClassifier(Set_i);
      trainedModelsRF.add(RF_i);
  }
  ```
* **Explanation:**
  * Each iteration \(i\) creates a progressively larger dataset \(Set_i\).
  * A new updated model \(\text{RF}_i\) is trained on \(Set_i\) and stored in memory.

---

### Lines 10 to 17: CfDmax Calculation & 10th Percentile Extraction
* **Pseudocode:**
  ```text
  10: Initialize a temporary array: temp
  11: for j = 1 to S do
  12:    for each m in M do
  13:       Compute CfDmax-m <- calculation using RF_j on m
  14:       Store CfDmax-m in temp
  15:    end for
  16:    Calculate the 10th percentile of temp and store it in memory
  17: end for
  ```
* **Java Implementation:**
  ```java
  for (int j = 0; j < S; j++) {
      RandomForest RF_j = trainedModelsRF.get(j);
      List<Double> temp = new ArrayList<>();
      
      for (int idx = 0; idx < M.numInstances(); idx++) {
          Instance m = M.instance(idx);
          double CfDmax_m = computeCfDmax(RF_j, m); // MaxProb - MinProb
          temp.add(CfDmax_m);
      }
      
      Collections.sort(temp);
      int pIndex = (int) Math.ceil((10 / 100.0) * temp.size()) - 1;
      predictionMemory.add(temp.get(pIndex));
  }
  ```
* **Explanation:**
  * **CfDmax (Confidence Difference):** Difference between maximum predicted class probability and minimum predicted class probability:
    \[
    \text{CfDmax} = P_{\max} - P_{\min}
    \]
  * **10th Percentile:** We take the 10th percentile of confidence to capture the lower-bound confidence for difficult samples without being skewed by extreme noise.

---

### Lines 18 to 22: Calculate Final Adaptive Threshold \(\alpha\)
* **Pseudocode:**
  ```text
  18: # Calculate mean and standard deviation in memory to obtain adaptive threshold alpha
  19: Calculate the mean in memory: mean
  20: Calculate the standard deviation in memory: std
  21: Adaptive threshold alpha = mean - std
  22: Return alpha
  ```
* **Java Implementation:**
  ```java
  double mean = sum / predictionMemory.size();
  double std = Math.sqrt(squaredDiffSum / predictionMemory.size());

  double alpha = mean - std;
  return Math.max(0.0, Math.min(1.0, alpha));
  ```
* **Explanation:**
  * We compute the statistical **mean** and **standard deviation (std)** of the 10th percentile scores across all memory iterations.
  * Setting \(\alpha = \text{mean} - \text{std}\) creates a statistically grounded decision boundary: any flow whose confidence falls below this threshold is flagged as **UNKNOWN (New Class Zero-Day Threat)**.

---

## 4. Connection to Existing Backend
* In the main running backend service, this exact logic is integrated inside:
  `backend/src/main/java/com/ccsutd/miniproject/service/AtsService.java` (Method: `calculateAdaptiveThreshold`).
* The code in `algorithms/algoflow 1/` is a dedicated, self-contained implementation matching the paper's Algorithm 1 step-by-step so your professor can verify every line directly.
