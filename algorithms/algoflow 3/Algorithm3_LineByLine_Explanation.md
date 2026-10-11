# Algorithm 3: Cascade Classification Algorithm - Complete Flow & Code Explanation

This document explains **Algorithm 3** from the research paper:
> *"Unknown Network Traffic Detection via Two-Stage Cascade Classifier with Adaptive Threshold"*

The code implementation for this algorithm is saved in:
👉 `algorithms/algoflow 3/CascadeClassification_Algorithm3.java`

---

## 1. Algorithm Overview & Core Innovation
Most network intrusion detection systems use a single-stage classifier (e.g., standard Random Forest or Deep Neural Network).
Single-stage classifiers suffer from a major weakness: **they force all traffic to be categorized into known labels**, causing brand new unknown zero-day attacks to be misclassified as benign known traffic.

**Algorithm 3 implements the Two-Stage Cascade Architecture:**
1. **Stage 1 (\(H_1\) Binary Classifier - "The Gatekeeper"):** Quickly inspects all traffic. If a flow exhibits strong anomalous patterns, it is caught immediately (\(y_1\)) and blocked before reaching Stage 2.
2. **Stage 2 (\(H_2\) Multi-Class Classifier + ATS Threshold \(\beta\)):** Handles flows passed by \(H_1\) (\(x_0\)). It calculates the distinctiveness of the class probabilities (\(\text{CfDmax}\)). If the confidence difference exceeds \(\beta\), it assigns a specific known class label (\(z_j\)). If confidence is lower than \(\beta\), it is caught as a subtle unknown zero-day threat (\(y_2\)).
3. **Final Result:** Total unknown zero-day attacks identified: \(y = y_1 \cup y_2\).

---

## 2. Input / Output / Requirements

| Parameter | Type / Source | Meaning & Role |
| :--- | :--- | :--- |
| **Require: \(X\)** | Dataset | Mixed unlabelled network flow dataset (containing both known and unknown traffic). |
| **Require: \(\beta\)** | Float (e.g., 0.90) | The \(\text{CfDmax}\) threshold (dynamically calculated via **Algorithm 1**, where \(\beta = \alpha\)). |
| **Output: \(z\)** | Set of subsets \(\{z_1, z_2, \dots, z_k\}\) | Confirmed Known Class labels for all \(k\) recognized traffic types. |
| **Output: \(y\)** | Dataset | Confirmed New Class (Unknown Zero-Day) traffic labels. |

---

## 3. Step-by-Step Line Mapping (Pseudocode vs Java Code)

### Line 1: Feature Extraction
* **Pseudocode:**
  ```text
  1: FE is applied to the dataset X to obtain the input sample x
  ```
* **Java Implementation:**
  ```java
  Instances x = new Instances(X);
  ```
* **Explanation:**
  * Raw network packets are transformed into 23 flow statistical features (e.g., Flow Duration, Packet Lengths, Inter-Arrival Times).

---

### Line 2: Stage 1 (\(H_1\)) Binary Classification
* **Pseudocode:**
  ```text
  2: H_1 classifies x to obtain y_1 and x_0, where y_1 in y
  ```
* **Java Implementation:**
  ```java
  double h1Prediction = H_1.classifyInstance(h1Instance);
  String h1Label = h1DatasetHeader.classAttribute().value((int) h1Prediction);

  if ("NEW".equals(h1Label)) {
      y_1.add(currentSample); // Caught early by H_1
  } else {
      x_0.add(currentSample); // Passed to H_2
  }
  ```
* **Explanation:**
  * \(H_1\) acts as an early bouncer.
  * \(y_1\): Samples that \(H_1\) identifies as clearly anomalous (New Class). These bypass Stage 2 entirely.
  * \(x_0\): Samples that \(H_1\) considers plausible known traffic. These move forward to Stage 2.

---

### Line 3 & Line 4: Stage 2 (\(H_2\)) Confidence Difference Calculation
* **Pseudocode:**
  ```text
  3: H_2 outputs the confidence set lt = {t_1, t_2, ..., t_k} for each sample l in x_0
  4: Calculate the difference between maximum confidence and minimum confidence in lt: beta_l = t_max - t_min
  ```
* **Java Implementation:**
  ```java
  double[] lt = H_2.distributionForInstance(l); // Probabilities for classes z_1 to z_k
  double beta_l = t_max - t_min;                // CfDmax calculation
  ```
* **Explanation:**
  * For each sample \(l \in x_0\), \(H_2\) outputs the probability distribution across all \(k\) known classes.
  * \(\beta_l = t_{\max} - t_{\min}\) represents how decisive the model is:
    * If one class has 96% and all others have ~1%, \(\beta_l \approx 0.95\) (High confidence).
    * If probabilities are evenly split (e.g., 35% vs 30%), \(\beta_l \approx 0.05\) (High uncertainty / anomaly).

---

### Lines 5 to 19: Threshold Decision & Known Class Assignment
* **Pseudocode:**
  ```text
  5: for l in x_0 do
  6:    if beta_l > beta then
  7:       l in z
  8:       Find the subscript index j of the maximum confidence in the confidence set lt of l
  9:       if j = 1 then
  10:         l in z_1
  11:      else if j = 2 then
  12:         l in z_2
  13:      else if j = k then
  14:         l in z_k
  15:      end if
  16:   else
  17:      l in y_2
  18:   end if
  19: end for
  ```
* **Java Implementation:**
  ```java
  if (beta_l > beta) {
      allKnownZ.add(l); // Belongs to Known classes (z)
      String targetClass = knownClassNames.get(maxIndexJ);
      z_partitions.get(targetClass).add(l); // Belongs to specific class z_j
  } else {
      y_2.add(l); // Belongs to New Class (y_2)
  }
  ```
* **Explanation:**
  * **If \(\beta_l > \beta\):** The flow is confidently recognized as a known application. It is assigned to the specific known class \(z_j\) corresponding to the maximum probability index \(j\).
  * **If \(\beta_l \le \beta\):** Even though \(H_1\) thought this flow might be known, \(H_2\)'s low confidence exposes it as a subtle Zero-Day threat. It is classified into **\(y_2\)**.

---

### Line 20: Combine All Detected Unknown Threats (\(y = y_1 \cup y_2\))
* **Pseudocode:**
  ```text
  20: y = y_1 U y_2
  ```
* **Java Implementation:**
  ```java
  List<Instance> y = new ArrayList<>();
  y.addAll(y_1);
  y.addAll(y_2);
  ```
* **Explanation:**
  * The final set of detected Zero-Day/Unknown attacks is the union of:
    * \(y_1\): Obvious threats caught early by \(H_1\).
    * \(y_2\): Deceptive threats caught by \(H_2\)'s adaptive threshold test.

---

## 4. How All 3 Algorithms Form the Complete System

```
                  ┌───────────────────────────────────────────────┐
                  │                 Dataset U                     │
                  └───────────────────────┬───────────────────────┘
                                          │
                                          ▼
                         ┌─────────────────────────────────┐
                         │   ALGORITHM 2 (Pseudo-Unknown)  │
                         │   Uses One-Class SVMs           │
                         └──────────────┬──────────────────┘
                                        │ Outputs: M and M_2
                                        ▼
                         ┌─────────────────────────────────┐
                         │   ALGORITHM 1 (ATS Engine)      │
                         │   Dynamically calculates alpha  │
                         └──────────────┬──────────────────┘
                                        │ Output: beta = alpha
                                        ▼
                  ┌───────────────────────────────────────────────┐
                  │   ALGORITHM 3 (Cascade Classifier)            │
                  │   Stage 1 (H_1) ──> Stage 2 (H_2 + beta)     │
                  └──────────────┬──────────────────┬─────────────┘
                                 │                  │
                                 ▼                  ▼
                         Known Classes (z)    Unknown Zero-Days (y)
```
