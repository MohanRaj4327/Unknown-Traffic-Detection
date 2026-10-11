# Algorithm 2: Selection for Pseudo Unknown Class Samples - Complete Flow & Code Explanation

This document explains **Algorithm 2** from the research paper:
> *"Unknown Network Traffic Detection via Two-Stage Cascade Classifier with Adaptive Threshold"*

The code implementation for this algorithm is saved in:
👉 `algorithms/algoflow 2/PseudoUnknownSelection_Algorithm2.java`

---

## 1. Algorithm Overview & Core Problem
Supervised machine learning algorithms (like Random Forest) cannot be trained on **Zero-Day / Unknown** attacks because there are no labelled examples of them beforehand!

**Algorithm 2 solves this problem:**
It uses **One-Class Support Vector Machines (OC-SVM)** and an unlabelled network flow dataset \(U\) to automatically extract **"Pseudo Unknown" (fake unknown)** samples without requiring manual labels.

* High-confidence known samples are sorted into **\(M_2\)**.
* High-confidence pseudo-unknown candidates are sorted into **\(M\)**.
* These two sets (\(M_2\) and \(M\)) are directly used as inputs to **Algorithm 1**!

---

## 2. Input / Output / Requirements

| Parameter | Type / Source | Meaning & Role |
| :--- | :--- | :--- |
| **Require: \(\text{KnownC}_k\)** | Set of subsets \(\{z_1, z_2, \dots, z_k\}\) | Training data for each of the \(k\) known traffic classes. |
| **Require: \(\alpha\)** | Float (e.g., 0.90) | The confidence threshold (\(\text{CfDmax}\) threshold). |
| **Require: \(\text{RF}_0\)** | Random Forest Model | Initial model trained on known traffic classes. |
| **Input: \(U\)** | Dataset | Unlabelled network traffic flow dataset. |
| **Output: \(M\)** | Dataset | Pseudo-unknown class samples whose \(\text{CfDmax} > \alpha\). |
| **Output: \(M_2\)** | Dataset | Confirmed known class flows. |

---

## 3. Step-by-Step Line Mapping (Pseudocode vs Java Code)

### Lines 1 to 3: Train One-Class SVM for Each Known Class
* **Pseudocode:**
  ```text
  1: for i = 1 to k do
  2:    Train with KnownC z_i to obtain OC-SVM_i
  3: end for
  ```
* **Java Implementation:**
  ```java
  for (int i = 0; i < k; i++) {
      Instances z_i = knownClassSubsets.get(i);
      svm_model ocSvm_i = trainOneClassSVM(z_i);
      ocSvmModels.add(ocSvm_i);
  }
  ```
* **Explanation:**
  * For each known class \(i\), an individual \(\text{OC-SVM}_i\) model is trained solely on samples of that class.
  * \(\text{OC-SVM}\) learns a tight hypersphere boundary around the normal behavior of that specific class.

---

### Lines 4 to 6: Classify Unlabelled Traffic \(U\)
* **Pseudocode:**
  ```text
  4: for i = 1 to k do
  5:    Classify U with OC-SVM_i and obtain U_{i1} and U_{i-1}
  6: end for
  ```
* **Java Implementation:**
  ```java
  // For each model ocSvm_i:
  // If prediction == +1 => added to U_inlier (U_{i1})
  // If prediction == -1 => added to U_outlier (U_{i-1})
  ```
* **Explanation:**
  * \(U_{i1}\): Flows in \(U\) accepted as belonging to class \(i\) (+1 inliers).
  * \(U_{i-1}\): Flows in \(U\) rejected by class \(i\) (-1 outliers).

---

### Lines 7 to 14: Partition into \(M_2\) (Definite Knowns) and \(M_3\) (Ambiguous Conflicts)
* **Pseudocode:**
  ```text
  7: M_1 = U_{11} U U_{21} U U_{31} U ... U U_{k1}
  8: for each m in M_1 do
  9:    if m only in U_{11} \lor U_{21} \lor U_{31} \lor ... \lor U_{k1} then
  10:      m in M_2
  11:   else
  12:      m in M_3
  13:   end if
  14: end for
  ```
* **Java Implementation:**
  ```java
  // matchCount = how many OC-SVM models accepted sample m
  if (matchCount == 1) {
      M_2_indices.add(mIndex); // Definite known
  } else {
      M_3_indices.add(mIndex); // Overlap conflict
  }
  ```
* **Explanation:**
  * \(M_1\): All flows accepted by at least one model.
  * If a sample was accepted by **only one** class model, it is unambiguous and added to **\(M_2\)**.
  * If a sample was accepted by **more than one** model, it falls in an overlap zone and is added to **\(M_3\)**.

---

### Line 15: Find Unanimous Outliers (\(M_4\))
* **Pseudocode:**
  ```text
  15: M_4 = U_{1-1} \cap U_{2-1} \cap U_{3-1} \cap ... \cap U_{k-1}
  ```
* **Java Implementation:**
  ```java
  // Find instances rejected as -1 by ALL k models
  if (rejectedByAll) {
      M_4_indices.add(mIndex);
  }
  ```
* **Explanation:**
  * \(M_4\) represents flows rejected as outliers by **all** \(k\) One-Class SVM models. These are strongly anomalous flows!

---

### Line 16: Combine Conflict and Outlier Sets (\(M_5\))
* **Pseudocode:**
  ```text
  16: M_5 = M_3 U M_4
  ```
* **Java Implementation:**
  ```java
  M_5_indices.addAll(M_3_indices);
  M_5_indices.addAll(M_4_indices);
  ```
* **Explanation:**
  * \(M_5\) brings together both the ambiguous boundary samples (\(M_3\)) and the unanimous outliers (\(M_4\)). These form our candidate pool for pseudo-unknowns.

---

### Lines 17 to 25: Confidence Filtering using \(\text{RF}_0\)
* **Pseudocode:**
  ```text
  17: RF output confidence set m_t = {t_1, t_2, ..., t_k} for each sample m in M_5
  18: Calculate the difference between maximum confidence and minimum confidence in m_t: alpha_m = t_max - t_min
  19: for each m in M_5 do
  20:    if alpha_m > alpha then
  21:       m in M
  22:    else
  23:       Discard m
  24:    end if
  25: end for
  ```
* **Java Implementation:**
  ```java
  double[] m_t = RF_0.distributionForInstance(m);
  double alpha_m = t_max - t_min; // CfDmax

  if (alpha_m > alpha) {
      M.add(m); // Added to final Pseudo-Unknown set M
  } else {
      // Discard m
  }
  ```
* **Explanation:**
  * \(\alpha_m = t_{\max} - t_{\min}\) is the **Confidence Difference (\(\text{CfDmax}\))**.
  * If \(\alpha_m > \alpha\), the model has high distinctiveness for this flow despite being in an anomalous set, making it a high-quality **Pseudo-Unknown** sample (\(M\)).
  * Otherwise, low-confidence ambiguous noise is safely discarded.

---

### Line 26: Return Output
* **Pseudocode:**
  ```text
  26: Return M, M_2
  ```
* **Java Implementation:**
  ```java
  return new Algorithm2Result(M, M_2);
  ```
* **Explanation:**
  * Returns \(M\) (Pseudo-unknown training instances) and \(M_2\) (Confirmed known training instances).

---

## 4. How Algorithm 2 Connects to Algorithm 1
Notice Line 1 and Line 2 of **Algorithm 1**:
1. *Line 1 of Algorithm 1:* "Obtain known flows \(M_2\) by Algorithm 2"
2. *Line 2 of Algorithm 1:* "Obtain subset \(M\) of unlabelled data with unknown classes whose \(\text{CfDmax} > \alpha_0\) by Algorithm 2"

This shows that **Algorithm 2 extracts the training datasets (\(M\) and \(M_2\)) that Algorithm 1 uses to dynamically calculate the adaptive threshold (\(\alpha\))!**
