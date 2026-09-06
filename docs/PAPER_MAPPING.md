# CCS-UTD Paper Mapping

This document explains how our Java-based college implementation maps to the original IEEE research paper: *"New Class Detection in Network Traffic Classification Using Confidence Information Embedded Cascade Structure"*.

## 1. Feature Extraction
*   **Paper:** Extracts 131 statistical and conditional-frequency features from raw packets.
*   **Our Implementation:** **Adapted**. We use the 23 derived flow-level features already present in the provided ARFF dataset (e.g., `duration`, `mean_fiat`, `flowPktsPerSecond`). We do not recreate the packet-level features as raw pcaps were not provided.

## 2. Feature Selection
*   **Paper:** Uses Pearson Correlation Coefficient (PCC) and Random Forest ranking to select ~20 features from 131.
*   **Our Implementation:** **Adapted / Simplified**. Since we start with 23 features, the heavy reduction is less critical. However, we will still implement PCC correlation checks to remove highly redundant features to honor the methodology.

## 3. Classifier H2 (Multi-class Known Classifier)
*   **Paper:** Uses Random Forest trained on Known Classes.
*   **Our Implementation:** **Reproduced directly**. We use Weka's Random Forest implementation trained exclusively on KnownC data. It outputs probability distributions used for confidence metrics.

## 4. Confidence Metrics (Cfmax, Cfmin, CfDmax)
*   **Paper:** Calculates CfDmax = Cfmax - Cfmin based on H2 outputs.
*   **Our Implementation:** **Reproduced directly**.

## 5. Unlabelled Data & Candidate Selection (OC-SVM)
*   **Paper:** Trains one One-Class SVM per known class to filter unlabelled traffic and identify suspicious candidates.
*   **Our Implementation:** **Reproduced directly**. We use Weka's LibSVM (One-Class mode) to build these models and filter the unlabelled pool.

## 6. Pseudo-Negative Selection & Adaptive Threshold (ATS)
*   **Paper:** Uses S progressive updates, calculating the 10th percentile of CfDmax, and sets $\alpha = mean - std$.
*   **Our Implementation:** **Reproduced directly**. We will dynamically generate the threshold $\alpha$ using the exact progressive update logic (default $S=15, \alpha_0=0.9$). Actual NewC labels will *never* be leaked into H1 training.

## 7. Classifier H1 (Binary Detector)
*   **Paper:** Trained on KnownC and selected pseudo-negative samples.
*   **Our Implementation:** **Reproduced directly**.

## 8. Final Testing Cascade
*   **Paper:** H1 -> H2. Compares CfDmax to $\beta$ ($\beta = \alpha$).
*   **Our Implementation:** **Reproduced directly**.
