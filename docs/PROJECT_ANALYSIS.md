# Project Analysis: CCS-UTD College Mini Project

## 1. Dataset Inspection & Setup

We are using the **Scenario A2 (15s NO-VPN)** dataset as the initial development dataset.

### ARFF Metadata details
**File:** `TimeBasedFeatures-Dataset-15s-NO-VPN.arff`
**Number of instances:** ~8966 records

**Exact 23 Feature Names:**
1. `duration`
2. `total_fiat`
3. `total_biat`
4. `min_fiat`
5. `min_biat`
6. `max_fiat`
7. `max_biat`
8. `mean_fiat`
9. `mean_biat`
10. `flowPktsPerSecond`
11. `flowBytesPerSecond`
12. `min_flowiat`
13. `max_flowiat`
14. `mean_flowiat`
15. `std_flowiat`
16. `min_active`
17. `mean_active`
18. `max_active`
19. `std_active`
20. `min_idle`
21. `mean_idle`
22. `max_idle`
23. `std_idle`

**Exact Class/Label Attribute:**
`class1 {BROWSING,CHAT,STREAMING,MAIL,VOIP,P2P,FT}`

## 2. Adaptation Strategy

**Important Note:** This project is a **Java-based adaptation based on the CCS-UTD paper**, *not* an exact reproduction of the authors' original source code.

*   **Dataset-based Adaptation:** The paper originally extracts 131 packet-level features. We will **not** recreate these 131 features because the provided ARFF dataset already contains 23 high-quality derived flow-level features. We will use these 23 flow features directly.
*   **Open-Set Simulation:** We will split the `class1` labels into "Known Classes" (KnownC) and held-out "New Classes" (NewC) during our train/test split.
*   **H1 & H2 Training:** 
    *   H2 (Multi-class Random Forest) will be trained **only** on KnownC classes.
    *   H1 (Binary Classifier) will be trained using KnownC samples and selected "pseudo-negative" samples obtained from an unlabelled data pool via One-Class SVM.
*   **Pseudo-negative Constraints:** We will strictly adhere to the rule that actual held-out NewC labels must *never* be used as H1 training data. They will be dynamically discovered via the ATS + OC-SVM -> RF -> CfDmax process.

## 3. Architecture

*   **Backend:** Java Spring Boot 3.2.x, Weka (ML), Spring Data JPA, PostgreSQL.
*   **Frontend:** HTML, CSS, Vanilla JS.
*   **Configurability:** ATS parameters like `alpha_0`, `S`, `percentile`, and cascade threshold `beta` will be fully configurable via `application.properties`.
