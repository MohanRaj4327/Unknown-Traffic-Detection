# Open-Set Network Traffic Detection (CCS-UTD)

This is a Java Spring Boot implementation inspired by the IEEE paper *"New Class Detection in Network Traffic Classification Using Confidence Information Embedded Cascade Structure"*.

## 🚀 Project Overview

Traditional Machine Learning models are "Closed-Set", meaning they only recognize traffic classes they were explicitly trained on. When confronted with Zero-Day or entirely new traffic protocols, they misclassify them. 

This project implements an **Open-Set Cascade Structure** that can:
1. **Accurately classify known traffic** (e.g., Browsing, Chat, VoIP).
2. **Dynamically detect and block unknown/new traffic** (e.g., P2P, File Transfer) using confidence bounds, *even if the AI has never been trained on it*.

### 🧠 The Cascade Architecture
* **H1 Binary Classifier (The Bouncer):** The first layer of defense. It evaluates incoming traffic and if it strays too far from known distributions, it flags it as an `UNKNOWN` threat and blocks it early.
* **H2 Multi-Class Classifier (The Expert):** If H1 passes the traffic as safe, H2 analyzes it and assigns it a specific known class label (e.g., "This is Chat traffic").
* **ATS (Adaptive Threshold Selection):** The system mathematically calculates the perfect strictness boundary (`alpha`) for H1 dynamically, adapting to live network environments instead of relying on human-guessed thresholds.

---

## 🛠️ Tech Stack
* **Backend:** Java 17, Spring Boot, REST APIs
* **Machine Learning:** Weka (Random Forest), LibSVM (One-Class SVM for Pseudo-Negative extraction)
* **Database:** PostgreSQL (Supabase Cloud DB) via Hibernate/JPA
* **Frontend:** HTML5, CSS3, Vanilla JS, Chart.js

---

## 📊 Dataset & Configuration
* **Source:** ISCX VPN-NonVPN traffic dataset (Scenario A2 - 15s Non-VPN).
* **Known Classes:** Browsing, Chat, Streaming, Mail, VoIP.
* **Hidden (Zero-Day) Classes:** P2P, File Transfer (FT).
* **Features:** 23 flow-based time features (reduced via Correlation Feature Selection).

---

## ⚙️ How to Run the Project

### 1. Database Setup
1. Create a free project on [Supabase](https://supabase.com).
2. Get your JDBC Connection String (Session Pooler mode - port `5432`).
3. Open `backend/src/main/resources/application.properties`.
4. Add your connection string and password to the top 3 lines.

### 2. Start the Backend Server
Run the provided PowerShell script in your terminal to automatically download Maven and boot the Spring application:
```powershell
cd backend
powershell -ExecutionPolicy Bypass -File .\run_server.ps1
```
The server will start on `http://localhost:8080` and Hibernate will automatically create the `experiment_runs` table in your cloud database.

### 3. Open the Dashboard
1. Open your File Explorer.
2. Navigate to `frontend/index.html`.
3. Double-click the file to open it in your web browser.

---

## 🧪 Testing the AI (Live Demo)
1. In the Dashboard, click on **Model Runner**.
2. Click **Run Full H1/H2 Cascade**.
3. The Java backend will dynamically extract pseudo-negatives via OC-SVM, calculate the ATS threshold, train the Random Forests, and test them against 2,300+ network flows.
4. The results, including an **~85% Normalized Accuracy** and a **98% Unknown Detection Rate**, will be rendered on a Chart.js graph and saved to your Supabase database.
5. To see the AI evaluate a single packet live, navigate to the **Live Single Prediction** tab and inject an unknown traffic sample.

---

## 📝 License & Credits
Developed as a college mini-project based on the concepts from the CCS-UTD research paper.
