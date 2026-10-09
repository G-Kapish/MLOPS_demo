# Telecom Tower Failure Prediction & Monitoring Pipeline (MLOps)

An end-to-end Machine Learning Operations (MLOps) project designed to predict telecommunication tower hardware failures within a 48-hour window. This repository provides a complete pipeline featuring model training with Scikit-Learn, experiment and drift tracking with MLflow, REST API deployment using Flask and Waitress, and automated load testing via Apache JMeter.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Prerequisites & Environment Setup](#prerequisites--environment-setup)
- [Pipeline Execution Guide](#pipeline-execution-guide)
  - [1. Train and Evaluate the Model](#1-train-and-evaluate-the-model)
  - [2. Monitor Model Drift & Log to MLflow](#2-monitor-model-drift--log-to-mlflow)
  - [3. Visualize MLflow Tracking Dashboard](#3-visualize-mlflow-tracking-dashboard)
  - [4. Serve the Prediction API](#4-serve-the-prediction-api)
- [API Verification & Testing](#api-verification--testing)
- [Load Testing with Apache JMeter](#load-testing-with-apache-jmeter)

---

## Project Overview

* **Objective:** Predict telecom tower failures (`Failure_Within_48Hrs`) based on live operational telemetry metrics including temperature, battery voltage, power consumption, fan speed, and signal quality.
* **Model Engine:** Random Forest Classifier (`RandomForestClassifier`) trained on historical operational records.
* **Experiment Tracking & Drift Detection:** Evaluates production telemetry against a performance threshold (80% accuracy) and logs run metrics to MLflow.
* **Serving Layer:** RESTful inference service built with Flask and production-served using multi-threaded Waitress WSGI.
* **Performance Testing:** Load and stress testing configured through Apache JMeter to analyze throughput, latency, and system stability under concurrency.

---

## Directory Structure

```text
MLOPS/
├── .gitignore                                 # Git ignore patterns
├── Telecom_Tower_Failure_Dataset_10000-1.xlsx # Baseline training dataset
├── new_tower_telemetry.xlsx                  # Production telemetry dataset for drift monitoring
├── predictive analysis.py                     # Model training & artifact serialization script
├── monitor_model.py                           # Model drift monitoring & MLflow logging script
├── app.py                                     # Flask REST API inference application
├── metrics.json                               # Exported evaluation metrics artifact
├── telecom_tower_model.pkl                    # Serialized machine learning model artifact
└── apache-jmeter-5.6.3/                       # Apache JMeter load testing suite

```

---

## Prerequisites & Environment Setup

* **Operating System:** Windows 10 / 11, macOS, or Linux
* **Python:** 3.10+ (Python 3.13 supported)
* **Java:** OpenJDK 17 LTS or 21 LTS (required for Apache JMeter)

### 1. Clone the Repository

```cmd
git clone [https://github.com/G-Kapish/MLOPS_demo.git](https://github.com/G-Kapish/MLOPS_demo.git)
cd MLOPS_demo

```

### 2. Create and Activate Virtual Environment

* **Command Prompt (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate

```


* **PowerShell:**
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1

```



### 3. Install Dependencies

```cmd
python -m pip install --upgrade pip
pip install pandas scikit-learn joblib mlflow evidently flask waitress matplotlib openpyxl

```

---

## Pipeline Execution Guide

### 1. Train and Evaluate the Model

Trains the Random Forest classifier on `Telecom_Tower_Failure_Dataset_10000-1.xlsx`, performs stratified validation, and exports `telecom_tower_model.pkl` and `metrics.json`:

```cmd
python "predictive analysis.py"

```

### 2. Monitor Model Drift & Log to MLflow

Evaluates `telecom_tower_model.pkl` against `new_tower_telemetry.xlsx`, computes classification metrics, flags degradation if accuracy drops below the 80% threshold, and logs the run to MLflow:

```cmd
python monitor_model.py

```

### 3. Visualize MLflow Tracking Dashboard

Launch the interactive web UI to inspect run parameters, classification metrics, and drift flags:

```cmd
mlflow ui --port 5001

```

Open your browser and navigate to: `http://127.0.0.1:5001`

---

### 4. Serve the Prediction API

#### Development Mode (Flask Built-in Server)

```cmd
python app.py

```

#### Production Mode (Multi-Threaded Waitress WSGI)

For concurrent requests and load testing, run the application using Waitress to avoid connection dropouts:

```cmd
waitress-serve --port=5000 app:app

```

*(Optional tuning for higher concurrency: `waitress-serve --port=5000 --threads=16 --connection-limit=1000 --backlog=1024 app:app`)*

---

## API Verification & Testing

With the server running on port `5000`, test the endpoints using a separate terminal window:

### Health Check (`GET /`)

* **PowerShell:**
```powershell
Invoke-RestMethod [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

```


* **cURL (Command Prompt / Linux / macOS):**
```cmd
curl [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

```



### Inference Request (`POST /predict`)

* **PowerShell:**
```powershell
Invoke-RestMethod -Uri [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict) -Method Post -ContentType "application/json" -Body '{"Temperature_C": 35.6, "Battery_Voltage": 50.9, "Power_Consumption_W": 2962, "Signal_Strength_Percent": 56, "Fan_Speed_RPM": 2838, "Humidity_Percent": 52, "Traffic_Load": 2612, "Tower_Age_Years": 4}'

```


* **cURL (Command Prompt):**
```cmd
curl.exe -X POST [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict) -H "Content-Type: application/json" -d "{\"Temperature_C\": 35.6, \"Battery_Voltage\": 50.9, \"Power_Consumption_W\": 2962, \"Signal_Strength_Percent\": 56, \"Fan_Speed_RPM\": 2838, \"Humidity_Percent\": 52, \"Traffic_Load\": 2612, \"Tower_Age_Years\": 4}"

```



**Expected JSON Response:**

```json
{
  "count": 1,
  "predictions": [0]
}

```

---

## Load Testing with Apache JMeter

Stress test the inference endpoint under concurrent client traffic using Apache JMeter.

### 1. Launch JMeter

Navigate to the JMeter binary folder and execute the startup script:

```cmd
cd apache-jmeter-5.6.3\bin
jmeter.bat

```

*(The JMeter Graphical User Interface will appear.)*

### 2. Configure the Test Plan

1. **Add a Thread Group:**
* Right-click **Test Plan** > **Add** > **Threads (Users)** > **Thread Group**.
* Set **Number of Threads (users):** `20`
* Set **Ramp-Up period (seconds):** `5`
* Set **Loop Count:** `10`


2. **Add HTTP Header Manager:**
* Right-click **Thread Group** > **Add** > **Config Element** > **HTTP Header Manager**.
* Click **Add** at the bottom:
* **Name:** `Content-Type`
* **Value:** `application/json`




3. **Add HTTP Request Sampler:**
* Right-click **Thread Group** > **Add** > **Sampler** > **HTTP Request**.
* **Protocol:** `http`
* **Server Name or IP:** `127.0.0.1`
* **Port Number:** `5000`
* **HTTP Method:** `POST`
* **Path:** `/predict`
* Under the **Body Data** tab, paste the payload:
```json
{
  "Temperature_C": 35.6,
  "Battery_Voltage": 50.9,
  "Power_Consumption_W": 2962,
  "Signal_Strength_Percent": 56,
  "Fan_Speed_RPM": 2838,
  "Humidity_Percent": 52,
  "Traffic_Load": 2612,
  "Tower_Age_Years": 4
}

```




4. **Add Listeners to View Results:**
* Right-click **Thread Group** > **Add** > **Listener** > **View Results Tree** (to inspect individual request/response bodies).
* Right-click **Thread Group** > **Add** > **Listener** > **Summary Report** (to monitor throughput, latency, and error percentage).



### 3. Run the Test

1. Make sure your Waitress or Flask server is active.
2. Click the green **Start** button (play icon) on the top toolbar.
3. Select **Summary Report** to review performance statistics under concurrent load.
