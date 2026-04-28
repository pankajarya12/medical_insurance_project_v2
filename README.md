# 🩺 Medical Insurance Cost Prediction — Advanced ML Project

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-ScikitLearn-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue)
![Docker](https://img.shields.io/badge/Container-Docker-blue?logo=docker)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

</p>

> A **production-grade end-to-end Machine Learning pipeline** to predict medical insurance costs with full lifecycle support — from data validation to deployment.

---

## 🚀 Overview

This project is designed to simulate a **real-world ML system**.  
It covers the complete pipeline including:

* Data ingestion & validation  
* Feature engineering  
* Multi-model training  
* MLflow experiment tracking  
* Model registry (version control)  
* Streamlit dashboard  
* Automated PDF reports  
* Dockerized deployment  

---

## 🎥 Live Demo

<p align="center">
  <img src="app/screenshots/demo.gif" width="800"/>
</p>

---

## 📸 Screenshots

### 🏠 Dashboard Home
<img width="1902" height="862" alt="image" src="https://github.com/user-attachments/assets/8a71da17-4291-4dd4-9f29-a2692f779f62" />
--

### 🔮 Prediction Page
<img width="1887" height="837" alt="image" src="https://github.com/user-attachments/assets/519b4c4e-51d5-4f6e-8db8-1820d0470121" />
--

### Dataset Manager
<img width="1896" height="841" alt="image" src="https://github.com/user-attachments/assets/b52dd409-47cc-4618-8f01-a4e8da3cf573" />
---

### 📊 EDA Insights
<img width="1897" height="857" alt="image" src="https://github.com/user-attachments/assets/a81b6475-f865-4716-9bc7-8eba703154b7" />
---

### Train Models
<img width="1900" height="852" alt="image" src="https://github.com/user-attachments/assets/36d26484-a7d3-44f4-8ee7-95e6e7204668" />
---

### 🗂 Model Registry
<img width="1890" height="790" alt="image" src="https://github.com/user-attachments/assets/d09886ff-b5c6-4d68-aa46-cdb164c66dc0" />
---

### 📑 Reports
<img width="1898" height="707" alt="image" src="https://github.com/user-attachments/assets/dc7df198-bad6-407b-a663-93093f672fed" />
----


> 📌 Store images inside: `app/screenshots/`

---

## 📊 Architecture Diagram

```mermaid
flowchart LR

A[User / CSV Upload] --> B[Data Validation - Pydantic]
B --> C[Feature Engineering]
C --> D[EDA Analysis]

D --> E[Model Training]
E --> F[MLflow Tracking]

F --> G[Model Registry]
G --> H[Active Model]

H --> I[Streamlit Dashboard]
I --> J[Prediction Output]

E --> K[PDF Report Generator]

subgraph Docker Services
    E
    F
    I
end
```

---

## ✨ Features

| Feature                 | Description                                   |
|------------------------|-----------------------------------------------|
| 🐳 Docker Deployment    | Run entire system with one command            |
| 📊 EDA Visualizations   | Medical-style insights using Seaborn & Plotly |
| 🤖 Multi-Model Training | 6 ML models with hyperparameter tuning        |
| 📈 MLflow Tracking      | Track experiments, metrics, parameters        |
| 🗂 Model Registry       | Versioned models with active model switching  |
| 📑 PDF Reports          | Auto-generated model + dataset reports        |
| 📂 CSV Upload           | Real dataset upload with validation           |
| 🔐 Security             | Safe file handling + path protection          |
| 🌐 Streamlit Dashboard  | 7-page interactive UI                         |
| 🧪 Testing              | Pytest-based test suite                       |

---

## 📁 Project Structure

## 📁 Project Structure (With Explanation)

```
medical_insurance_project/
│
├── app/                          # 🌐 Streamlit web application
│   └── streamlit_app.py          # 🎛️ Multi-page dashboard (Predict, EDA, Train, Registry, Reports)
│
├── data/                         # 📂 Data storage
│   ├── medical_insurance.csv     # 📊 Default dataset
│   └── uploads/                  # 📥 User-uploaded CSV files (validated & safe)
│
├── models/                       # 🤖 Trained models storage
│   └── registry/                 # 🗂 Model registry (versioned models + active model)
│
├── reports/                      # 📑 Auto-generated PDF reports
├── mlruns/                       # 📈 MLflow experiment tracking logs
│
├── src/                          # 🧠 Core ML pipeline (main logic)
│   ├── paths.py                  # 🔐 Secure file paths (prevents path traversal)
│   ├── data_loader.py            # 📥 Load & validate data (Pydantic schema)
│   ├── preprocessing.py          # 🧹 Feature engineering & transformations
│   ├── eda.py                    # 📊 Data visualization (EDA plots)
│   ├── train.py                  # 🤖 Model training & evaluation
│   ├── registry.py               # 🗂 Model versioning & management
│   └── report.py                 # 📑 PDF report generation
│
├── tests/                        # 🧪 Unit tests (pytest)
│
├── Dockerfile                    # 🐳 Docker image configuration
├── docker-compose.yml            # ⚙️ Multi-service setup (app + trainer + MLflow)
├── requirements.txt              # 📦 Python dependencies
└── README.md                     # 📘 Project documentation
```


## 📊 System Structure Overview

```mermaid
flowchart TD

A[Project Root]

A --> B[App Layer - Streamlit]
A --> C[Data Layer]
A --> D[Model Layer]
A --> E[ML Pipeline]
A --> F[Tracking]
A --> G[Reports]
A --> H[Testing]
A --> I[Deployment]

B --> B1[streamlit_app.py]
C --> C1[data and uploads]
D --> D1[model registry]
E --> E1[src modules]
F --> F1[mlruns MLflow]
G --> G1[pdf reports]
H --> H1[pytest]
I --> I1[docker setup]
```

---

## 🔄 ML Pipeline

1. Data Loading & Validation  
2. Feature Engineering  
3. Exploratory Data Analysis  
4. Model Training & Evaluation  
5. Experiment Tracking (MLflow)  
6. Model Registration  
7. Deployment (Streamlit)  
8. Report Generation  

---

## 🧠 Feature Engineering

* `smoker_bmi` → captures combined risk  
* `age_smoker` → interaction feature  
* BMI categories  
* Age groups  

✔ Improves prediction accuracy  

---

## 📊 Model Training

Models used:

* Linear Regression  
* Ridge Regression ⭐ (Best)  
* Lasso Regression  
* Random Forest  
* Gradient Boosting  
* XGBoost  

### 🏆 Best Model
- **Ridge Regression**
- **R² ≈ 0.86**

---

## 📈 MLflow Tracking

MLflow is used to track and manage experiments across the ML lifecycle:

- Logs parameters, metrics, and artifacts  
- Enables comparison of multiple experiments  
- Ensures reproducibility of results  


### 🔍 Workflow

```mermaid
flowchart LR

A[Training] --> B[MLflow]
B --> C[Parameters]
B --> D[Metrics]
B --> E[Artifacts]
C --> F[Comparison]
D --> F
E --> F
F --> G[Reproducibility]
```  

---

## 🗂 Model Registry

A built-in model registry enables seamless model lifecycle management:

- Version-controlled model storage  
- Metadata tracking (accuracy, parameters, timestamps)  
- Dynamic active model switching  

The Streamlit dashboard always uses the currently active model for real-time predictions.
---

## 🌐 Streamlit Dashboard

Pages:

* 🏠 Home  
* 🔮 Predict  
* 📂 Dataset  
* 📊 EDA  
* ⚙️ Train  
* 🗂 Registry  
* 📑 Reports  

---

## 📑 Automated Reports

Generated using `fpdf2`, including:

- 📊 Dataset summary  
- 📈 Statistical insights  
- 🤖 Model comparison  
- 📉 EDA visuals  

## 🐳 Docker Setup

```bash
docker compose up --build
```

### Services:

* Trainer  
* Streamlit App → localhost:8501  
* MLflow UI → localhost:5000  

---

## 💻 Local Setup

```bash
pip install -r requirements.txt

python -m src.eda
python -m src.train
python -m src.report

streamlit run app/streamlit_app.py
```

---

## 🔐 Security

Security is enforced throughout the data pipeline:

- Validates uploaded files  
- Applies schema validation (Pydantic)  
- Stores files in restricted directories  
- Prevents path traversal attacks  

### 🔒 Workflow

```mermaid
flowchart LR

A[Upload] --> B[File Check]
B --> C[Schema Validation]
C --> D[Safe Storage]
D --> E[Path Protection]
E --> F[Secure Usage]
```

---

## 🧪 Testing

```bash
pytest
```

---

## 📊 Use Cases Overview

```mermaid
flowchart LR

A[🩺 ML System] --> B[🏥 Prediction]
A --> C[🏢 Pricing]
A --> D[📊 Analytics]
A --> E[🤖 ML Learning]
A --> F[📈 Business]
A --> G[🎓 Portfolio]
A --> H[🔐 Security]
A --> I[🌐 Dashboard]
```

## 📊 Use Cases Diagram

```mermaid
flowchart LR

A[🩺 Medical Insurance ML System]

A --> B[🏥 Prediction]
B --> B1[💰 Estimate cost]

A --> C[🏢 Pricing]
C --> C1[📊 Risk analysis]
C --> C2[💵 Premium adjustment]
C --> C3[⚠️ High-risk users]

A --> D[📊 Analytics]
D --> D1[🚬 Smoking impact]
D --> D2[⚖️ BMI vs cost]
D --> D3[🌍 Regional trends]

A --> E[🤖 ML System]
E --> E1[⚙️ Pipeline]
E --> E2[📈 MLflow]
E --> E3[🚀 Deployment]

A --> F[📈 Business]
F --> F1[📊 Forecasting]
F --> F2[📦 Optimization]
F --> F3[💼 Planning]

A --> G[🎓 Portfolio]
G --> G1[💻 Skills]
G --> G2[🏗 System design]

A --> H[🔐 Security]
H --> H1[📥 Upload safety]
H --> H2[✔️ Validation]

A --> I[🌐 Dashboard]
I --> I1[📂 Upload]
I --> I2[📊 Visuals]
I --> I3[🤖 Train]
I --> I4[📑 Reports]
```

## 📌 Conclusion

End-to-end ML system covering:

**Data → Model → Deployment → Monitoring**

---

## 👨‍💻 Author

👨‍💻 **Pankaj Kumar**  
🎯 Aspiring Data Scientist  

💼 Skills:  
📊 Machine Learning | 📈 Data Analysis | 🧠 AI  

🚀 Focus:  
Building scalable, production-ready ML systems 

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
