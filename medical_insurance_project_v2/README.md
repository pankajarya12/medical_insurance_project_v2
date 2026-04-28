# 🩺 Medical Insurance Cost Prediction — Advanced ML Project

> Production-style end-to-end ML pipeline with **EDA · Multi-model training · MLflow tracking · Local Model Registry · Streamlit Dashboard · Automated PDF Reports · Dockerised one-command launch**.

---

## ✨ Features (NEW in this version)

| # | Feature | File |
|---|---------|------|
| 1 | 🐳 **One-command Docker launch** (app + trainer + MLflow UI) | `Dockerfile`, `docker-compose.yml` |
| 2 | 📑 **Automated PDF reports** (dataset, model card, EDA, comparison table) | `src/report.py` |
| 3 | 🗂 **Model Registry UI** (versioning, promote-to-active, delete) | `src/registry.py` + `app/streamlit_app.py` |
| 4 | 📂 **Real CSV upload** with schema validation (Pydantic) | `src/data_loader.py` |
| 5 | 🛡 **Hardened file-path handling** (path-traversal blocked) | `src/paths.py` |
| 6 | 🩻 **Medical-themed advanced visuals** (Plotly + Seaborn) | `src/eda.py` |
| 7 | 🚀 **Multi-page Streamlit dashboard** | `app/streamlit_app.py` |
| 8 | 🧪 **Pytest test suite** | `tests/` |
| 9 | 🎯 **Engineered features** (`smoker_bmi`, `age_smoker`, BMI category, age group) | `src/preprocessing.py` |
| 10 | 📊 **MLflow experiment tracking** | `src/train.py` |

---

## 📁 Project Structure

```
medical_insurance_project/
├── app/
│   ├── streamlit_app.py        # 7-page dashboard (Home/Predict/Dataset/EDA/Train/Registry/Reports)
│   └── eda_images/             # auto-generated PNGs
├── data/
│   ├── medical_insurance.csv   # default dataset (1338 rows)
│   └── uploads/                # user-uploaded CSVs land here (sandboxed)
├── models/
│   ├── model_comparison.csv
│   └── registry/               # versioned models  <name>__v<n>/{model.pkl, meta.json}
│       └── active.json         # pointer to currently-active model
├── reports/                    # generated PDFs
├── mlruns/                     # MLflow tracking store
├── src/
│   ├── paths.py                # 🛡 hardened path helpers
│   ├── data_loader.py          # CSV load + Pydantic validation + safe upload
│   ├── preprocessing.py        # feature engineering + ColumnTransformer
│   ├── eda.py                  # 6 medical-themed plots
│   ├── train.py                # 6-model GridSearch + MLflow + registry
│   ├── registry.py             # JSON-backed model registry
│   └── report.py               # fpdf2 PDF builder
├── tests/                      # pytest suite
├── scripts/run_all.sh          # full pipeline shell helper
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Option A — Docker (recommended)

```bash
docker compose up --build
```

This will:
1. **trainer** service runs EDA → trains all models → registers best → builds PDF
2. **app** service serves the Streamlit dashboard at <http://localhost:8501>
3. **mlflow** service serves the experiment UI at <http://localhost:5000>

### Option B — Local Python

```bash
pip install -r requirements.txt
python -m src.eda          # generate visuals
python -m src.train        # train + register best model
python -m src.report       # build PDF
streamlit run app/streamlit_app.py
```

Or simply:  `make docker`  /  `make app`  /  `make test`

---

## 📚 Hindi Code Explanation (हर फाइल का काम)

### `src/paths.py` — रास्ते की सुरक्षा
सारी डायरेक्टरीज़ (`data/`, `models/`, `reports/`) यहाँ define होती हैं।  
`safe_path()` function user input को resolve करके check करता है कि वो allowed folder से बाहर तो नहीं जा रहा — इससे `../../etc/passwd` जैसे path-traversal attacks block होते हैं।

### `src/data_loader.py` — डेटा लोडर
- `InsuranceRow` (Pydantic) हर row का schema check करता है — गलत data train होने से पहले reject।
- `save_uploaded_csv()` user upload को sanitised filename के साथ `data/uploads/` में रखता है, validate करता है, fail होने पर delete कर देता है।

### `src/preprocessing.py` — फीचर इंजीनियरिंग
- `smoker_bmi` = smoker × bmi (smoker + obese का double impact)
- `age_smoker` = smoker × age
- `bmi_category`, `age_group` — clinical buckets
- `build_preprocessor()` — numeric को scale, categorical को one-hot encode।

### `src/eda.py` — मेडिकल विज़ुअल्स
Seaborn + Matplotlib से 6 graphs:
1. Distributions (age/bmi/charges) 2. Smoker boxplot  
3. BMI×Smoker scatter (danger zone)  4. Age trend (smoker vs non)  
5. Region heatmap  6. Correlation matrix.

### `src/train.py` — ट्रेनिंग
6 models train होते हैं (Linear, Ridge, Lasso, RandomForest, GBR, XGBoost), 5-fold CV होता है, हर run MLflow में log होता है, और best model **registry में auto-register** हो जाता है।

### `src/registry.py` — मॉडल रजिस्ट्री
हर मॉडल का अपना folder + `meta.json` (algorithm, metrics, params, dataset, timestamp)।  
`set_active()` से production version switch कर सकते हैं — Streamlit prediction page हमेशा active model use करता है।

### `src/report.py` — PDF रिपोर्ट
`fpdf2` से एक complete PDF बनती है: dataset summary, descriptive stats, active model card, comparison table, सारे EDA graphs।

### `app/streamlit_app.py` — डैशबोर्ड
7 pages: Home (KPIs), Predict (form), Dataset (upload + preview), EDA (gallery), Train (button), Registry (promote/delete), Reports (download PDF)।

### `Dockerfile` + `docker-compose.yml`
तीन services — trainer (एक बार चलकर मॉडल बनाता है), app (Streamlit), mlflow (tracking UI)। एक command `docker compose up` से सब चालू।

### `tests/`
pytest से paths की security, preprocessing, और registry round-trip verify होती है।

---

## 🔐 Security Notes
- All file uploads → `data/uploads/` only; filenames sanitised, extensions whitelisted, path traversal blocked.
- CSVs schema-validated before being trusted.
- Streamlit `enableXsrfProtection=true`, max upload 50 MB.

---

## 📈 Best Model (default dataset)

`Ridge Regression` typically wins with **R² ≈ 0.86** on test split (the previous 0.955 was on a leaky split — we now use the proper 80/20 holdout).  XGBoost and GradientBoosting are very close behind.

Switch the active model anytime from the **Registry** page — predictions will instantly use the newly chosen version.

---

Made with ❤️ for healthcare ML enthusiasts.
