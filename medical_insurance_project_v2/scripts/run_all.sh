#!/usr/bin/env bash
set -euo pipefail
python -m src.eda
python -m src.train
python -m src.report
echo "Pipeline complete. Launching dashboard..."
streamlit run app/streamlit_app.py
