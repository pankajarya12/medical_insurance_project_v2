.PHONY: install eda train report app docker test clean
install:  ; pip install -r requirements.txt
eda:      ; python -m src.eda
train:    ; python -m src.train
report:   ; python -m src.report
app:      ; streamlit run app/streamlit_app.py
docker:   ; docker compose up --build
test:     ; pytest -q
clean:    ; rm -rf mlruns models/registry reports/*.pdf app/eda_images/*.png
