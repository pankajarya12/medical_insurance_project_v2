"""
Streamlit Dashboard - Medical Insurance Cost Predictor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.paths import list_datasets, REPORTS_DIR
from src import registry
from src.data_loader import load_dataset, save_uploaded_csv
from src.preprocessing import add_features

st.set_page_config(page_title="Medical Insurance ML", page_icon="🩺", layout="wide")

# -------- Sidebar nav --------
PAGES = ["🏠 Home", "🔮 Predict", "📂 Dataset", "📊 EDA",
         "⚙️ Train", "🗂 Registry", "📑 Reports"]
page = st.sidebar.radio("Navigate", PAGES)

st.sidebar.markdown("---")
active = registry.get_active()

if active:
    st.sidebar.success(f"Active model:\n**{active[0]} v{active[1]}**")
else:
    st.sidebar.warning("No active model. Train one!")

# ============ HOME ============
if page == "🏠 Home":
    st.title("🩺 Medical Insurance Cost Prediction")

    try:
        df = load_dataset()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Patients", f"{len(df):,}")
        c2.metric("Avg charges", f"${df['charges'].mean():,.0f}")
        c3.metric("Smokers", f"{(df['smoker']=='yes').mean()*100:.1f}%")
        c4.metric("Avg BMI", f"{df['bmi'].mean():.1f}")

        fig = px.histogram(
            df,
            x="charges",
            nbins=40,
            color="smoker",
            color_discrete_map={"yes": "#dc2626", "no": "#0d9488"},
            title="Charges distribution by smoking status"
        )

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Dataset load error: {e}")

# ============ PREDICT ============
elif page == "🔮 Predict":
    st.title("🔮 Predict Insurance Cost")

    if not active:
        st.error("No active model. Train one first.")
        st.stop()

    model, _ = registry.load_active()

    with st.form("pred"):
        c1, c2, c3 = st.columns(3)

        age = c1.slider("Age", 18, 80, 35)
        bmi = c2.slider("BMI", 15.0, 50.0, 27.0)
        children = c3.slider("Children", 0, 5, 1)

        sex = c1.selectbox("Sex", ["male", "female"])
        smoker = c2.selectbox("Smoker", ["no", "yes"])
        region = c3.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

        ok = st.form_submit_button("Predict 💰")

    if ok:
        row = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": 0
        }])

        from src.preprocessing import NUMERIC, CATEGORICAL
        X = add_features(row)[NUMERIC + CATEGORICAL]

        pred = float(model.predict(X)[0])

        st.success(f"Estimated charge: ${pred:,.2f}")
        st.progress(min(pred / 60000, 1.0))

# ============ DATASET ============
elif page == "📂 Dataset":
    st.title("📂 Dataset Manager")

    up = st.file_uploader("Upload CSV", type=["csv"])

    if up:
        try:
            saved = save_uploaded_csv(up.getvalue(), up.name)
            st.success(f"Saved: {saved.name}")
        except Exception as e:
            st.error(str(e))

    datasets = list_datasets()

    if datasets:
        choice = st.selectbox("Preview dataset", datasets)
        df = load_dataset(choice, validate=False)
        st.dataframe(df.head(50))
    else:
        st.info("No datasets available")

# ============ EDA ============
elif page == "📊 EDA":
    st.title("📊 EDA Dashboard")

    from src.eda import generate_all, EDA_DIR

    if st.button("Generate Plots"):
        with st.spinner("Generating..."):
            generate_all()
        st.success("Done")

    imgs = sorted(EDA_DIR.glob("*.png"))

    if not imgs:
        st.warning("No EDA images found")

    for img in imgs:
        st.image(str(img), caption=img.stem, width=700)

# ============ TRAIN ============
elif page == "⚙️ Train":
    st.title("⚙️ Train Models")

    ds = st.selectbox("Dataset", list_datasets() or ["medical_insurance.csv"])

    if st.button("Train"):
        with st.spinner("Training..."):
            from src.train import train_all
            comp = train_all(ds)

        st.success("Training done")
        st.dataframe(comp)

        fig = px.bar(comp, x="model", y="r2", color="r2")
        st.plotly_chart(fig)

# ============ REGISTRY ============
elif page == "🗂 Registry":
    st.title("🗂 Model Registry")

    cards = registry.list_models()

    if not cards:
        st.warning("No models yet")
        st.stop()

    rows = [{
        "Name": c.name,
        "Version": c.version,
        "R2": c.metrics.get("r2", 0),
        "Active": active == (c.name, c.version)
    } for c in cards]

    st.dataframe(pd.DataFrame(rows))

# ============ REPORTS ============
elif page == "📑 Reports":
    st.title("📑 Reports")

    if st.button("Generate Report"):
        from src.report import build_report
        from src.eda import generate_all

        generate_all()
        path = build_report()

        st.success(f"Created: {path.name}")

    pdfs = sorted(REPORTS_DIR.glob("*.pdf"))

    for p in pdfs:
        with open(p, "rb") as f:
            st.download_button(p.name, f.read(), file_name=p.name)