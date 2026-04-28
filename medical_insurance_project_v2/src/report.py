"""
report.py
---------
Automated PDF report generator (fpdf2).

Combines:
  * Dataset summary (rows, columns, descriptive stats)
  * Active model card (algorithm, metrics, params)
  * All EDA images
  * Model comparison table
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
from fpdf import FPDF
from .paths import REPORTS_DIR, EDA_DIR, MODELS_DIR
from .data_loader import load_dataset
from . import registry


class InsurancePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(13, 148, 136)
        self.cell(0, 10, "Medical Insurance - ML Report", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(13, 148, 136); self.line(10, 22, 200, 22)
        self.ln(4)

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}  |  Generated {datetime.now():%Y-%m-%d %H:%M}",
                  align="C")

    def h2(self, text):
        self.set_font("Helvetica", "B", 12); self.set_text_color(15, 23, 42)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT"); self.ln(1)

    def body(self, text):
        self.set_font("Helvetica", "", 10); self.set_text_color(30, 41, 59)
        self.multi_cell(0, 5, text); self.ln(2)


def _table(pdf: FPDF, df: pd.DataFrame, max_rows: int = 10):
    pdf.set_font("Helvetica", "B", 9)
    col_w = (pdf.w - 20) / len(df.columns)
    for c in df.columns:
        pdf.cell(col_w, 6, str(c)[:14], border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for _, row in df.head(max_rows).iterrows():
        for v in row:
            txt = f"{v:.3f}" if isinstance(v, float) else str(v)
            pdf.cell(col_w, 5, txt[:14], border=1)
        pdf.ln()
    pdf.ln(3)


def build_report(dataset: str | None = None, output_name: str | None = None) -> Path:
    df = load_dataset(dataset)
    pdf = InsurancePDF()
    pdf.add_page()

    # 1. Overview
    pdf.h2("1. Dataset Overview")
    pdf.body(
        f"Source file : {dataset or 'medical_insurance.csv'}\n"
        f"Rows        : {len(df):,}\n"
        f"Columns     : {', '.join(df.columns)}\n"
        f"Avg charges : ${df['charges'].mean():,.2f}\n"
        f"Smokers     : {(df['smoker']=='yes').sum()} ({(df['smoker']=='yes').mean()*100:.1f}%)\n"
    )

    # 2. Descriptive stats
    pdf.h2("2. Descriptive Statistics")
    desc = df.describe().round(2).reset_index()
    _table(pdf, desc)

    # 3. Active model
    pdf.h2("3. Active Production Model")
    info = registry.get_active()
    if info:
        name, ver = info
        cards = {(c.name, c.version): c for c in registry.list_models()}
        c = cards[(name, ver)]
        pdf.body(
            f"Name      : {c.name} (v{c.version})\n"
            f"Algorithm : {c.algorithm}\n"
            f"Trained   : {c.created_at}\n"
            f"Dataset   : {c.dataset}  ({c.rows_trained_on} rows)\n\n"
            f"Metrics:\n"
            f"  R2        = {c.metrics.get('r2', 0):.4f}\n"
            f"  MAE       = {c.metrics.get('mae', 0):.2f}\n"
            f"  RMSE      = {c.metrics.get('rmse', 0):.2f}\n"
            f"  CV R2 avg = {c.metrics.get('cv_r2_mean', 0):.4f}\n"
        )
    else:
        pdf.body("No active model registered yet.")

    # 4. Comparison table
    comp_path = MODELS_DIR / "model_comparison.csv"
    if comp_path.exists():
        pdf.h2("4. Model Comparison (sorted by R2)")
        _table(pdf, pd.read_csv(comp_path).round(4))

    # 5. EDA gallery
    pdf.h2("5. Exploratory Data Analysis")
    images = sorted(EDA_DIR.glob("*.png"))
    for img in images:
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, img.stem.replace("_", " ").title(),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.image(str(img), w=180)
        pdf.ln(4)

    name = output_name or f"insurance_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    out = REPORTS_DIR / name
    pdf.output(str(out))
    return out


if __name__ == "__main__":
    p = build_report()
    print("Report written:", p)
