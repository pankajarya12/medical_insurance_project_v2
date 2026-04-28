"""
eda.py
------
Generates rich, medical-themed EDA visuals using Plotly + Matplotlib.
Saves PNGs to app/eda_images/ for the Streamlit gallery + PDF report.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from .paths import EDA_DIR
from .data_loader import load_dataset

sns.set_theme(style="whitegrid", palette="viridis")
PALETTE = {"yes": "#dc2626", "no": "#0d9488"}


def _save(fig, name: str):
    out = EDA_DIR / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col, color in zip(axes, ["age", "bmi", "charges"],
                              ["#0d9488", "#7c3aed", "#dc2626"]):
        sns.histplot(df[col], kde=True, color=color, ax=ax)
        ax.set_title(f"Distribution of {col.title()}")
    fig.suptitle("Patient Population Overview", fontsize=14, weight="bold")
    return _save(fig, "01_distributions.png")


def plot_smoker_impact(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="smoker", y="charges", palette=PALETTE, ax=ax)
    ax.set_title("Smoking dramatically increases insurance charges", weight="bold")
    ax.set_ylabel("Annual charges (USD)")
    return _save(fig, "02_smoker_impact.png")


def plot_bmi_smoker_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker",
                    palette=PALETTE, alpha=0.7, s=60, ax=ax)
    ax.axvline(30, ls="--", color="grey", alpha=0.6, label="Obesity threshold")
    ax.set_title("BMI × Smoking Interaction (the danger zone)", weight="bold")
    ax.legend()
    return _save(fig, "03_bmi_smoker.png")


def plot_age_trend(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.regplot(data=df[df["smoker"] == "no"], x="age", y="charges",
                color=PALETTE["no"], label="non-smoker", ax=ax,
                scatter_kws={"alpha": 0.4})
    sns.regplot(data=df[df["smoker"] == "yes"], x="age", y="charges",
                color=PALETTE["yes"], label="smoker", ax=ax,
                scatter_kws={"alpha": 0.4})
    ax.set_title("Charges grow with Age - faster for smokers", weight="bold")
    ax.legend()
    return _save(fig, "04_age_trend.png")


def plot_region_heatmap(df: pd.DataFrame):
    pivot = df.pivot_table(index="region", columns="smoker",
                           values="charges", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", ax=ax)
    ax.set_title("Avg charges by region × smoking status", weight="bold")
    return _save(fig, "05_region_heatmap.png")


def plot_correlation(df: pd.DataFrame):
    num = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation", weight="bold")
    return _save(fig, "06_correlation.png")


def generate_all(df: pd.DataFrame | None = None) -> list[str]:
    """Run every plot - returns list of generated file names."""
    if df is None:
        df = load_dataset()
    outputs = [
        plot_distributions(df), plot_smoker_impact(df),
        plot_bmi_smoker_scatter(df), plot_age_trend(df),
        plot_region_heatmap(df), plot_correlation(df),
    ]
    return [p.name for p in outputs]


if __name__ == "__main__":
    files = generate_all()
    print(f"Generated {len(files)} EDA plots in {EDA_DIR}")
    for f in files: print(" -", f)
