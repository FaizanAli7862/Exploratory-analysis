import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page Configuration

st.set_page_config(
    page_title="Titanic Data Analysis",
    layout="wide"
)

# Load Dataset

df = pd.read_csv("Titanic-Dataset.csv")

# Title

st.title("Titanic Data Analysis Dashboard")
st.caption("Exploratory analysis of Titanic passenger data")

st.divider()

# Dataset Overview

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Passengers", len(df))
col2.metric("Survived", int(df["Survived"].sum()))
col3.metric("Did Not Survive", int((df["Survived"] == 0).sum()))
col4.metric("Columns", len(df.columns))

st.divider()

# Survival Analysis

st.subheader("🧍 Survival Analysis")

survival_counts = df["Survived"].value_counts()

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    ["Did Not Survive", "Survived"],
    [
        survival_counts.get(0, 0),
        survival_counts.get(1, 0)
    ]
)

ax.set_xlabel("Survival Status")
ax.set_ylabel("Number of Passengers")
ax.set_title("Titanic Survival Distribution")

st.pyplot(fig)

# Gender Analysis

st.subheader("Survival by Gender")

gender_survival = pd.crosstab(
    df["Sex"],
    df["Survived"]
)

fig2, ax2 = plt.subplots(figsize=(8, 5))

gender_survival.plot(
    kind="bar",
    ax=ax2
)

ax2.set_xlabel("Gender")
ax2.set_ylabel("Number of Passengers")
ax2.set_title("Survival by Gender")
ax2.legend(["Did Not Survive", "Survived"])

st.pyplot(fig2)

# Passenger Class Analysis

st.subheader("🎫 Survival by Passenger Class")

class_survival = pd.crosstab(
    df["Pclass"],
    df["Survived"]
)

fig3, ax3 = plt.subplots(figsize=(8, 5))

class_survival.plot(
    kind="bar",
    ax=ax3
)

ax3.set_xlabel("Passenger Class")
ax3.set_ylabel("Number of Passengers")
ax3.set_title("Survival by Passenger Class")
ax3.legend(["Did Not Survive", "Survived"])

st.pyplot(fig3)

# Age Distribution

st.subheader("🎂 Age Distribution")

fig4, ax4 = plt.subplots(figsize=(10, 5))

ax4.hist(
    df["Age"].dropna(),
    bins=20
)

ax4.set_xlabel("Age")
ax4.set_ylabel("Number of Passengers")
ax4.set_title("Passenger Age Distribution")

st.pyplot(fig4)

# Fare Distribution

st.subheader("💰 Fare Distribution")

fig5, ax5 = plt.subplots(figsize=(10, 5))

ax5.hist(
    df["Fare"],
    bins=20
)

ax5.set_xlabel("Fare")
ax5.set_ylabel("Number of Passengers")
ax5.set_title("Passenger Fare Distribution")

st.pyplot(fig5)

# -----------------------------
# Missing Values
# -----------------------------
st.subheader("🔍 Missing Values")

missing_values = df.isna().sum()

missing_df = pd.DataFrame({
    "Column": missing_values.index,
    "Missing Values": missing_values.values
})

st.dataframe(
    missing_df,
    use_container_width=True
)

# Statistical Summary

st.subheader("📈 Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# Raw Dataset

st.subheader("📋 Dataset")

st.dataframe(
    df,
    use_container_width=True
)
