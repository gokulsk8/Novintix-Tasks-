import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# TITLE
# -----------------------------
st.title("Healthcare Data Analysis, ML & AI Doctor Recommendation")

# -----------------------------
# LOAD DATA
# -----------------------------
st.header("📌 Load Dataset")

uploaded = st.file_uploader("Upload healthcare.csv", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())
else:
    st.warning("Upload the dataset to begin...")
    st.stop()

# -----------------------------
# TASK 1 — EDA
# -----------------------------
st.header("📊 Task 1 — Exploratory Data Analysis")

st.subheader("Distribution Plots")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["Age"], ax=ax[0], kde=True)
ax[0].set_title("Age Distribution")

sns.histplot(df["Billing Amount"], ax=ax[1], kde=True)
Ax1= ax[1].set_title("Billing Amount Distribution")

sns.histplot(df["Room Number"], ax=ax[2], kde=True)
ax[2].set_title("Room Number Distribution")

st.pyplot(fig)

# Frequency plots for categorical variables
st.subheader("Frequency of Categorical Columns")

cat_cols = ["Medical Condition", "Admission Type", "Medication"]

for col in cat_cols:
    st.write(f"### {col}")
    fig = plt.figure(figsize=(8,4))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"{col} Frequency")
    st.pyplot(fig)

# -----------------------------
# TASK 2 — SUPERVISED LEARNING
# Predict Test Results
# -----------------------------
st.header("🤖 Task 2 — Test Result Prediction (Supervised ML)")

# Encode categorical variables
df_ml = df.copy()
label_map = {}
for col in df_ml.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    label_map[col] = le

# Prepare data
X = df_ml.drop("Test Results", axis=1)
y = df_ml["Test Results"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Show evaluation
st.subheader("Model Evaluation")
st.write("**RMSE:**", np.sqrt(mean_squared_error(y_test, y_pred)))
st.write("**R² Score:**", r2_score(y_test, y_pred))

# Pred vs Actual
st.subheader("Predicted vs Actual Test Results")
pred_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
st.dataframe(pred_df.head(20))

# -----------------------------
# TASK 3 — ANOMALY DETECTION
# -----------------------------
st.header("⚠️ Task 3 — Billing Amount Anomaly Detection")

iso = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = iso.fit_predict(df[["Billing Amount"]])

df["Anomaly Flag"] = df["Anomaly"].apply(lambda x: "Anomaly" if x == -1 else "Normal")

st.write("### Anomalies Detected")
st.dataframe(df[df["Anomaly Flag"] == "Anomaly"])

st.info("Anomalies represent unusually high/low billing values — possible rare cases or data entry errors.")

# -----------------------------
# TASK 4 — AI DOCTOR RECOMMENDATION
# -----------------------------
st.header("🩺 Task 4 — AI Doctor Recommendation Generator")

# Choose a test case
sample = df.sample(1).iloc[0]

age = sample["Age"]
cond = sample["Medical Condition"]
med = sample["Medication"]
test_res = pred_df["Predicted"].iloc[0] if len(pred_df) > 0 else "N/A"

st.write(f"### Sample Input")
st.write(f"**Age:** {age}")
st.write(f"**Medical Condition:** {cond}")
st.write(f"**Medication:** {med}")
st.write(f"**Predicted Test Result:** {test_res}")

st.write("### Doctor Recommendation (Sample Output)")
recommendation = f"""
Based on the patient’s predicted test result and medical condition:

- **Age:** {age}
- **Condition:** {cond}
- **Medication:** {med}

### 🩺 Doctor's Recommendation
The predicted test result indicates a potential abnormality requiring medical attention.  
I recommend continuing the prescribed medication and scheduling a follow-up examination within a week.

### 📘 Health Advice
- Maintain a balanced diet and stay hydrated.  
- Avoid stress and get adequate sleep.  
- Report any unusual symptoms immediately.

**This recommendation is AI-generated and should not replace real medical consultation.**
"""

st.write(recommendation)

