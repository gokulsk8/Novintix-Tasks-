import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings("ignore")

st.title("Wind Turbine SCADA Data Analysis & Forecasting Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
st.header("📂 Load Dataset")

file = st.file_uploader("Upload the SCADA CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df = df.sort_values("Date/Time")
    st.success("Dataset loaded successfully!")

    st.dataframe(df.head())
else:
    st.stop()

# -----------------------------
# TASK 1 — EDA
# -----------------------------
st.header("📊 Task 1 — Exploratory Data Analysis")

st.subheader("Time-Series Trends")
fig, ax = plt.subplots(4, 1, figsize=(12, 12))
ax[0].plot(df['Date/Time'], df['LV ActivePower (kW)'])
ax[0].set_title("LV Active Power")

ax[1].plot(df['Date/Time'], df['Wind Speed (m/s)'])
ax[1].set_title("Wind Speed")

ax[2].plot(df['Date/Time'], df['Theoretical_Power_Curve (kWh)'])
ax[2].set_title("Theoretical Power Curve")

ax[3].plot(df['Date/Time'], df['Wind Direction (°)'])
ax[3].set_title("Wind Direction")

st.pyplot(fig)

st.subheader("Missing / Abnormal Readings")
st.write(df.describe())
st.write("Missing Values:")
st.write(df.isnull().sum())

st.subheader("Wind Speed vs LV ActivePower Scatter Plot")
fig = plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Wind Speed (m/s)', y='LV ActivePower (kW)')
st.pyplot(fig)

# -----------------------------
# TASK 2 — TIME-SERIES FORECASTING
# -----------------------------
st.header("📈 Task 2 — Time-Series Forecasting for Four Variables")

# Select four variables
cols = [
    "LV ActivePower (kW)",
    "Wind Speed (m/s)",
    "Theoretical_Power_Curve (kWh)",
    "Wind Direction (°)"
]

data = df[cols]

# Scale the data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

def create_windows(dataset, window_size=24):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i+window_size])
        y.append(dataset[i+window_size])
    return np.array(X), np.array(y)

window_size = 24
X, y = create_windows(scaled, window_size)

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# LSTM Model
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(window_size, 4)),
    Dense(32, activation="relu"),
    Dense(4)
])
model.compile(optimizer=Adam(0.001), loss="mse")

with st.spinner("Training LSTM model..."):
    model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled)
y_actual = scaler.inverse_transform(y_test)

st.success("Forecasting Completed!")

# Plot predictions
st.subheader("Predicted vs Actual Values")

for i, col in enumerate(cols):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(y_actual[:, i], label="Actual")
    plt.plot(pred[:, i], label="Predicted")
    plt.title(f"{col} — Prediction")
    plt.legend()
    st.pyplot(fig)

    rmse = np.sqrt(mean_squared_error(y_actual[:, i], pred[:, i]))
    st.write(f"**RMSE for {col}:** {rmse:.3f}")

# -----------------------------
# TASK 3 — ANOMALY DETECTION
# -----------------------------
st.header("⚠️ Task 3 — Underperformance Anomaly Detection")

df["Perf Ratio"] = df["LV ActivePower (kW)"] / df["Theoretical_Power_Curve (kWh)"]
df["Perf Ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
df["Perf Ratio"].fillna(df["Perf Ratio"].mean(), inplace=True)

iso = IsolationForest(contamination=0.03)
df["Anomaly"] = iso.fit_predict(df[["Perf Ratio"]])
df["Anomaly Label"] = df["Anomaly"].apply(lambda x: "Underperformance" if x == -1 else "Normal")

st.write("### Detected Underperformance Points")
st.dataframe(df[df["Anomaly Label"] == "Underperformance"].head(20))

fig = plt.figure(figsize=(10, 4))
plt.scatter(df["Wind Speed (m/s)"], df["LV ActivePower (kW)"],
            c=df["Anomaly"], cmap="coolwarm", s=10)
plt.title("Anomaly Detection — Actual Power vs Wind Speed")
plt.xlabel("Wind Speed")
plt.ylabel("Active Power")
st.pyplot(fig)

# -----------------------------
# TASK 4 — AI TASK
# -----------------------------
st.header("🤖 Task 4 — AI Turbine Performance Score Generator")

latest = df.iloc[-1]

perf_ratio = latest["Perf Ratio"]
score = np.clip(perf_ratio * 100, 0, 100)

if score > 80:
    status = "Good"
    suggestion = "Turbine is performing efficiently. Maintain current settings."
elif score > 50:
    status = "Moderate"
    suggestion = "Slight underperformance detected. Inspect blade pitch & yaw alignment."
else:
    status = "Poor"
    suggestion = "Significant underperformance. Check for mechanical or electrical faults."

st.subheader("Performance Summary")
st.write(f"**Performance Score:** {score:.2f}/100")
st.write(f"**Turbine State:** {status}")
st.write(f"**Recommendation:** {suggestion}")

