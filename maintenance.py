import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def generate_sensor_data(n_samples=1000, random_state=42):
    """
    Generate synthetic sensor data for industrial equipment.
    Features: temperature, vibration, pressure, runtime_hours, rpm
    Target: failure (1) or healthy (0)
    """
    np.random.seed(random_state)

    # Healthy equipment — normal operating ranges
    n_healthy = int(n_samples * 0.8)
    healthy = pd.DataFrame({
        'temperature':   np.random.normal(75, 10, n_healthy),
        'vibration':     np.random.normal(1.5, 0.3, n_healthy),
        'pressure':      np.random.normal(100, 8, n_healthy),
        'runtime_hours': np.random.uniform(0, 5000, n_healthy),
        'rpm':           np.random.normal(1500, 50, n_healthy),
        'failure':       0
    })

    # Failing equipment — abnormal readings
    n_failing = n_samples - n_healthy
    failing = pd.DataFrame({
        'temperature':   np.random.normal(110, 15, n_failing),
        'vibration':     np.random.normal(4.5, 0.8, n_failing),
        'pressure':      np.random.normal(75, 12, n_failing),
        'runtime_hours': np.random.uniform(4000, 8000, n_failing),
        'rpm':           np.random.normal(1350, 100, n_failing),
        'failure':       1
    })

    data = pd.concat([healthy, failing], ignore_index=True)
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return data


def train_model(data):
    """
    Train Random Forest classifier on sensor data.
    Returns trained model, scaler, and test results.
    """
    features = ['temperature', 'vibration', 'pressure',
                'runtime_hours', 'rpm']

    X = data[features]
    y = data['failure']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return model, scaler, features, accuracy, X_test, y_test, y_pred


def calculate_maintenance_roi(failure_probability,
                               preventive_cost=5000,
                               reactive_cost=50000,
                               downtime_cost_per_hour=2000,
                               downtime_hours=24,
                               discount_rate=0.1):
    """
    Calculate NPV of preventive vs reactive maintenance strategy.
    """
    total_downtime_cost = downtime_cost_per_hour * downtime_hours
    expected_reactive_cost = failure_probability * (reactive_cost + total_downtime_cost)

    npv_preventive = -preventive_cost + (1 - failure_probability) * \
                     (reactive_cost + total_downtime_cost)

    roi = (expected_reactive_cost - preventive_cost) / preventive_cost * 100

    recommend_preventive = preventive_cost < expected_reactive_cost

    return {
        'failure_probability': failure_probability,
        'preventive_cost': preventive_cost,
        'expected_reactive_cost': round(expected_reactive_cost, 2),
        'npv_preventive': round(npv_preventive, 2),
        'roi': round(roi, 2),
        'recommendation': 'MAINTAIN NOW' if recommend_preventive else 'MONITOR'
    }


def main():
    st.set_page_config(page_title="Maintenance ROI Analyser",
                       page_icon="🔧", layout="wide")
    st.title("🔧 Industrial Asset Risk & Maintenance ROI Analyser")
    st.caption("Predicts equipment failure probability and calculates financial ROI of preventive maintenance")

    with st.spinner("Training model on sensor data..."):
        data = generate_sensor_data()
        model, scaler, features, accuracy, X_test, y_test, y_pred = train_model(data)

    st.success(f"Model trained — Accuracy: {accuracy:.1%}")

    st.divider()
    st.subheader("📊 Enter Equipment Sensor Readings")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.slider("Temperature (°C)", 50.0, 150.0, 75.0)
        vibration = st.slider("Vibration (mm/s)", 0.5, 8.0, 1.5)
        pressure = st.slider("Pressure (PSI)", 40.0, 140.0, 100.0)

    with col2:
        runtime_hours = st.slider("Runtime Hours", 0, 8000, 2000)
        rpm = st.slider("RPM", 1000, 2000, 1500)
        preventive_cost = st.number_input("Preventive Maintenance Cost (£)",
                                           value=5000, step=500)

    sensor_input = pd.DataFrame([[temperature, vibration, pressure,
                                   runtime_hours, rpm]], columns=features)
    sensor_scaled = scaler.transform(sensor_input)
    failure_prob = model.predict_proba(sensor_scaled)[0][1]

    roi_result = calculate_maintenance_roi(failure_prob,
                                            preventive_cost=preventive_cost)

    st.divider()
    st.subheader("🎯 Risk & Financial Analysis")

    col3, col4, col5, col6 = st.columns(4)
    col3.metric("Failure Probability", f"{failure_prob:.1%}")
    col4.metric("Preventive Cost", f"£{roi_result['preventive_cost']:,}")
    col5.metric("Expected Failure Cost", f"£{roi_result['expected_reactive_cost']:,}")
    col6.metric("ROI of Prevention", f"{roi_result['roi']:.1f}%")

    if roi_result['recommendation'] == 'MAINTAIN NOW':
        st.error("⚠️ RECOMMENDATION: MAINTAIN NOW — Prevention saves money")
    else:
        st.success("✅ RECOMMENDATION: MONITOR — Risk is within acceptable range")

    st.divider()
    st.subheader("📈 Feature Importance — What Drives Failure Risk?")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='steelblue'
    ))
    fig.update_layout(title="Random Forest Feature Importance",
                      xaxis_title="Importance Score",
                      height=300)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
