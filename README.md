# 🔧 Industrial Asset Risk & Maintenance ROI Analyser

**Live App:** https://maintenance-roi-xbcacv7b83tasfgboru9t5.streamlit.app/

A machine learning tool that predicts industrial equipment failure probability from sensor readings and calculates the financial ROI of preventive versus reactive maintenance — helping plant managers make data-driven maintenance decisions before breakdowns occur.

## About Me

Mechatronics Engineering Graduate | MSc Finance — Henley Business School  
This project combines my engineering background in industrial systems with quantitative finance techniques learned during my postgraduate studies.

- 💼 LinkedIn: [linkedin.com/in/sauravsen34](https://www.linkedin.com/in/sauravsen34)
- 📧 Email: saurav0sen34@gmail.com

---

## The Problem

Every plant manager faces the same decision daily: **maintain now or wait?**

Preventive maintenance costs money upfront. Reactive maintenance after a breakdown costs far more — emergency repairs, unplanned downtime, lost production revenue. Without data, this decision is guesswork.

This tool makes it quantitative.

---

## How It Works — Three Layers

### Layer 1 — Sensor Data
The model ingests five equipment health indicators:
- **Temperature (°C)** — bearings and motors overheat before failing
- **Vibration (mm/s)** — mechanical wear shows up as abnormal vibration
- **Pressure (PSI)** — hydraulic pressure drops as seals and components degrade
- **Runtime Hours** — cumulative operating time accelerates wear
- **RPM** — rotational speed deviations indicate mechanical imbalance

In production, these readings would be streamed directly from IoT sensors, triggering automatic alerts to maintenance personnel when thresholds are breached.

### Layer 2 — ML Failure Prediction
A Random Forest classifier trained on 1,000 equipment records predicts the probability of failure from current sensor readings. The model identifies which sensors deviate most significantly from healthy operating ranges.

**Feature Importance Results:**
- Vibration: 0.45 (dominant predictor — consistent with real-world engineering)
- Pressure: 0.20
- Runtime Hours: 0.17
- Temperature: 0.17
- RPM: 0.05 (least predictive)

Vibration is the strongest predictor because mechanical wear manifests as vibration long before temperature or pressure readings deteriorate significantly.

### Layer 3 — Financial ROI Analysis
The ML failure probability feeds directly into an NPV model comparing two strategies:

```
Expected Reactive Cost = Failure Probability × (Repair Cost + Downtime Cost)
ROI of Prevention = (Expected Reactive Cost - Preventive Cost) / Preventive Cost × 100%
```

**Example output at 35% failure probability:**
- Preventive maintenance cost: £5,000
- Expected failure cost: £34,300
- ROI of prevention: 586%
- Recommendation: **MAINTAIN NOW**

At 35% failure probability, spending £5,000 now avoids an expected £34,300 in repair and downtime costs. The decision is mathematical, not intuitive.

---

## Honest Limitations

**100% training accuracy** — the model achieves 100% accuracy on synthetic data because the healthy and failing distributions were generated with clear separation. Real industrial sensor data contains noise, edge cases, and ambiguous readings that would reduce accuracy to a more realistic 85-95%. The architecture and methodology are valid — the synthetic data is simply too clean.

**Real deployment** would require: actual sensor data from IoT devices, rolling training windows as equipment ages, and calibrated failure cost inputs specific to each asset type.

---

## Research References

This project draws on established research in predictive maintenance and industrial asset management:

- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. NASA Ames Research Center. — The foundational paper behind the CMAPSS turbofan engine dataset, the industry standard benchmark for predictive maintenance research.

- McKinsey & Company (2017). *Maintenance Revolution: Predictive Maintenance and the Future of Infrastructure Management*. — McKinsey research demonstrating that predictive maintenance reduces unexpected equipment failures by up to 50% and lowers maintenance costs by 10-25% versus reactive strategies.

---

## Future Upgrades

- Replace synthetic data with NASA CMAPSS turbofan engine dataset (publicly available real sensor data)
- Integrate Delphyne — a pre-trained transformer model for financial time-series — for more sophisticated failure pattern detection
- Add automated email/SMS alerts to maintenance personnel when MAINTAIN NOW is triggered
- Deploy on Azure IoT Hub for real-time sensor stream integration

---

## How To Run Locally

```bash
git clone https://github.com/sauravsen3/maintenance-roi.git
cd maintenance-roi
pip install -r requirements.txt
streamlit run maintenance.py
```

Or use the live app directly: https://maintenance-roi-xbcacv7b83tasfgboru9t5.streamlit.app/

---

## Project Structure

```
maintenance-roi/
│
├── maintenance.py      # Sensor data generation, ML model, ROI calculator, Streamlit UI
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Tech Stack

- **scikit-learn** — Random Forest classifier, StandardScaler, train/test split
- **Streamlit** — interactive web dashboard with live sliders
- **Plotly** — interactive feature importance chart
- **pandas / numpy** — data generation and manipulation
