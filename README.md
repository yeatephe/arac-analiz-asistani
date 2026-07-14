# 🚗 AI Vehicle Valuation Assistant

A multimodal AI web app that analyzes a car photo, identifies the vehicle and its condition, and estimates its market price using a machine learning model trained on real Turkish used-car data (~53,500 listings, April 2026).

Upload a photo → the AI recognizes the make, model, specs and visible damage → the details auto-fill a smart price form → a CatBoost model predicts the value as a range, factoring in damage.

## 🚀 Live Demo

👉 [Try the app here](https://arac-ekspertiz-tahmin.streamlit.app)

## 📸 Screenshot

![App screenshot](screenshot.png)

## 🛠️ Tech Stack

- **Python** – core language
- **Google Gemini (Vision)** – multimodal image understanding
- **CatBoost** – gradient-boosting price model (with log-transform)
- **pandas / NumPy** – data processing
- **Streamlit** – web interface, deployment and secrets management

## ✨ Key Features

- **Photo analysis:** the AI identifies make, series, model, color, body type, drivetrain, engine size and power from a single image.
- **Honest AI:** the model states which details it is unsure about (e.g. trim level) instead of guessing silently, and notes its condition read is a preliminary impression — not a real inspection.
- **Smart, data-driven form:** dropdowns show only options that actually exist for the selected car in the data (no impossible combinations). Fields with a single valid option are auto-filled instead of asking the user.
- **Damage-aware pricing:** optional inspection fields (damage record, replaced/painted parts) shown only when relevant; the model learns their price impact from data.
- **Uncertainty shown, not hidden:** predictions are given as a range with the number of comparable listings, plus a warning when data is sparse and a table of real comparable cars.
- **Input validation:** absurd or negative values are blocked at the input level.
- **Robust API handling:** automatic retry and multi-model fallback for rate-limit (429) and high-demand (503) errors.

## 📊 Model & Evaluation

The price model was evaluated on a held-out 20% test set and iteratively improved, with each change measured rather than assumed:

| Model | MAE | MAPE | R² |
|-------|-----|------|-----|
| RandomForest (baseline) | ~67,400 TL | 10.6% | 0.953 |
| **CatBoost + log-transform** | **~60,200 TL** | **9.3%** | **0.962** |

Switching to CatBoost with a log-transform on price reduced the mean absolute error by ~10.7%. Group-based outlier cleaning was tested but removed only ~0.2% of rows and did not improve results, so it was intentionally left out — a reminder that not every technique helps, and measurement is what tells you.

Feature-importance analysis showed model year (~53%), engine power and transmission are the strongest price drivers.

## 💻 Run Locally

```bash
pip install -r requirements.txt
# Add your Gemini API key to .streamlit/secrets.toml as GEMINI_API_KEY
streamlit run app.py
```

## 📁 Data Source

Turkey Used Car Prices (Kaggle) – used-car listing data from Türkiye.

## ⚠️ Note

The AI's condition assessment is a visual preliminary impression based only on the photo and does not replace a professional vehicle inspection.

## 👤 Author

Yiğit Efe USTA – Computer Engineering Student
