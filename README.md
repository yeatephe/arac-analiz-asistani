# 🚗 AI Vehicle Valuation Assistant

A multimodal AI web app that analyzes a car photo, identifies the vehicle and its visible condition, and then estimates its market price using a machine learning model trained on real Turkish used-car data.

Upload a photo → the AI recognizes the make, model, color and visible damage → the details auto-fill a price form → a trained ML model predicts the value, including the effect of damage.

## 🚀 Live Demo

👉 [Try the app here](https://guanrynhgariacpngg9q6d.streamlit.app)

## 📸 Screenshot

![App screenshot](screenshot.png)

## 🛠️ Tech Stack

- **Python** – core language
- **Google Gemini (Vision)** – multimodal image understanding (make/model/condition detection)
- **scikit-learn** – price prediction model (Random Forest + Pipeline & OneHotEncoder)
- **pandas** – data processing and cleaning
- **Streamlit** – web interface, deployment and secrets management

## ✨ Features

- **Photo analysis:** the AI identifies make, model, color, body type and visible condition.
- **Honest condition assessment:** reports only what is visible and notes it is a preliminary impression, not a real inspection.
- **Automatic form filling:** detected make, series and estimated year are pre-filled into the price form.
- **Damage-aware pricing:** optional
