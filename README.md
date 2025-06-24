# 🫀 Heart Disease Prediction System

A Python-based heart disease prediction system using Random Forest Classifier that also locates nearby hospitals for high-risk cases.

## ✨ Features

- 🔬 Predict heart disease risk using clinical parameters
- 📊 Model evaluation with accuracy, confusion matrix, ROC curve
- 📍 Location-based hospital finder using OpenStreetMap APIs
- 🏥 Emergency contacts for major Indian cities
- 🆘 First-aid precautions and remedies for high-risk cases

## Setup

1. Clone the repository
```
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
```

2. Install dependencies
```
pip install pandas numpy matplotlib scikit-learn geopy requests
```

3. Place your heart.csv dataset in the project folder

4. Run the program
```
python heart_disease_prediction.py
```

## How It Works

- Trained on heart.csv using RandomForestClassifier with 500 estimators
- User enters medical data via terminal prompts
- If high-risk is predicted, system shows emergency guidance and locates nearby hospitals
- Falls back to major city emergency contacts if location lookup fails

## Supported Cities

Mumbai, Delhi, Bangalore, Chennai, Kolkata, Pune

## ⚠️ Disclaimer

This tool is for educational purposes only and is not a substitute for professional medical advice. Always consult a healthcare provider for medical concerns.
