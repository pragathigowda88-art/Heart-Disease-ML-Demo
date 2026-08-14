# CardioPredict — Flask Heart Disease Prediction

A beginner-friendly Flask + Tailwind CSS + Material Tailwind HTML project.

## 1. Project structure

```text
heart_disease_flask_app/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── heart.csv              <-- PUT YOUR DOWNLOADED CSV HERE
│
├── model/
│   ├── heart_model.joblib     <-- generated after training
│   └── metrics.json           <-- generated after training
│
├── static/
└── templates/
    ├── base.html
    ├── index.html
    ├── result.html
    └── about.html
```

## 2. Expected dataset

This version is designed for the common 13-feature heart-disease dataset with columns:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
target
```

Rename your downloaded CSV to:

```text
heart.csv
```

and put it in:

```text
data/heart.csv
```

## 3. Install

Open Command Prompt/PowerShell inside this folder:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## 4. Train the model

Run:

```bash
python train_model.py
```

You should see accuracy, classification report and confusion matrix.

The trained model will be saved as:

```text
model/heart_model.joblib
```

## 5. Start Flask

Run:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## 6. If you get a column error

Run:

```bash
python -c "import pandas as pd; print(pd.read_csv('data/heart.csv').columns.tolist())"
```

Compare the printed columns with the expected list in `train_model.py`.

## 7. Important note

This is an educational machine-learning project. A model trained on a public CSV is not a medical diagnostic system and should not be used for real medical decisions.

## 8. Design

The frontend uses Tailwind CSS Play CDN and Material Tailwind HTML assets for a beginner-friendly setup. For a production application, use a proper Tailwind build process instead of the Play CDN.
