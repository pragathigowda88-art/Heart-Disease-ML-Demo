import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. FILE LOCATIONS
# ============================================================

DATA_PATH = "data/heart.csv"
MODEL_DIR = "model"
MODEL_PATH = "model/heart_model.joblib"


# ============================================================
# 2. EXPECTED INPUT FEATURES
# ============================================================

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]


# ============================================================
# 3. CHECK CSV FILE
# ============================================================

if not os.path.exists(DATA_PATH):

    print("\nERROR: heart.csv was not found!")
    print()
    print("Please put your CSV file here:")
    print("data/heart.csv")
    print()

    exit()


# ============================================================
# 4. READ CSV
# ============================================================

print("\nLoading heart disease dataset...")

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")

print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumns in your CSV:")
print(df.columns.tolist())


# ============================================================
# 5. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

print("\nCleaned columns:")
print(df.columns.tolist())


# ============================================================
# 6. FIND TARGET COLUMN
# ============================================================

target_columns = [
    "target",
    "output",
    "condition",
    "heartdisease",
    "heart_disease",
    "num"
]

target = None

for column in target_columns:

    if column in df.columns:
        target = column
        break


if target is None:

    print("\nERROR: Target column not found!")

    print("\nYour CSV contains:")
    print(df.columns.tolist())

    print("\nExpected target column can be:")
    print("target")
    print("output")
    print("condition")
    print("heartdisease")
    print("heart_disease")
    print("num")

    exit()


print("\nTarget column:", target)


# ============================================================
# 7. CHECK FEATURES
# ============================================================

missing_features = []

for feature in FEATURES:

    if feature not in df.columns:
        missing_features.append(feature)


if len(missing_features) > 0:

    print("\nERROR: Some required columns are missing!")

    print("\nMissing columns:")
    print(missing_features)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    exit()


# ============================================================
# 8. SELECT DATA
# ============================================================

X = df[FEATURES].copy()

y = df[target].copy()


# ============================================================
# 9. CONVERT DATA TO NUMERIC
# ============================================================

for column in FEATURES:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


y = pd.to_numeric(
    y,
    errors="coerce"
)


# ============================================================
# 10. REMOVE ROWS WITHOUT TARGET
# ============================================================

valid_rows = y.notna()

X = X[valid_rows]

y = y[valid_rows]


# ============================================================
# 11. CONVERT TARGET TO BINARY
# ============================================================

# 0 = No heart disease
# 1 = Heart disease

y = (y > 0).astype(int)


print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 12. SPLIT DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 13. CREATE MACHINE LEARNING PIPELINE
# ============================================================

model = Pipeline([

    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000
        )
    )

])


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ============================================================
# 15. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 16. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n======================================")
print("       HEART DISEASE MODEL")
print("======================================")

print("\nAccuracy:")
print(
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 18. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 19. CREATE MODEL FOLDER
# ============================================================

if not os.path.exists(MODEL_DIR):

    os.makedirs(MODEL_DIR)


# ============================================================
# 20. SAVE TRAINED MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)


print("\n======================================")
print("MODEL SAVED SUCCESSFULLY!")
print("======================================")

print("\nModel location:")
print(MODEL_PATH)

print("\nYou can now start Flask using:")

print("python app.py")

print("\nOpen in browser:")

print("http://127.0.0.1:5000")