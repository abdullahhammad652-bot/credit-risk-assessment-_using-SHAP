"""
Credit Risk Prediction API
--------------------------
FastAPI + Scikit-learn + Random Forest

Features:
- Automatically trains the model if .pkl files do not exist
- Uses credit_risk_dataset.csv if available
- Otherwise creates a synthetic dataset
- Saves the trained model and threshold
- Loads model when FastAPI starts
- Provides:
    GET  /
    GET  /health
    POST /predict
    GET  /docs
- Serves frontend files from ./static
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = BASE_DIR / "credit_risk_model.pkl"
THRESHOLD_PATH = BASE_DIR / "best_threshold.pkl"
DATA_PATH = BASE_DIR / "credit_risk_dataset.csv"


# ============================================================
# 2. GLOBAL MODEL STORAGE
# ============================================================

ml_model = {
    "model": None,
    "threshold": None
}


# ============================================================
# 3. COLUMN DEFINITIONS
# ============================================================

CATEGORICAL_COLUMNS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

NUMERICAL_COLUMNS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

TARGET_COLUMN = "loan_status"


# ============================================================
# 4. CREATE SYNTHETIC DATASET
# ============================================================

def create_synthetic_dataset():

    print("No CSV dataset found.")
    print("Creating synthetic credit-risk dataset...")

    rng = np.random.default_rng(42)

    n = 2000

    df = pd.DataFrame({

        "person_age":
            rng.integers(20, 65, n),

        "person_income":
            rng.integers(15000, 200000, n),

        "person_home_ownership":
            rng.choice(
                ["RENT", "OWN", "MORTGAGE", "OTHER"],
                n
            ),

        "person_emp_length":
            rng.integers(0, 40, n),

        "loan_intent":
            rng.choice(
                [
                    "PERSONAL",
                    "EDUCATION",
                    "MEDICAL",
                    "VENTURE",
                    "HOMEIMPROVEMENT",
                    "DEBTCONSOLIDATION"
                ],
                n
            ),

        "loan_grade":
            rng.choice(
                list("ABCDEFG"),
                n
            ),

        "loan_amnt":
            rng.integers(500, 35000, n),

        "loan_int_rate":
            rng.uniform(5.0, 22.0, n),

        "loan_percent_income":
            rng.uniform(0.0, 0.6, n),

        "cb_person_default_on_file":
            rng.choice(
                ["Y", "N"],
                n,
                p=[0.15, 0.85]
            ),

        "cb_person_cred_hist_length":
            rng.integers(0, 30, n)
    })

    # --------------------------------------------------------
    # Create a simple risk rule
    # --------------------------------------------------------

    risk_score = (

        (df["loan_percent_income"] > 0.35).astype(int)

        + (df["loan_int_rate"] > 15).astype(int)

        + (df["person_income"] < 40000).astype(int)

        + (df["cb_person_default_on_file"] == "Y").astype(int)
    )

    df[TARGET_COLUMN] = (
        risk_score >= 2
    ).astype(int)

    return df


# ============================================================
# 5. LOAD DATASET
# ============================================================

def load_training_data():

  if DATA_PATH.exists():

    print("----------------------------------------")
    print("CSV dataset found.")
    print(f"Loading: {DATA_PATH}")
    print("----------------------------------------")

    try:
        df = pd.read_csv(DATA_PATH)

        if df.empty or len(df.columns) == 0:
            raise ValueError("CSV file is empty or has no columns.")

    except (pd.errors.EmptyDataError, ValueError) as error:

        print(f"CSV cannot be used: {error}")
        print("Using synthetic dataset instead.")

        df = create_synthetic_dataset()

    else:
        # CSV loaded successfully; keep the loaded dataset
        pass

  else:
    print("CSV dataset not found.")
    print("Using synthetic dataset instead.")
    df = create_synthetic_dataset()

  # Remove completely empty rows
  df = df.dropna().reset_index(drop=True)

  required_columns = (
          CATEGORICAL_COLUMNS
          + NUMERICAL_COLUMNS
          + [TARGET_COLUMN]
      )

  missing_columns = [
          column
          for column in required_columns
          if column not in df.columns
      ]

  if missing_columns:

          raise ValueError(
              "Dataset is missing these required columns: "
              + ", ".join(missing_columns)
          )

  return df


# ============================================================
# 6. TRAIN MODEL
# ============================================================

def build_and_save_model():

    print("")
    print("========================================")
    print("TRAINING CREDIT RISK MODEL")
    print("========================================")

    # Import sklearn here
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_training_data()

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Features and target
    # --------------------------------------------------------

    X = df.drop(columns=[TARGET_COLUMN])

    y = df[TARGET_COLUMN].astype(int)

    print("Target values:")
    print(y.value_counts())

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "numerical",
                StandardScaler(),
                NUMERICAL_COLUMNS
            ),

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_COLUMNS
            )
        ]
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        n_jobs=-1,

        class_weight="balanced"
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    model = Pipeline(

        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                random_forest
            )
        ]
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # --------------------------------------------------------
    # Find best threshold
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    best_threshold = 0.5
    best_f1 = -1

    for threshold in np.linspace(
        0.05,
        0.95,
        91
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = float(
                threshold
            )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("")
    print("----------------------------------------")
    print("MODEL RESULTS")
    print("----------------------------------------")

    print(
        f"Best threshold: {best_threshold:.3f}"
    )

    print(
        f"Best F1 score: {best_f1:.3f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        best_threshold,
        THRESHOLD_PATH
    )

    print("")
    print("Files saved successfully:")

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Threshold: {THRESHOLD_PATH}"
    )

    print("========================================")
    print("TRAINING FINISHED")
    print("========================================")

    return model, best_threshold


# ============================================================
# 7. LOAD MODEL
# ============================================================

def load_model():

    # --------------------------------------------------------
    # Train if files don't exist
    # --------------------------------------------------------

    if (
        not MODEL_PATH.exists()
        or not THRESHOLD_PATH.exists()
    ):

        print("")
        print(
            "Model files not found."
        )

        print(
            "Training a new model..."
        )

        build_and_save_model()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = joblib.load(
            MODEL_PATH
        )

        print(
            "Model loaded successfully."
        )

    except Exception as error:

        raise RuntimeError(
            f"Could not load model: {error}"
        )

    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    try:

        threshold = float(
            joblib.load(
                THRESHOLD_PATH
            )
        )

        print(
            f"Threshold loaded: {threshold:.3f}"
        )

    except Exception as error:

        raise RuntimeError(
            f"Could not load threshold: {error}"
        )

    ml_model["model"] = model
    ml_model["threshold"] = threshold


# ============================================================
# 8. FASTAPI LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("")
    print("========================================")
    print("STARTING CREDIT RISK API")
    print("========================================")

    load_model()

    print("")
    print("API is ready.")
    print("========================================")

    yield

    print("")
    print("Shutting down API...")

    ml_model["model"] = None
    ml_model["threshold"] = None


# ============================================================
# 9. CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Credit Risk Prediction API",

    description=(
        "Machine Learning API for predicting "
        "credit default risk."
    ),

    version="1.0.0",

    lifespan=lifespan
)


# ============================================================
# 10. CORS
# ============================================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=False,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# 11. INPUT MODEL
# ============================================================

class LoanApplication(BaseModel):

    person_age: int

    person_income: float

    person_home_ownership: str

    person_emp_length: int

    loan_intent: str

    loan_grade: str

    loan_amnt: float

    loan_int_rate: float

    loan_percent_income: float

    cb_person_default_on_file: str

    cb_person_cred_hist_length: int


# ============================================================
# 12. STATIC FILES
# ============================================================

app.mount(

    "/static",

    StaticFiles(
        directory=str(STATIC_DIR)
    ),

    name="static"
)


# ============================================================
# 13. HOME PAGE
# ============================================================

@app.get("/")
def home():

    index_file = STATIC_DIR / "index.html"

    if index_file.exists():

        return FileResponse(
            index_file
        )

    return {
        "message": "Credit Risk API is running.",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
        "warning": "static/index.html not found."
    }


# ============================================================
# 14. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    model_loaded = (
        ml_model["model"] is not None
    )

    threshold_loaded = (
        ml_model["threshold"] is not None
    )

    return {

        "status": "ok",

        "model_loaded":
            model_loaded,

        "threshold_loaded":
            threshold_loaded
    }


# ============================================================
# 15. PREDICT
# ============================================================

@app.post("/predict")
def predict(data: LoanApplication):

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if ml_model["model"] is None:

        raise HTTPException(

            status_code=503,

            detail="Machine learning model is not loaded."
        )

    try:

        # ----------------------------------------------------
        # Convert request to DataFrame
        # ----------------------------------------------------

        input_data = data.model_dump()

        input_df = pd.DataFrame(
            [input_data]
        )

        # ----------------------------------------------------
        # Get model
        # ----------------------------------------------------

        model = ml_model["model"]

        # ----------------------------------------------------
        # Predict probability
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            input_df
        )[0]

        classes = list(
            model.classes_
        )

        if 1 not in classes:

            raise ValueError(
                "Model does not contain class 1."
            )

        positive_index = classes.index(1)

        probability = float(
            probabilities[positive_index]
        )

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        threshold = float(
            ml_model["threshold"]
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = int(
            probability >= threshold
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        if prediction == 1:

            result = "High Risk"

        else:

            result = "Low Risk"

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "default_probability":
                round(probability, 4),

            "default_prediction":
                prediction,

            "threshold":
                round(threshold, 4),

            "result":
                result
        }

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=f"Prediction failed: {error}"
        )


# ============================================================
# 16. RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="127.0.0.1",

        port=8000,

        reload=True
    )