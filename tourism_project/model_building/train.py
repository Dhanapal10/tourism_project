
# =========================
# Imports
# =========================
import pandas as pd
import os
import joblib

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Model & tuning
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# MLflow
import mlflow
import mlflow.sklearn

# Hugging Face
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# =========================
# MLflow Configuration
# =========================
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-prediction")

# =========================
# Hugging Face API
# =========================
api = HfApi(token=os.getenv("HF_TOKEN"))

# =========================
# Load Data
# =========================
DATASET_REPO = "dhanapalpalanisamy/tourism-dataset"

Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").values.ravel()
ytest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").values.ravel()

print("Train and test data loaded successfully")


# =========================
# Feature Groups (Tourism CSV)
# =========================
numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender',
    'ProductPitched', 'MaritalStatus', 'Designation'
]


# =========================
# Preprocessing Pipeline
# =========================
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)


# =========================
# Model Definition
# =========================
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False
)

pipeline = make_pipeline(preprocessor, xgb_model)


# =========================
# Hyperparameter Grid
# =========================
param_grid = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0]
}


# =========================
# Training with MLflow
# =========================
with mlflow.start_run():

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)

    # Log all hyperparameter runs
    results = grid_search.cv_results_
    for i, params in enumerate(results["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_f1", results["mean_test_score"][i])

    # Best model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Predictions
    y_train_pred = best_model.predict(Xtrain)
    y_test_pred  = best_model.predict(Xtest)
    y_test_prob  = best_model.predict_proba(Xtest)[:, 1]

    # Metrics
    metrics = {
        "train_accuracy": accuracy_score(ytrain, y_train_pred),
        "test_accuracy": accuracy_score(ytest, y_test_pred),
        "test_precision": precision_score(ytest, y_test_pred),
        "test_recall": recall_score(ytest, y_test_pred),
        "test_f1": f1_score(ytest, y_test_pred),
        "test_roc_auc": roc_auc_score(ytest, y_test_prob)
    }

    mlflow.log_metrics(metrics)

    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # =========================
    # Save Model
    # =========================
    model_path = "best_wellness_tourism_model.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved as {model_path}")

    # =========================
    # Upload to Hugging Face
    # =========================
    MODEL_REPO = "dhanapalpalanisamy/tourism-model" 

    try:
        api.repo_info(repo_id=MODEL_REPO, repo_type="model")
        print("Model repository exists.")
    except RepositoryNotFoundError:
        create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
        print("Model repository created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.pkl",
        repo_id=MODEL_REPO,
        repo_type="model"
    )

    print("Model uploaded to Hugging Face Model Hub successfully")
