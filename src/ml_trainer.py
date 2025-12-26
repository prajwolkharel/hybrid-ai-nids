import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime

# Configuration
PROCESSED_FILE = Path("data/processed/cicids2017_processed.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Sample size (adjust if your machine has more RAM)
SAMPLE_SIZE = 500000  # ~18% of data, keeps class ratios

def load_and_prepare_data():
    print(f"Loading processed data from {PROCESSED_FILE}...")
    df = pd.read_csv(PROCESSED_FILE)
    print(f"Full shape: {df.shape}")
    
    # Stratified sample to preserve attack ratios
    df_sample = df.groupby('Label', group_keys=False).apply(
        lambda x: x.sample(frac=SAMPLE_SIZE / len(df), random_state=42)
    )
    print(f"Sampled shape: {df_sample.shape}")
    print("Sample label distribution:")
    print(df_sample['Label'].value_counts())
    
    # Features (drop non-numeric or useless)
    X = df_sample.drop('Label', axis=1)
    y = df_sample['Label']
    
    # Simple NaN fill (median for numeric)
    X = X.fillna(X.median(numeric_only=True))
    
    return X, y

def train_random_forest(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="RandomForest"):
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print("Random Forest Accuracy:", acc)
        print(classification_report(y_test, y_pred))
        
        # Log to MLflow
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model")
        
        # Save locally
        model_path = MODELS_DIR / "random_forest.pkl"
        import joblib
        joblib.dump(rf, model_path)
        mlflow.log_artifact(str(model_path))

def train_xgboost(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="XGBoost"):
        label_map = {label: idx for idx, label in enumerate(y_train.unique())}
        y_train_num = y_train.map(label_map)
        y_test_num = y_test.map(label_map)
        
        dtrain = xgb.DMatrix(X_train, label=y_train_num)
        dtest = xgb.DMatrix(X_test, label=y_test_num)
        
        params = {
            'objective': 'multi:softprob',
            'num_class': len(label_map),
            'eval_metric': 'mlogloss',
            'eta': 0.3,
            'max_depth': 6
        }
        
        bst = xgb.train(params, dtrain, num_boost_round=100)
        
        y_pred_prob = bst.predict(dtest)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_pred_labels = [list(label_map.keys())[list(label_map.values()).index(i)] for i in y_pred]
        
        acc = accuracy_score(y_test, y_pred_labels)
        
        print("XGBoost Accuracy:", acc)
        print(classification_report(y_test, y_pred_labels))
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.xgboost.log_model(bst, "model")
        
        # Save locally
        model_path = MODELS_DIR / "xgboost.json"
        bst.save_model(model_path)
        mlflow.log_artifact(str(model_path))

def main():
    mlflow.set_experiment("Hybrid_NIDS_ML_Experiments")
    
    X, y = load_and_prepare_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("Training Random Forest...")
    train_random_forest(X_train, X_test, y_train, y_test)
    
    print("\nTraining XGBoost...")
    train_xgboost(X_train, X_test, y_train, y_test)
    
    print("\nML training complete! View runs with: mlflow ui")

if __name__ == "__main__":
    main()