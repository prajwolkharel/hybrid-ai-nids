import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import shap
import mlflow
import mlflow.tensorflow
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Config
PROCESSED_FILE = Path("data/processed/cicids2017_processed.csv")
MODELS_DIR = Path("models")
SHAP_DIR = Path("docs/shap_plots")
MODELS_DIR.mkdir(exist_ok=True)
SHAP_DIR.mkdir(exist_ok=True)
SAMPLE_SIZE = 300000  # Slightly larger to help rare classes

def prepare_sequences(X, y, sequence_length=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_samples = len(X_scaled) // sequence_length
    X_seq = X_scaled[:n_samples * sequence_length].reshape((n_samples, sequence_length, X_scaled.shape[1]))
    y_seq = y[:n_samples * sequence_length:sequence_length]
    
    return X_seq, y_seq, scaler

def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    mlflow.set_experiment("Hybrid_NIDS_DL_Experiments")
    
    with mlflow.start_run(run_name="LSTM_SMOTE_SHAP"):
        print("Loading data...")
        df = pd.read_csv(PROCESSED_FILE)
        
        # Encode labels
        le = LabelEncoder()
        df['Label_encoded'] = le.fit_transform(df['Label'])
        
        feature_cols = [col for col in df.columns if col not in ['Label', 'Label_encoded']]
        X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
        y = df['Label_encoded']
        
        # Sample + SMOTE with low k_neighbors for rare classes
        X_sample = X.sample(n=SAMPLE_SIZE, random_state=42)
        y_sample = y.loc[X_sample.index]
        
        # Use k_neighbors=1 to handle classes with very few samples
        smote = SMOTE(random_state=42, k_neighbors=1)
        X_res, y_res = smote.fit_resample(X_sample, y_sample)
        
        print(f"After SMOTE: {X_res.shape[0]} samples, {len(np.unique(y_res))} classes")
        
        # Create sequences
        seq_length = 10
        X_seq, y_seq, scaler = prepare_sequences(X_res.values, y_res.values, seq_length)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)
        
        num_classes = len(np.unique(y_seq))
        
        # Train LSTM
        model = build_lstm_model((seq_length, X_seq.shape[2]), num_classes)
        model.fit(X_train, y_train, epochs=15, batch_size=256, validation_split=0.1, verbose=1)
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nLSTM Test Accuracy: {test_acc:.4f}")
        
        # MLflow logging
        mlflow.log_param("model", "LSTM")
        mlflow.log_param("sequence_length", seq_length)
        mlflow.log_param("epochs", 15)
        mlflow.log_param("smote_k_neighbors", 1)
        mlflow.log_metric("test_accuracy", test_acc)
        
                # SHAP with DeepExplainer (better for neural nets)
        print("Generating SHAP explanations...")
        background = X_train[:100]  # Small background
        test_samples = X_test[:20]   # Small for speed
        
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test_samples)
        
        # Plot summary for first class (or average)
        shap.summary_plot(shap_values[0], test_samples[:, -1, :], feature_names=feature_cols, show=False)  # Use last timestep
        plt.savefig(SHAP_DIR / "lstm_shap_summary.png", bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(str(SHAP_DIR / "lstm_shap_summary.png"))
        
        print("SHAP plot saved!")        
        # Save model and helpers
        model.save(MODELS_DIR / "lstm_model.keras")
        joblib.dump(le, MODELS_DIR / "label_encoder_dl.pkl")
        joblib.dump(scaler, MODELS_DIR / "scaler_dl.pkl")
        mlflow.tensorflow.log_model(model, "lstm_model")
        
        print("Phase 6 complete! LSTM + SHAP ready.")

if __name__ == "__main__":
    main()