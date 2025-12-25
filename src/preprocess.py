import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_FILE = PROCESSED_DIR / "cicids2017_processed.csv"

# Create processed directory
PROCESSED_DIR.mkdir(exist_ok=True)

def load_and_clean_single_file(file_path: Path) -> pd.DataFrame:
    print(f"Loading {file_path.name}...")
    df = pd.read_csv(file_path, low_memory=False)

    # Fix column names: strip leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Replace infinity values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    print(f"Shape after load: {df.shape}")
    print(f"NaN count after inf fix: {df.isna().sum().sum()}")

    return df

def main():
    # List all CSV files
    csv_files = sorted([f for f in RAW_DIR.iterdir() if f.suffix == ".csv"])
    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"- {f.name}")
    
    # Load and clean each file
    dfs = []
    for file_path in csv_files:
        df_clean = load_and_clean_single_file(file_path)
        dfs.append(df_clean)
    
    # Merge all
    print("Merging all files...")
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Final merged shape: {full_df.shape}")

    # Basic cleaning
    full_df = full_df.dropna(axis=1, how='all')  # Drop empty columns
    
    # Label column is 'Label' (after strip)
    print("\nFinal label distribution:")
    print(full_df['Label'].value_counts())
    
    # Save processed
    print(f"Saving processed dataset to {PROCESSED_FILE}...")
    full_df.to_csv(PROCESSED_FILE, index=False)
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
