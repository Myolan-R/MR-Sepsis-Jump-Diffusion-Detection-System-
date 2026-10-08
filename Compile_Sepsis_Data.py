"""
Compile_Sepsis_Data.py

Combines the individual patient files from the PhysioNet/Computing in
Cardiology Challenge 2019 (training Set A) into one table.

Before running, download the training data from PhysioNet and place the
.psv patient files in:  data/training_setA/

Outputs (written to the current folder):
    combined_sepsis_data.csv         full combined dataset
    combined_sepsis_data_sample.csv  50,000 row random sample for a quick look

Run from a terminal:
    python Compile_Sepsis_Data.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data") / "training_setA"
OUTPUT_CSV = "combined_sepsis_data.csv"
OUTPUT_CSV_SAMPLE = "combined_sepsis_data_sample.csv"


def compile_all_patients(data_dir):
    """Read every .psv patient file and stack them into one DataFrame."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Could not find {data_dir}. Download the PhysioNet 2019 training data "
            "and place the .psv files in this folder."
        )

    psv_files = list(data_dir.rglob("*.psv"))
    print(f"Found {len(psv_files)} patient files.")

    all_dfs = []
    for i, filepath in enumerate(psv_files):
        df = pd.read_csv(filepath, sep="|")
        df["patient_id"] = filepath.stem
        df["hour"] = df.index
        all_dfs.append(df)

        if (i + 1) % 2000 == 0:
            print(f"  Loaded {i + 1} / {len(psv_files)} files...")

    return pd.concat(all_dfs, ignore_index=True)


def main():
    combined = compile_all_patients(DATA_DIR)

    print(f"\nCombined dataset shape: {combined.shape[0]:,} rows x {combined.shape[1]} columns")
    print(f"Unique patients: {combined['patient_id'].nunique():,}")
    print(
        f"Sepsis positive rows: {combined['SepsisLabel'].sum():,} "
        f"({100 * combined['SepsisLabel'].mean():.2f}% of all rows)"
    )

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFull dataset saved to {OUTPUT_CSV}")

    combined.sample(n=min(50000, len(combined)), random_state=42).to_csv(
        OUTPUT_CSV_SAMPLE, index=False
    )
    print(f"A 50,000 row random sample saved to {OUTPUT_CSV_SAMPLE}")


if __name__ == "__main__":
    main()
