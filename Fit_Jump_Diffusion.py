"""
Fit_Jump_Diffusion.py, pooled by group version.

Run this AFTER:
  1. Compile_Sepsis_Data.py has produced combined_sepsis_data.csv
  2. Clustering_Mechanics.py has saved septic_cluster_assignments.csv

Run from a terminal:
    python Fit_Jump_Diffusion.py
"""

import pandas as pd
from Jump_Diffusion_Model import fit_all_patients_by_group

VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
INPUT_PATH = "combined_sepsis_data.csv"
CLUSTER_PATH = "septic_cluster_assignments.csv"
OUTPUT_DATA_PATH = "combined_sepsis_data_with_jumpdiff.csv"
OUTPUT_GROUP_PARAMS_PATH = "jump_diffusion_group_params.csv"
OUTPUT_PATIENT_PARAMS_PATH = "jump_diffusion_patient_params.csv"


def main():
    print("Loading compiled data...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {df.shape[0]:,} rows, {df['patient_id'].nunique():,} patients.")

    print("Forward-filling vitals and computing hourly changes...")
    df[VITALS] = df.groupby('patient_id')[VITALS].ffill()
    for v in VITALS:
        df[f'{v}_change'] = df.groupby('patient_id')[v].diff()

    print("Loading cluster assignments...")
    clusters = pd.read_csv(CLUSTER_PATH)

    df = df.merge(clusters, on='patient_id', how='left')
    df['group'] = df['cluster'].apply(
        lambda c: f"septic_cluster_{int(c)}" if pd.notna(c) else "non_septic"
    )
    df = df.drop(columns=['cluster'])

    print(df['group'].value_counts())

    print("Fitting pooled jump-diffusion model by group...")
    print("(This is the slow part, so expect it to take a while.)")
    df, group_results, patient_results = fit_all_patients_by_group(
        df, VITALS, group_col="group",
        min_group_observations=200,
        min_patient_observations=15,
    )

    print(f"\nFitted {len(group_results)} group-vital models and "
          f"{len(patient_results)} patient-vital models.")
    print(group_results[['group', 'vital', 'lambda', 'mu_j', 'sigma_j', 'n_obs']])

    df.to_csv(OUTPUT_DATA_PATH, index=False)
    group_results.to_csv(OUTPUT_GROUP_PARAMS_PATH, index=False)
    patient_results.to_csv(OUTPUT_PATIENT_PARAMS_PATH, index=False)

    print(f"\nSaved full dataset with jump probabilities to {OUTPUT_DATA_PATH}")
    print(f"Saved group-level parameters to {OUTPUT_GROUP_PARAMS_PATH}")
    print(f"Saved patient-level parameters to {OUTPUT_PATIENT_PARAMS_PATH}")


if __name__ == "__main__":
    main()