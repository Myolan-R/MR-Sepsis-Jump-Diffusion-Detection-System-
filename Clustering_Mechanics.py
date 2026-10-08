"""
Clustering_Mechanics.py

Builds jump and trend features for each vital sign, summarises every septic
patient's history before sepsis onset, clusters the septic patients with
k means, and saves the cluster assignments.

Run order:
    1. python Compile_Sepsis_Data.py     (creates combined_sepsis_data.csv)
    2. python Clustering_Mechanics.py    (this script, creates septic_cluster_assignments.csv)
    3. python Fit_Jump_Diffusion.py      (pooled jump diffusion fitting)

The same steps are walked through interactively, with plots, in
Sepsis_detection.ipynb.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

INPUT_PATH = "combined_sepsis_data.csv"
OUTPUT_CLUSTERS_PATH = "septic_cluster_assignments.csv"
FIGURE_DIR = Path("figures")

VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
WINDOW_SIZES = {'HR': 12, 'O2Sat': 12, 'SBP': 12, 'MAP': 12, 'Resp': 12, 'DBP': 12, 'Temp': 20}
TREND_WINDOW = 12
JUMP_SIGMA = 3          # a jump is a change larger than 3 rolling standard deviations
MAX_MISSING_STATS = 10  # drop patients with more than this many missing summary statistics
FINAL_K = 3             # chosen after looking at the elbow and silhouette plots


def build_features(df):
    """Forward fill vitals, then add hourly changes, jump flags and rolling trends."""
    df[VITALS] = df.groupby('patient_id')[VITALS].ffill()
    for v in VITALS:
        df[f'{v}_change'] = df.groupby('patient_id')[v].diff()

    for v in VITALS:
        w = WINDOW_SIZES[v]
        df[f'{v}_rolling_std'] = (
            df.groupby('patient_id')[f'{v}_change']
            .transform(lambda x: x.rolling(window=w, min_periods=3).std())
        )
        df[f'{v}_jump'] = (
            df[f'{v}_change'].abs() > JUMP_SIGMA * df[f'{v}_rolling_std']
        ).astype(int)

    for v in VITALS:
        df[f'{v}_rolling_trend'] = (
            df.groupby('patient_id')[v]
            .transform(lambda x: x.rolling(window=TREND_WINDOW, min_periods=6).mean().diff(TREND_WINDOW // 2))
        )
    return df


def summarize_patient(group):
    """One row of summary statistics for one patient's pre onset history."""
    summary = {}
    for v in VITALS:
        valid_vals = group[v].dropna()
        summary[f'{v}_mean'] = group[v].mean()
        summary[f'{v}_std'] = group[v].std()
        summary[f'{v}_total_change'] = (
            valid_vals.iloc[-1] - valid_vals.iloc[0] if len(valid_vals) > 1 else np.nan
        )
        summary[f'{v}_jump_count'] = group[f'{v}_jump'].sum()
        summary[f'{v}_trend_max'] = group[f'{v}_rolling_trend'].max()
    summary['pre_onset_hours'] = len(group)
    return pd.Series(summary)


def main():
    df = pd.read_csv(INPUT_PATH)
    df = build_features(df)

    onset_lookup = (
        df[df['SepsisLabel'] == 1]
        .groupby('patient_id')['hour']
        .min()
        .rename('onset_hour')
    )
    septic_ids = onset_lookup.index
    print(f"Loaded {df.shape[0]:,} rows, {df['patient_id'].nunique():,} patients, {len(septic_ids):,} septic.")

    # Everything before onset, with no fixed cutoff, to capture the whole trajectory.
    septic_window = df[df['patient_id'].isin(septic_ids)].merge(onset_lookup, on='patient_id')
    septic_window = septic_window[septic_window['hour'] < septic_window['onset_hour']]

    septic_summary = septic_window.groupby('patient_id').apply(summarize_patient)
    print(septic_summary.shape)

    # Drop patients with too little real data, then fill the remaining gaps with 0.
    nan_counts = septic_summary.isna().sum(axis=1)
    septic_summary_filtered = septic_summary[nan_counts <= MAX_MISSING_STATS]
    septic_summary_clean = septic_summary_filtered.fillna(0)
    print(
        f"Kept {len(septic_summary_clean)} of {len(septic_summary)} patients "
        f"(dropped {len(septic_summary) - len(septic_summary_filtered)} for having too little real data)"
    )

    X_scaled = StandardScaler().fit_transform(septic_summary_clean)

    # Choose k: elbow and silhouette plots.
    inertias, silhouettes = [], []
    k_range = range(2, 8)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(k_range), inertias, marker='o')
    axes[0].set_xlabel('Number of clusters')
    axes[0].set_ylabel('Inertia (lower means tighter clusters)')
    axes[0].set_title('Elbow method')
    axes[1].plot(list(k_range), silhouettes, marker='o', color='orange')
    axes[1].set_xlabel('Number of clusters')
    axes[1].set_ylabel('Silhouette score (higher means better separated)')
    axes[1].set_title('Silhouette score')
    plt.tight_layout()
    FIGURE_DIR.mkdir(exist_ok=True)
    plt.savefig(FIGURE_DIR / "cluster_selection.png", dpi=150)
    plt.show()

    km_final = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
    septic_summary_clean['cluster'] = km_final.fit_predict(X_scaled)

    print(septic_summary_clean['cluster'].value_counts())
    print(septic_summary_clean.groupby('cluster').mean(numeric_only=True))

    cluster_assignments = septic_summary_clean[['cluster']].reset_index()
    cluster_assignments.to_csv(OUTPUT_CLUSTERS_PATH, index=False)
    print(f"Saved cluster assignments for {len(cluster_assignments)} patients to {OUTPUT_CLUSTERS_PATH}.")


if __name__ == "__main__":
    main()
