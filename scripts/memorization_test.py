import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# =========================================================
# LOAD SYNTHETIC DATASETS
# =========================================================
synth_df0 = pd.read_pickle('../data/outputs/synthetic/synthetic_GC_diversions_default_300.pkl')
synth_df1 = pd.read_pickle('../data/outputs/synthetic/synthetic_TVAE_diversions_default_300.pkl')
synth_df2 = pd.read_pickle('../data/outputs/synthetic/synthetic_TVAE_diversions_optimal_300.pkl')
synth_df3 = pd.read_pickle('../data/outputs/synthetic/synthetic_CTGAN_diversions_default_300.pkl')
synth_df4 = pd.read_pickle('../data/outputs/synthetic/synthetic_CTGAN_diversions_optimal_300.pkl')
synth_df5 = pd.read_pickle('../data/outputs/synthetic/synthetic_COPGAN_diversions_default_300.pkl')
synth_df6 = pd.read_pickle('../data/outputs/synthetic/synthetic_COPGAN_diversions_optimal_300.pkl')

# =========================================================
# STORE DATASETS IN LIST
# =========================================================
synthetic_dfs = [
    synth_df0,
    synth_df1,
    synth_df2,
    synth_df3,
    synth_df4,
    synth_df5,
    synth_df6
]

cases = [
    "GC default",
    "TVAE default",
    "TVAE optimal",
    "CTGAN default",
    "CTGAN optimal",
    "COPGAN default",
    "COPGAN optimal"
]

# =========================================================
# LOAD REAL DATA
# =========================================================
real_df = pd.read_pickle(
    '../data/preprocessed_data/real_div_with_relational.pkl'
)

# =========================================================
# KEEP ONLY NUMERICAL COLUMNS
# =========================================================
numeric_cols = real_df.select_dtypes(include=np.number).columns

real_num = real_df[numeric_cols].copy()

# =========================================================
# HANDLE NaNs
# =========================================================
real_num = real_num.fillna(real_num.median())

# =========================================================
# NORMALIZE USING REAL DATA STATISTICS
# =========================================================
scaler = StandardScaler()

real_scaled = scaler.fit_transform(real_num)

# =========================================================
# FIT NEAREST-NEIGHBOR MODEL ON REAL DATA
# =========================================================
nn = NearestNeighbors(
    n_neighbors=1,
    metric="euclidean"
)

nn.fit(real_scaled)

# =========================================================
# REAL-TO-REAL BASELINE
# =========================================================
nn_real = NearestNeighbors(
    n_neighbors=2,  # first neighbor is itself
    metric="euclidean"
)

nn_real.fit(real_scaled)

real_distances, _ = nn_real.kneighbors(real_scaled)

# Second neighbor = closest OTHER real sample
real_baseline = real_distances[:, 1]

baseline_mean = real_baseline.mean()

print("\n========== Real-to-Real Baseline ==========")
print(f"Mean baseline DCR: {baseline_mean:.4f}")

# =========================================================
# LOOP OVER SYNTHETIC DATASETS
# =========================================================
for i, synth_df in enumerate(synthetic_dfs):

    print("\n" + "=" * 60)
    print(f"Case: {cases[i]}")
    print("=" * 60)

    synth_df = synth_df.copy()

    # -----------------------------------------------------
    # Keep only columns existing in both datasets
    # -----------------------------------------------------
    common_cols = [
        col for col in numeric_cols
        if col in synth_df.columns
    ]

    synth_num = synth_df[common_cols].copy()
    real_num_subset = real_num[common_cols].copy()

    # -----------------------------------------------------
    # Handle NaNs
    # -----------------------------------------------------
    synth_num = synth_num.fillna(real_num_subset.median())

    # -----------------------------------------------------
    # Refit scaler using matching columns only
    # -----------------------------------------------------
    scaler = StandardScaler()

    real_scaled_subset = scaler.fit_transform(real_num_subset)
    synth_scaled = scaler.transform(synth_num)

    # -----------------------------------------------------
    # Fit NN on matching real subset
    # -----------------------------------------------------
    nn.fit(real_scaled_subset)

    # -----------------------------------------------------
    # Compute DCR
    # -----------------------------------------------------
    distances, indices = nn.kneighbors(synth_scaled)

    dcr_values = distances.flatten()

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------
    print("\n========== DCR Statistics ==========")
    print(f"Mean DCR   : {dcr_values.mean():.4f}")
    print(f"Median DCR : {np.median(dcr_values):.4f}")
    print(f"Min DCR    : {dcr_values.min():.4f}")
    print(f"Max DCR    : {dcr_values.max():.4f}")

    # -----------------------------------------------------
    # Add DCR column
    # -----------------------------------------------------
    synth_df["DCR"] = dcr_values

    # -----------------------------------------------------
    # Detect potential memorization
    # -----------------------------------------------------
    threshold = 0.05

    memorized = synth_df[synth_df["DCR"] < threshold]

    print("\n========== Potential Memorization ==========")
    print(f"Threshold            : {threshold}")
    print(f"Potential copies     : {len(memorized)}")
    print(
        f"Percentage memorized : "
        f"{100 * len(memorized) / len(synth_df):.2f}%"
    )

    # -----------------------------------------------------
    # Compare against baseline
    # -----------------------------------------------------
    print("\n========== Comparison ==========")
    print(f"Synthetic->Real mean DCR : {dcr_values.mean():.4f}")
    print(f"Real->Real mean DCR      : {baseline_mean:.4f}")

    ratio = dcr_values.mean() / baseline_mean

    print(f"DCR ratio                : {ratio:.4f}")

    if ratio < 1:
        print("Potential memorization detected.")
    else:
        print("No strong memorization evidence.")