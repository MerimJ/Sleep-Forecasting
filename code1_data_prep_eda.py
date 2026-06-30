"""
=============================================================================
CODE 1 — DATA PREPARATION & EXPLORATORY DATA ANALYSIS
Project : Forecasting Cardiovascular Health & Sleep Quality from
          Wearable Time-Series (MLPR / CDA)
Author  : Merim Jusufbegovic
Data    : Didiconn wearable export  2025-11-01 → 2026-05-04
=============================================================================

USAGE (Cursor / terminal):
    pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
    python code1_data_prep_eda.py

Outputs (saved to ./outputs/):
    processed_dataset.csv   — cleaned, merged, feature-engineered dataset
    *.png                   — all EDA figures
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, skew, kurtosis
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS  —  edit DATA_DIR if your files live elsewhere
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.expanduser("~/Downloads/MLPR-CDA")
OUT_DIR   = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

ACTIVITY_FILE    = os.path.join(DATA_DIR, "Activity-Didiconn-2025-11-01-2026-05-04.csv")
SLEEP_FILE       = os.path.join(DATA_DIR, "Sleep-Didiconn-2025-11-01-2026-05-04.csv")
VITALS_FILE      = os.path.join(DATA_DIR, "Vital Signs-Didiconn-2025-11-01-2026-05-04.csv")

print("=" * 70)
print("CODE 1 — DATA PREPARATION & EDA")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading raw data …")

activity = pd.read_csv(ACTIVITY_FILE, parse_dates=["Date"])
vitals   = pd.read_csv(VITALS_FILE,   parse_dates=["Date"])
sleep_raw = pd.read_csv(SLEEP_FILE)

print(f"    Activity : {activity.shape}")
print(f"    Vitals   : {vitals.shape}")
print(f"    Sleep    : {sleep_raw.shape}")
print("\n--- Activity dtypes ---");   print(activity.dtypes)
print("\n--- Vitals dtypes ---");     print(vitals.dtypes)
print("\n--- Sleep dtypes ---");      print(sleep_raw.dtypes)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CLEAN SLEEP DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Cleaning sleep data …")

sleep = sleep_raw.copy()

# Parse datetime columns
for col in ["Start Time", "End Time", "Falling Asleep Time", "Wake-up time"]:
    sleep[col] = pd.to_datetime(sleep[col], errors="coerce")

# Remove % from Sleep Time Ratio and convert to float
sleep["Sleep Time Ratio(%)"] = (
    sleep["Sleep Time Ratio(%)"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .astype(float)
)

# Derive the calendar date of sleep (night the sleep started)
sleep["Date"] = sleep["Start Time"].dt.normalize()
# For nights starting before midnight use same day; after midnight → previous day
# Convention: "sleep date" = date of the evening (start date if before 3am next day)
sleep["sleep_date"] = sleep["Start Time"].apply(
    lambda t: (t - pd.Timedelta(hours=3)).date() if pd.notnull(t) else pd.NaT
)
sleep["sleep_date"] = pd.to_datetime(sleep["sleep_date"])

# Keep only main sleep sessions (longest per night) — drop naps
# A nap = total sleep < 180 min OR duration < 3h
sleep["total_duration_min"] = (
    (sleep["End Time"] - sleep["Start Time"]).dt.total_seconds() / 60
)
sleep_main = (
    sleep[sleep["total_duration_min"] >= 180]
    .sort_values("Time Asleep(min)", ascending=False)
    .drop_duplicates(subset="sleep_date", keep="first")
    .copy()
)
print(f"    Main sleep sessions after nap removal: {len(sleep_main)}")

# Engineered sleep features
sleep_main["sleep_onset_latency_min"] = (
    (sleep_main["Falling Asleep Time"] - sleep_main["Start Time"])
    .dt.total_seconds() / 60
).clip(lower=0)

sleep_main["sleep_efficiency"] = (
    sleep_main["Time Asleep(min)"] / sleep_main["total_duration_min"] * 100
)

sleep_main["wake_after_sleep_onset"] = sleep_main["Sleep Stages - Awake(min)"]

sleep_main["rem_ratio"]   = sleep_main["Sleep Stages - REM(min)"]        / sleep_main["Time Asleep(min)"] * 100
sleep_main["deep_ratio"]  = sleep_main["Sleep Stages - Deep Sleep(min)"] / sleep_main["Time Asleep(min)"] * 100
sleep_main["light_ratio"] = sleep_main["Sleep Stages - Light Sleep(min)"]/ sleep_main["Time Asleep(min)"] * 100

# Binary quality label  (threshold: Sleep Time Ratio ≥ 80 % = Good)
sleep_main["sleep_quality_label"] = (sleep_main["Sleep Time Ratio(%)"] >= 80).astype(int)
# 0 = Poor, 1 = Good

sleep_clean = sleep_main[[
    "sleep_date",
    "Sleep Time Ratio(%)", "Time Asleep(min)", "total_duration_min",
    "sleep_onset_latency_min", "sleep_efficiency",
    "wake_after_sleep_onset",
    "Sleep Stages - Awake(min)", "Sleep Stages - REM(min)",
    "Sleep Stages - Light Sleep(min)", "Sleep Stages - Deep Sleep(min)",
    "rem_ratio", "deep_ratio", "light_ratio",
    "sleep_quality_label"
]].rename(columns={"sleep_date": "Date"})

print(f"    sleep_clean shape: {sleep_clean.shape}")
print(f"    Good nights (label=1): {sleep_clean['sleep_quality_label'].sum()} / {len(sleep_clean)}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CLEAN VITALS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Cleaning vitals …")

# Strip % from SpO2 columns
for col in vitals.columns:
    if "Spo2" in col or "SpO2" in col:
        vitals[col] = vitals[col].astype(str).str.replace("%", "", regex=False).astype(float)

print(vitals.describe())


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MERGE ALL THREE DATASETS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Merging datasets …")

# Align on Date
df = (
    activity
    .merge(vitals,     on="Date", how="outer")
    .merge(sleep_clean, on="Date", how="outer")
    .sort_values("Date")
    .reset_index(drop=True)
)
print(f"    Merged shape: {df.shape}")
print(f"    Date range  : {df['Date'].min()} → {df['Date'].max()}")

# Reindex to full daily calendar to expose implicit missing days
full_idx = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
df = df.set_index("Date").reindex(full_idx).rename_axis("Date").reset_index()
print(f"    After reindex (daily): {df.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MISSING VALUE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Missing value analysis …")

miss = df.isnull().sum()
miss_pct = miss / len(df) * 100
miss_df = pd.DataFrame({"missing": miss, "pct": miss_pct}).query("missing > 0")
print(miss_df.to_string())

# Visualise missingness
fig, ax = plt.subplots(figsize=(12, 5))
miss_df["pct"].sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Missing (%)")
ax.set_title("Missing Values per Feature")
ax.axvline(20, color="red", linestyle="--", label="20 % threshold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_missing_values.png"))
plt.close()
print("    → saved 01_missing_values.png")

# Missingness heatmap (calendar view)
fig, ax = plt.subplots(figsize=(14, 6))
miss_matrix = df.set_index("Date").isnull().astype(int)
sns.heatmap(miss_matrix.T, cbar=False, ax=ax, cmap="Reds",
            xticklabels=False, yticklabels=True)
ax.set_title("Missingness Heatmap (red = missing)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_missingness_heatmap.png"))
plt.close()
print("    → saved 02_missingness_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  IMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Imputing missing values …")

# Columns with <20% missing → KNN imputation (k=5)
# Sleep columns missing because no sleep recorded → forward-fill then KNN
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Forward-fill sleep columns (carry last known value over skipped days)
sleep_cols = [c for c in numeric_cols if any(
    k in c for k in ["Sleep", "sleep", "rem", "deep", "light", "wake", "onset"]
)]
df[sleep_cols] = df[sleep_cols].fillna(method="ffill", limit=2)

# KNN imputation for remaining numeric columns
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

print(f"    Remaining nulls after imputation: {df.isnull().sum().sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  OUTLIER DETECTION  (IQR + Z-score)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Outlier detection …")

key_features = [
    "Steps", "Calories(kcal)",
    "Avg. Heart Rate(bpm)", "Avg. HRV(ms)",
    "Avg. Spo2(%)",
    "Sleep Time Ratio(%)", "Time Asleep(min)",
    "sleep_onset_latency_min", "wake_after_sleep_onset"
]
key_features = [f for f in key_features if f in df.columns]

outlier_summary = {}
for col in key_features:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_iqr = ((df[col] < lower) | (df[col] > upper)).sum()

    z = np.abs(stats.zscore(df[col].dropna()))
    n_z = (z > 3).sum()

    outlier_summary[col] = {"IQR_outliers": n_iqr, "Z_outliers": n_z,
                             "lower_fence": round(lower,2), "upper_fence": round(upper,2)}

outlier_df = pd.DataFrame(outlier_summary).T
print(outlier_df.to_string())

# Box plots for key features
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(key_features):
    if i < len(axes):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="lightblue"))
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel("Value")
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Box Plots — Key Features (Outlier Detection)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_boxplots_outliers.png"))
plt.close()
print("    → saved 03_boxplots_outliers.png")

# Winsorize at 1st/99th percentile (cap, do not remove rows — small dataset)
for col in key_features:
    p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=p1, upper=p99)
print("    Outliers winsorized at 1st/99th percentile.")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Feature engineering …")

df = df.sort_values("Date").reset_index(drop=True)

# ── 8a. Lag features (daytime predictors → NEXT night's sleep)
lag_src = ["Steps", "Calories(kcal)", "Avg. HRV(ms)", "Min. HRV(ms)",
           "Avg. Heart Rate(bpm)", "Avg. Spo2(%)", "Min. Spo2(%)"]
lag_src = [f for f in lag_src if f in df.columns]

for col in lag_src:
    for lag in [1, 2, 3]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

# ── 8b. Rolling statistics (3-day and 7-day window)
for col in ["Steps", "Avg. HRV(ms)", "Avg. Heart Rate(bpm)"]:
    if col in df.columns:
        df[f"{col}_roll3_mean"] = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_roll7_mean"] = df[col].rolling(7, min_periods=1).mean()
        df[f"{col}_roll7_std"]  = df[col].rolling(7, min_periods=1).std()

# ── 8c. HRV-based features
if "Avg. HRV(ms)" in df.columns and "Min. HRV(ms)" in df.columns:
    df["hrv_range"]    = df["Max. HRV(ms)"] - df["Min. HRV(ms)"]
    df["hrv_low_flag"] = (df["Avg. HRV(ms)"] < df["Avg. HRV(ms)"].quantile(0.25)).astype(int)

# ── 8d. Activity intensity
if "Steps" in df.columns:
    df["activity_low"]  = (df["Steps"] < 5000).astype(int)
    df["activity_high"] = (df["Steps"] > 10000).astype(int)

# ── 8e. Day-of-week cyclical encoding
df["dow"] = df["Date"].dt.dayofweek
df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
df["is_weekend"] = (df["dow"] >= 5).astype(int)

# ── 8f. Month seasonality
df["month"] = df["Date"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# ── 8g. Target: next-night sleep quality (shift sleep features back by 1)
df["next_sleep_ratio"]   = df["Sleep Time Ratio(%)"].shift(-1)
df["next_sleep_quality"] = df["sleep_quality_label"].shift(-1)   # 0/1

print(f"    Dataset after feature engineering: {df.shape}")
print(f"    New columns: {[c for c in df.columns if 'lag' in c or 'roll' in c][:10]} …")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Distribution analysis …")

dist_cols = key_features
n = len(dist_cols)
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()

norm_results = {}
for i, col in enumerate(dist_cols):
    if i >= len(axes):
        break
    data = df[col].dropna()
    axes[i].hist(data, bins=20, color="steelblue", edgecolor="white", alpha=0.8, density=True)
    # Overlay normal KDE
    xmin, xmax = data.min(), data.max()
    x = np.linspace(xmin, xmax, 100)
    mu, sigma = data.mean(), data.std()
    axes[i].plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label="Normal PDF")
    axes[i].set_title(f"{col}\nSkew={skew(data):.2f} | Kurt={kurtosis(data):.2f}", fontsize=8)
    axes[i].legend(fontsize=7)

    # Shapiro-Wilk normality test
    stat, p = shapiro(data[:50] if len(data) > 50 else data)   # Shapiro needs ≤5000
    norm_results[col] = {"W": round(stat, 4), "p": round(p, 4),
                         "normal_H0": "Reject" if p < 0.05 else "Fail to reject"}

for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Distributions & Normality (red = Normal PDF)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_distributions.png"))
plt.close()
print("    → saved 04_distributions.png")

norm_df = pd.DataFrame(norm_results).T
print("\n    Shapiro-Wilk results:")
print(norm_df.to_string())

# Q-Q plots
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
axes = axes.flatten()
for i, col in enumerate(dist_cols):
    if i >= len(axes): break
    sm.qqplot(df[col].dropna(), line="s", ax=axes[i], alpha=0.5)
    axes[i].set_title(f"Q-Q: {col}", fontsize=9)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Q-Q Plots (Normality Check)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_qq_plots.png"))
plt.close()
print("    → saved 05_qq_plots.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  TIME-SERIES VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Time-series plots …")

ts_groups = {
    "Activity"      : ["Steps", "Calories(kcal)"],
    "Heart Rate"    : ["Avg. Heart Rate(bpm)", "Min. Heart Rate(bpm)", "Max. Heart Rate(bpm)"],
    "HRV"           : ["Avg. HRV(ms)", "Min. HRV(ms)", "Max. HRV(ms)"],
    "SpO2"          : ["Avg. Spo2(%)", "Min. Spo2(%)"],
    "Sleep Quality" : ["Sleep Time Ratio(%)", "Time Asleep(min)", "sleep_efficiency"],
    "Sleep Stages"  : ["Sleep Stages - REM(min)", "Sleep Stages - Deep Sleep(min)",
                       "Sleep Stages - Light Sleep(min)", "Sleep Stages - Awake(min)"],
}

for group, cols in ts_groups.items():
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 2.5 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        ax.plot(df["Date"], df[col], lw=1.5, color="steelblue")
        ax.fill_between(df["Date"], df[col], alpha=0.15, color="steelblue")
        # 7-day rolling mean overlay
        roll = df[col].rolling(7, min_periods=1).mean()
        ax.plot(df["Date"], roll, color="crimson", lw=1.5, linestyle="--", label="7d MA")
        ax.set_ylabel(col, fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].set_xlabel("Date")
    plt.suptitle(f"Time Series — {group}", fontsize=12)
    plt.tight_layout()
    fname = f"06_ts_{group.lower().replace(' ','_')}.png"
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()
    print(f"    → saved {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  CORRELATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] Correlation analysis …")

corr_features = [
    "Steps", "Calories(kcal)",
    "Avg. Heart Rate(bpm)", "Avg. HRV(ms)", "Min. HRV(ms)", "hrv_low_flag",
    "Avg. Spo2(%)", "Min. Spo2(%)",
    "Sleep Time Ratio(%)", "Time Asleep(min)", "sleep_efficiency",
    "wake_after_sleep_onset", "sleep_onset_latency_min",
    "Sleep Stages - REM(min)", "Sleep Stages - Deep Sleep(min)",
    "rem_ratio", "deep_ratio"
]
corr_features = [c for c in corr_features if c in df.columns]

corr_df = df[corr_features].dropna()

# Pearson correlation matrix
pearson_corr = corr_df.corr(method="pearson")
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
sns.heatmap(pearson_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, ax=ax, linewidths=0.3, annot_kws={"size": 7})
ax.set_title("Pearson Correlation Matrix", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_pearson_corr.png"))
plt.close()
print("    → saved 07_pearson_corr.png")

# Spearman correlation matrix
spearman_corr = corr_df.corr(method="spearman")
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(spearman_corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, ax=ax, linewidths=0.3, annot_kws={"size": 7})
ax.set_title("Spearman Rank Correlation Matrix", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_spearman_corr.png"))
plt.close()
print("    → saved 08_spearman_corr.png")

# ── Correlations with sleep quality target
print("\n    Correlation with Sleep Time Ratio (Spearman):")
target_corr = spearman_corr["Sleep Time Ratio(%)"].drop("Sleep Time Ratio(%)").sort_values()
print(target_corr.to_string())

fig, ax = plt.subplots(figsize=(8, 7))
colors = ["crimson" if v < 0 else "steelblue" for v in target_corr]
target_corr.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Spearman ρ with Sleep Time Ratio (%)")
ax.set_xlabel("Spearman ρ")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "09_target_correlation.png"))
plt.close()
print("    → saved 09_target_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  HRV → SLEEP FRAGMENTATION ANALYSIS (Research Question 3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] HRV vs next-night sleep fragmentation …")

if "Avg. HRV(ms)" in df.columns and "next_sleep_ratio" in df.columns:
    hrv_sleep = df[["Date", "Avg. HRV(ms)", "hrv_low_flag",
                    "next_sleep_ratio", "next_sleep_quality",
                    "wake_after_sleep_onset"]].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: HRV vs next-night sleep ratio
    c = hrv_sleep["next_sleep_quality"].map({0: "crimson", 1: "steelblue"})
    axes[0].scatter(hrv_sleep["Avg. HRV(ms)"], hrv_sleep["next_sleep_ratio"],
                    c=c, alpha=0.7, edgecolors="k", linewidths=0.3)
    m, b = np.polyfit(hrv_sleep["Avg. HRV(ms)"], hrv_sleep["next_sleep_ratio"], 1)
    x_line = np.linspace(hrv_sleep["Avg. HRV(ms)"].min(), hrv_sleep["Avg. HRV(ms)"].max(), 100)
    axes[0].plot(x_line, m * x_line + b, "k--", lw=1.5, label=f"Trend (slope={m:.2f})")
    axes[0].set_xlabel("Avg. HRV (ms)")
    axes[0].set_ylabel("Next Night Sleep Ratio (%)")
    axes[0].set_title("HRV (Day) vs Next-Night Sleep Quality")
    axes[0].legend()
    # Add blue/red legend
    from matplotlib.patches import Patch
    axes[0].legend(handles=[
        Patch(color="steelblue", label="Good sleep"),
        Patch(color="crimson",   label="Poor sleep"),
        plt.Line2D([0],[0], color="k", linestyle="--", label=f"Trend (slope={m:.2f})")
    ])

    # Box plot: Low HRV day → sleep quality
    hrv_sleep.boxplot(column="next_sleep_ratio", by="hrv_low_flag", ax=axes[1])
    axes[1].set_title("Next-Night Sleep Ratio by Low HRV Flag")
    axes[1].set_xlabel("Low HRV Day (1=below 25th pct)")
    axes[1].set_ylabel("Sleep Time Ratio (%)")
    axes[1].xaxis.set_ticklabels(["Normal HRV", "Low HRV"])
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "10_hrv_vs_sleep.png"))
    plt.close()
    print("    → saved 10_hrv_vs_sleep.png")

    # Mann-Whitney U test: Low vs normal HRV nights
    low_hrv_sleep  = hrv_sleep[hrv_sleep["hrv_low_flag"] == 1]["next_sleep_ratio"]
    high_hrv_sleep = hrv_sleep[hrv_sleep["hrv_low_flag"] == 0]["next_sleep_ratio"]
    u_stat, p_val = stats.mannwhitneyu(low_hrv_sleep, high_hrv_sleep, alternative="less")
    print(f"    Mann-Whitney U: U={u_stat:.1f}, p={p_val:.4f}")
    print(f"    Low HRV  → mean sleep ratio: {low_hrv_sleep.mean():.1f}%")
    print(f"    Norm HRV → mean sleep ratio: {high_hrv_sleep.mean():.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  STATIONARITY TESTS  (ADF)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] Stationarity tests (ADF) …")

adf_cols = ["Steps", "Avg. HRV(ms)", "Avg. Heart Rate(bpm)", "Sleep Time Ratio(%)"]
adf_cols = [c for c in adf_cols if c in df.columns]

for col in adf_cols:
    series = df[col].dropna()
    adf_result = adfuller(series, autolag="AIC")
    print(f"    {col}: ADF={adf_result[0]:.4f}, p={adf_result[1]:.4f}  "
          f"→ {'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'}")


# ─────────────────────────────────────────────────────────────────────────────
# 14.  ACF / PACF  (autocorrelation in sleep quality)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] ACF / PACF for sleep quality …")

sleep_series = df["Sleep Time Ratio(%)"].dropna()
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_acf (sleep_series, lags=20, ax=axes[0], title="ACF  — Sleep Time Ratio (%)")
plot_pacf(sleep_series, lags=20, ax=axes[1], title="PACF — Sleep Time Ratio (%)",
          method="ywm")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "11_acf_pacf_sleep.png"))
plt.close()
print("    → saved 11_acf_pacf_sleep.png")


# ─────────────────────────────────────────────────────────────────────────────
# 15.  PAIRPLOT (core variables)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[15] Pairplot …")

pair_cols = ["Steps", "Avg. HRV(ms)", "Avg. Heart Rate(bpm)",
             "Sleep Time Ratio(%)", "Time Asleep(min)", "wake_after_sleep_onset"]
pair_cols = [c for c in pair_cols if c in df.columns]

pair_df = df[pair_cols + ["sleep_quality_label"]].dropna()
pair_df["Sleep Quality"] = pair_df["sleep_quality_label"].map({0: "Poor", 1: "Good"})

g = sns.pairplot(pair_df.drop("sleep_quality_label", axis=1),
                 hue="Sleep Quality", palette={"Poor": "crimson", "Good": "steelblue"},
                 plot_kws={"alpha": 0.6, "s": 30}, diag_kind="kde")
g.fig.suptitle("Pairplot — Core Features by Sleep Quality Label", y=1.02)
plt.savefig(os.path.join(OUT_DIR, "12_pairplot.png"), bbox_inches="tight")
plt.close()
print("    → saved 12_pairplot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 16.  WEEKEND vs WEEKDAY SLEEP COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n[16] Weekend vs weekday analysis …")

weekend_df = df[["is_weekend", "Sleep Time Ratio(%)", "Time Asleep(min)",
                 "sleep_onset_latency_min", "Avg. HRV(ms)"]].dropna()

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, col in zip(axes, ["Sleep Time Ratio(%)", "Time Asleep(min)",
                           "sleep_onset_latency_min", "Avg. HRV(ms)"]):
    weekend_df.boxplot(column=col, by="is_weekend", ax=ax)
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("")
    ax.xaxis.set_ticklabels(["Weekday", "Weekend"])
plt.suptitle("Weekend vs Weekday Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "13_weekend_weekday.png"))
plt.close()
print("    → saved 13_weekend_weekday.png")


# ─────────────────────────────────────────────────────────────────────────────
# 17.  DESCRIPTIVE STATISTICS SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[17] Descriptive statistics …")

desc = df[key_features].describe(percentiles=[0.25, 0.5, 0.75]).T
desc["skewness"] = df[key_features].apply(lambda c: skew(c.dropna()))
desc["kurtosis"] = df[key_features].apply(lambda c: kurtosis(c.dropna()))
print(desc.round(2).to_string())


# ─────────────────────────────────────────────────────────────────────────────
# 18.  SAVE PROCESSED DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[18] Saving processed dataset …")

# Drop rows where the next-night target is unknown (last row(s))
df_model = df.dropna(subset=["next_sleep_ratio", "next_sleep_quality"]).copy()
df_model = df_model.drop(columns=["dow"])   # redundant after encoding

out_path = os.path.join(DATA_DIR, "processed_dataset.csv")
df_model.to_csv(out_path, index=False)
print(f"    Saved → {out_path}")
print(f"    Shape : {df_model.shape}")
print(f"    Target distribution:")
print(df_model["next_sleep_quality"].value_counts(normalize=True).rename({0:"Poor", 1:"Good"}).to_string())

print("\n" + "=" * 70)
print("CODE 1 COMPLETE — all outputs saved to:", OUT_DIR)
print("=" * 70)
