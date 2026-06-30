"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MLPR-CDA · CODE 1 · DATA PREPARATION & EXPLORATORY DATA ANALYSIS          ║
║  Project : Forecasting Cardiovascular Health & Sleep Quality                 ║
║            from Wearable Time-Series                                         ║
║  Author  : Merim Jusufbegovic                                                ║
║  Data    : Didiconn wearable  2025-11-01 → 2026-05-04                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE
─────
  pip install -r requirements.txt
  python code1_eda.py

OUTPUT → ~/Downloads/MLPR-CDA/outputs/figures/
  fig1_dataset_overview.png       fig7_sleep_stage_composition.png
  fig2_missing_data.png           fig8_hrv_sleep_scatter.png
  fig3_timeseries_dashboard.png   fig9_good_vs_poor_sleep.png
  fig4_outlier_detection.png      fig10_statistical_tests.png
  fig5_distribution_analysis.png  fig11_acf_pacf.png
  fig6_correlation_matrix.png     processed_dataset.csv
"""

# ═══════════════════════════════════════════════════════════════════════════════
# §0  IMPORTS & GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════════════
import os, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy import stats
from scipy.stats import (shapiro, mannwhitneyu, pearsonr, spearmanr,
                          skew, kurtosis, normaltest)
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings("ignore")

# ─── Publication-quality style ───────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi"        : 150,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "savefig.facecolor" : "white",
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#FAFBFD",
    "axes.edgecolor"    : "#CCCCCC",
    "axes.linewidth"    : 0.8,
    "axes.grid"         : True,
    "grid.color"        : "#E5E5E5",
    "grid.linestyle"    : "--",
    "grid.linewidth"    : 0.5,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "font.family"       : "sans-serif",
    "font.sans-serif"   : ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size"         : 10,
    "axes.titlesize"    : 12,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 10,
    "axes.labelcolor"   : "#2C3E50",
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "legend.framealpha" : 0.9,
    "legend.edgecolor"  : "#CCCCCC",
    "lines.linewidth"   : 1.8,
})

# ─── Colour palette ─────────────────────────────────────────────────────────
C = {
    "navy"     : "#1B2A4A",
    "teal"     : "#0B7A75",
    "blue"     : "#2471A3",
    "orange"   : "#E67E22",
    "red"      : "#C0392B",
    "green"    : "#1E8449",
    "purple"   : "#7D3C98",
    "gray"     : "#808B96",
    "lightbg"  : "#EBF5FB",
    "good"     : "#1E8449",
    "poor"     : "#C0392B",
    "hrv"      : "#0B7A75",
    "steps"    : "#2471A3",
    "sleep"    : "#7D3C98",
    "hr"       : "#C0392B",
    "spo2"     : "#E67E22",
}
PALETTE = [C["blue"], C["teal"], C["orange"], C["red"],
           C["green"], C["purple"], C["gray"]]

def section(title):
    bar = "═" * 70
    print(f"\n{bar}\n  {title}\n{bar}")

def subsection(title):
    print(f"\n  ▸ {title}")

print("╔" + "═"*68 + "╗")
print("║  MLPR-CDA · CODE 1 · DATA PREPARATION & EDA" + " "*23 + "║")
print("╚" + "═"*68 + "╝")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  PATHS
# ═══════════════════════════════════════════════════════════════════════════════
# Use project root if CSVs are present there, otherwise fall back to ~/Downloads/MLPR-CDA
_script_dir = os.path.dirname(os.path.abspath(__file__))
_downloads  = os.path.expanduser("~/Downloads/MLPR-CDA")
DATA_DIR    = _script_dir if os.path.exists(os.path.join(_script_dir, "Activity-Didiconn-2025-11-01-2026-05-04.csv")) else _downloads
FIG_DIR     = os.path.join(_script_dir, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

ACTIVITY_F = os.path.join(DATA_DIR, "Activity-Didiconn-2025-11-01-2026-05-04.csv")
SLEEP_F    = os.path.join(DATA_DIR, "Sleep-Didiconn-2025-11-01-2026-05-04.csv")
VITALS_F   = os.path.join(DATA_DIR, "Vital Signs-Didiconn-2025-11-01-2026-05-04.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# §2  LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════
section("§2  LOADING RAW DATA")

act  = pd.read_csv(ACTIVITY_F, parse_dates=["Date"])
vit  = pd.read_csv(VITALS_F,   parse_dates=["Date"])
slp  = pd.read_csv(SLEEP_F)

for col in ["Start Time", "End Time", "Falling Asleep Time", "Wake-up time"]:
    slp[col] = pd.to_datetime(slp[col], errors="coerce")

slp["Sleep Time Ratio(%)"] = (
    slp["Sleep Time Ratio(%)"].astype(str)
    .str.replace("%", "", regex=False).astype(float)
)

for c in vit.columns:
    if "Spo2" in c or "SpO2" in c:
        vit[c] = vit[c].astype(str).str.replace("%", "", regex=False).astype(float)

print(f"  Activity : {act.shape[0]} days  |  Vitals : {vit.shape[0]} days  |  Sleep sessions : {slp.shape[0]}")


# ═══════════════════════════════════════════════════════════════════════════════
# §3  CLEAN & ENGINEER SLEEP DATA
# ═══════════════════════════════════════════════════════════════════════════════
section("§3  CLEAN & ENGINEER SLEEP DATA")

slp["total_min"] = (slp["End Time"] - slp["Start Time"]).dt.total_seconds() / 60
slp["sleep_date"] = slp["Start Time"].apply(
    lambda t: (t - pd.Timedelta(hours=3)).normalize() if pd.notnull(t) else pd.NaT
)

# Keep only main sleep (≥ 180 min), one per night
slp_main = (
    slp[slp["total_min"] >= 180]
    .sort_values("Time Asleep(min)", ascending=False)
    .drop_duplicates("sleep_date", keep="first")
    .copy()
)

slp_main["onset_latency"]   = ((slp_main["Falling Asleep Time"] - slp_main["Start Time"])
                                .dt.total_seconds() / 60).clip(lower=0)
slp_main["efficiency"]      = slp_main["Time Asleep(min)"] / slp_main["total_min"] * 100
slp_main["waso"]            = slp_main["Sleep Stages - Awake(min)"]
slp_main["rem_ratio"]       = slp_main["Sleep Stages - REM(min)"]          / slp_main["Time Asleep(min)"] * 100
slp_main["deep_ratio"]      = slp_main["Sleep Stages - Deep Sleep(min)"]   / slp_main["Time Asleep(min)"] * 100
slp_main["light_ratio"]     = slp_main["Sleep Stages - Light Sleep(min)"]  / slp_main["Time Asleep(min)"] * 100
slp_main["quality_label"]   = (slp_main["Sleep Time Ratio(%)"] >= 80).astype(int)

slp_clean = slp_main.rename(columns={"sleep_date": "Date"})[[
    "Date", "Sleep Time Ratio(%)", "Time Asleep(min)", "total_min",
    "onset_latency", "efficiency", "waso",
    "Sleep Stages - Awake(min)", "Sleep Stages - REM(min)",
    "Sleep Stages - Light Sleep(min)", "Sleep Stages - Deep Sleep(min)",
    "rem_ratio", "deep_ratio", "light_ratio", "quality_label"
]]

nap_cnt = len(slp) - len(slp_main)
print(f"  Nap sessions removed : {nap_cnt}  |  Main sleep nights : {len(slp_clean)}")
print(f"  Good nights (≥80%) : {slp_clean['quality_label'].sum()}  |  "
      f"Poor nights (<80%) : {(slp_clean['quality_label']==0).sum()}")


# ═══════════════════════════════════════════════════════════════════════════════
# §4  MERGE & REINDEX
# ═══════════════════════════════════════════════════════════════════════════════
section("§4  MERGE & DAILY REINDEX")

df = (act.merge(vit, on="Date", how="outer")
         .merge(slp_clean, on="Date", how="outer")
         .sort_values("Date").reset_index(drop=True))

full_idx = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
df = df.set_index("Date").reindex(full_idx).rename_axis("Date").reset_index()

print(f"  Merged shape  : {df.shape}  |  {df['Date'].min().date()} → {df['Date'].max().date()}")


# ═══════════════════════════════════════════════════════════════════════════════
# §5  FIGURE 1 – DATASET OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
section("§5  FIGURE 1 – DATASET OVERVIEW DASHBOARD")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("white")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

# ── Panel A: Calendar-style availability ─────────────────────────────────────
ax_cal = fig.add_subplot(gs[0, :3])
avail = pd.DataFrame({
    "Date"     : df["Date"],
    "Activity" : (~df["Steps"].isna()).astype(int),
    "Vitals"   : (~df["Avg. Heart Rate(bpm)"].isna()).astype(int),
    "Sleep"    : (~df["Sleep Time Ratio(%)"].isna()).astype(int),
})
months = df["Date"].dt.to_period("M").unique()
for i, col in enumerate(["Activity", "Vitals", "Sleep"]):
    dates_present = df.loc[~df["Steps" if col=="Activity"
                               else "Avg. Heart Rate(bpm)" if col=="Vitals"
                               else "Sleep Time Ratio(%)"].isna(), "Date"]
    ax_cal.scatter(dates_present, [i]*len(dates_present),
                   c=PALETTE[i], s=18, marker="s", alpha=0.85, label=col)

ax_cal.set_yticks([0, 1, 2])
ax_cal.set_yticklabels(["Activity", "Vitals", "Sleep"], fontsize=9)
ax_cal.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax_cal.xaxis.set_major_locator(mdates.MonthLocator())
ax_cal.set_title("A   Data Availability by Source and Date", loc="left")
ax_cal.set_xlim(df["Date"].min() - pd.Timedelta(days=2),
                df["Date"].max() + pd.Timedelta(days=2))
ax_cal.tick_params(axis="x", rotation=30)

# ── Panel B: Data completeness donut ─────────────────────────────────────────
ax_don = fig.add_subplot(gs[0, 3])
total  = len(df)
n_act  = df["Steps"].notna().sum()
n_vit  = df["Avg. Heart Rate(bpm)"].notna().sum()
n_slp  = df["Sleep Time Ratio(%)"].notna().sum()
wedges, texts, autotexts = ax_don.pie(
    [n_act, n_vit, n_slp],
    labels=["Activity", "Vitals", "Sleep"],
    colors=PALETTE[:3], autopct="%1.0f%%",
    wedgeprops=dict(width=0.55), startangle=90,
    textprops=dict(fontsize=8)
)
ax_don.set_title("B   Completeness\n(% of 185 days)", loc="center", fontsize=10)

# ── Panel C-E: Distributions of 3 key variables ───────────────────────────────
plot_vars = [
    ("Steps",             C["steps"], "Daily Steps"),
    ("Avg. HRV(ms)",      C["hrv"],   "Avg. HRV (ms)"),
    ("Sleep Time Ratio(%)", C["sleep"], "Sleep Quality (%)"),
]
for j, (col, color, label) in enumerate(plot_vars):
    ax = fig.add_subplot(gs[1, j])
    data = df[col].dropna()
    ax.hist(data, bins=22, color=color, alpha=0.75, edgecolor="white", density=True)
    xmin, xmax = data.min(), data.max()
    x  = np.linspace(xmin, xmax, 200)
    mu, sigma = data.mean(), data.std()
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="black", lw=1.5,
            linestyle="--", label="Normal PDF")
    ax.axvline(mu, color=color, lw=1.2, alpha=0.8, label=f"Mean = {mu:.1f}")
    ax.set_title(f"C{j+1}   {label}", loc="left")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    stat, p = shapiro(data[:50])
    ax.text(0.97, 0.93, f"Shapiro-Wilk\np = {p:.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=C["lightbg"], alpha=0.8))

# ── Panel F: Class balance ─────────────────────────────────────────────────────
ax_bar = fig.add_subplot(gs[1, 3])
good_n = slp_clean["quality_label"].sum()
poor_n = (slp_clean["quality_label"] == 0).sum()
bars = ax_bar.bar(["Poor (<80%)", "Good (≥80%)"], [poor_n, good_n],
                  color=[C["poor"], C["good"]], edgecolor="white", linewidth=0.8)
for bar, n in zip(bars, [poor_n, good_n]):
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(n), ha="center", va="bottom", fontweight="bold")
ax_bar.set_title("F   Sleep Quality Balance", loc="left")
ax_bar.set_ylabel("Nights")
ax_bar.set_ylim(0, max(good_n, poor_n) * 1.2)

# ── Panel G: Descriptive stats table ──────────────────────────────────────────
ax_tbl = fig.add_subplot(gs[2, :])
ax_tbl.axis("off")
key_vars = ["Steps", "Calories(kcal)", "Avg. Heart Rate(bpm)",
            "Avg. HRV(ms)", "Avg. Spo2(%)", "Sleep Time Ratio(%)",
            "Time Asleep(min)", "efficiency"]
key_vars = [v for v in key_vars if v in df.columns]
rows, col_labels = [], ["Variable", "N", "Mean ± SD", "Median", "IQR", "Min", "Max"]
for v in key_vars:
    d = df[v].dropna()
    q25, q75 = d.quantile(0.25), d.quantile(0.75)
    rows.append([v, str(len(d)),
                 f"{d.mean():.1f} ± {d.std():.1f}",
                 f"{d.median():.1f}",
                 f"[{q25:.1f}–{q75:.1f}]",
                 f"{d.min():.1f}", f"{d.max():.1f}"])
tbl = ax_tbl.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center",
                   bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(C["navy"])
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#EBF5FB")
    cell.set_edgecolor("#DDDFE2")
ax_tbl.set_title("G   Descriptive Statistics Summary", loc="left", fontsize=11, fontweight="bold", pad=4)

fig.suptitle("Figure 1 · Dataset Overview — Cardiovascular & Sleep Wearable Data (Nov 2025 – May 2026)",
             fontsize=13, fontweight="bold", y=1.01)
plt.savefig(os.path.join(FIG_DIR, "fig1_dataset_overview.png"))
plt.close()
print("  ✓  fig1_dataset_overview.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §6  MISSING VALUES & IMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════
section("§6  MISSING VALUE ANALYSIS & IMPUTATION")

miss_count = df.isnull().sum()
miss_pct   = df.isnull().mean() * 100
# Keep ALL features so the bar chart is never empty; sort by count descending
miss_df    = pd.DataFrame({"count": miss_count, "pct": miss_pct}).sort_values("count", ascending=True)
n_with_missing = (miss_count > 0).sum()
subsection(f"Features with missing data: {n_with_missing} / {len(miss_count)}")

fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(miss_df) * 0.35 + 1)),
                         gridspec_kw={"width_ratios": [1.2, 1]})

# Panel A: Missing COUNT per feature (shows 0 bars clearly for complete features)
colors_miss = [C["red"] if c > miss_count.max() * 0.5
               else C["orange"] if c > 0
               else C["teal"]
               for c in miss_df["count"].values]
axes[0].barh(range(len(miss_df)), miss_df["count"].values, color=colors_miss,
             edgecolor="white", height=0.7)
axes[0].set_yticks(range(len(miss_df)))
axes[0].set_yticklabels([textwrap.shorten(n, 28) for n in miss_df.index], fontsize=8)
axes[0].set_xlabel("Missing count (days)")
axes[0].set_xlim(left=0)                           # always start at 0
axes[0].set_title("A   Missing Count per Feature", loc="left")
for i, v in enumerate(miss_df["count"].values):
    if v > 0:
        axes[0].text(v + 0.2, i, str(int(v)), va="center", fontsize=7)

# Panel B: Missing PERCENTAGE with 5% threshold line
axes[1].barh(range(len(miss_df)), miss_df["pct"].values, color=colors_miss,
             edgecolor="white", height=0.7)
axes[1].set_yticks(range(len(miss_df)))
axes[1].set_yticklabels([textwrap.shorten(n, 28) for n in miss_df.index], fontsize=8)
axes[1].axvline(5, color=C["red"], lw=1.5, ls="--", alpha=0.8, label="5% threshold")
axes[1].set_xlabel("Missing (%)")
axes[1].set_xlim(left=0)
axes[1].set_title("B   Missing Percentage per Feature", loc="left")
axes[1].legend(fontsize=8)
for i, v in enumerate(miss_df["pct"].values):
    if v > 0:
        axes[1].text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=7)

fig.suptitle("Figure 2 · Missing Value Analysis (pre-imputation)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_missing_data.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  ✓  fig2_missing_data.png saved")

# ── Imputation ────────────────────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
sleep_cols   = [c for c in numeric_cols if any(k in c for k in
                ["Sleep", "sleep", "rem", "deep", "light", "waso", "onset", "quality", "efficiency", "total_min"])]
df[sleep_cols] = df[sleep_cols].fillna(method="ffill", limit=2)
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
print(f"  Remaining nulls after KNN imputation: {df[numeric_cols].isnull().sum().sum()}")


# ═══════════════════════════════════════════════════════════════════════════════
# §7  OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
section("§7  OUTLIER DETECTION (IQR + Z-score + Grubbs)")

key_features = ["Steps", "Calories(kcal)", "Avg. Heart Rate(bpm)",
                "Min. Heart Rate(bpm)", "Avg. HRV(ms)", "Min. HRV(ms)",
                "Avg. Spo2(%)", "Sleep Time Ratio(%)", "Time Asleep(min)",
                "onset_latency", "waso", "efficiency"]
key_features = [f for f in key_features if f in df.columns]

# Winsorize at 1st / 99th before plotting
df_raw = df[key_features].copy()
for col in key_features:
    p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=p1, upper=p99)

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i, col in enumerate(key_features):
    ax = axes[i]
    data_raw   = df_raw[col].dropna()
    data_clean = df[col].dropna()
    # Violin
    parts = ax.violinplot([data_raw, data_clean], positions=[1, 2],
                          showmedians=True, showextrema=True)
    for pc, c in zip(parts["bodies"], [C["orange"], C["teal"]]):
        pc.set_facecolor(c); pc.set_alpha(0.65)
    parts["cmedians"].set_color("black")
    parts["cmaxes"].set_color("gray")
    parts["cmins"].set_color("gray")
    # IQR fences
    Q1, Q3 = data_raw.quantile(0.25), data_raw.quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((data_raw < Q1-1.5*IQR) | (data_raw > Q3+1.5*IQR)).sum()
    ax.set_xticks([1, 2]); ax.set_xticklabels(["Raw", "Winsorized"], fontsize=8)
    ax.set_title(textwrap.shorten(col, 22), fontsize=8, fontweight="bold")
    ax.text(0.97, 0.97, f"IQR outliers\nn = {n_out}", transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD", alpha=0.9))

for j in range(len(key_features), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Figure 3 · Outlier Detection: Raw vs Winsorized (Violin Plots)",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig3_outlier_detection.png"))
plt.close()
print("  ✓  fig3_outlier_detection.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §8  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
section("§8  FEATURE ENGINEERING")

df = df.sort_values("Date").reset_index(drop=True)

lag_sources = ["Steps", "Calories(kcal)", "Avg. HRV(ms)", "Min. HRV(ms)",
               "Avg. Heart Rate(bpm)", "Avg. Spo2(%)", "Min. Spo2(%)"]
lag_sources = [c for c in lag_sources if c in df.columns]

for col in lag_sources:
    for lag in [1, 2, 3]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

for col in ["Steps", "Avg. HRV(ms)", "Avg. Heart Rate(bpm)"]:
    if col in df.columns:
        df[f"{col}_roll3"]   = df[col].rolling(3, min_periods=1).mean()
        df[f"{col}_roll7"]   = df[col].rolling(7, min_periods=1).mean()
        df[f"{col}_roll7sd"] = df[col].rolling(7, min_periods=1).std()

if "Avg. HRV(ms)" in df.columns:
    hrv_q25 = df["Avg. HRV(ms)"].quantile(0.25)
    df["hrv_range"]    = df["Max. HRV(ms)"] - df["Min. HRV(ms)"]
    df["hrv_low_flag"] = (df["Avg. HRV(ms)"] < hrv_q25).astype(int)

if "Steps" in df.columns:
    df["activity_low"]  = (df["Steps"] < 5000).astype(int)
    df["activity_high"] = (df["Steps"] > 10000).astype(int)

df["dow"]        = df["Date"].dt.dayofweek
df["dow_sin"]    = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"]    = np.cos(2 * np.pi * df["dow"] / 7)
df["is_weekend"] = (df["dow"] >= 5).astype(int)
df["month_sin"]  = np.sin(2 * np.pi * df["Date"].dt.month / 12)
df["month_cos"]  = np.cos(2 * np.pi * df["Date"].dt.month / 12)

# Targets
df["next_sleep_ratio"]   = df["Sleep Time Ratio(%)"].shift(-1)
df["next_sleep_quality"] = (df["next_sleep_ratio"] >= 80).astype("Int64")

subsection(f"Final feature count: {df.shape[1]} columns  |  {df.shape[0]} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# §9  FIGURE 4 – TIME-SERIES DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
section("§9  FIGURE 4 – TIME-SERIES DASHBOARD")

ts_config = [
    ("Steps",               C["steps"],  "Daily Steps"),
    ("Avg. HRV(ms)",        C["hrv"],    "Avg. HRV (ms)"),
    ("Avg. Heart Rate(bpm)", C["hr"],    "Avg. Heart Rate (bpm)"),
    ("Avg. Spo2(%)",        C["spo2"],   "Avg. SpO₂ (%)"),
    ("Sleep Time Ratio(%)", C["sleep"],  "Sleep Quality (%)"),
    ("Time Asleep(min)",    C["purple"], "Time Asleep (min)"),
]
ts_config = [(c, col, lbl) for c, col, lbl in ts_config if c in df.columns]

fig, axes = plt.subplots(len(ts_config), 1, figsize=(16, 2.8*len(ts_config)),
                          sharex=True, gridspec_kw={"hspace": 0.35})

for ax, (col, color, label) in zip(axes, ts_config):
    y = df[col].values
    x = df["Date"]
    ax.fill_between(x, y, alpha=0.18, color=color)
    ax.plot(x, y, color=color, lw=1.3, alpha=0.85)
    roll = df[col].rolling(7, min_periods=1).mean()
    ax.plot(x, roll, color="black", lw=1.6, ls="--", alpha=0.7, label="7-day MA")
    ax.set_ylabel(label, fontsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.legend(loc="upper right", fontsize=8)
    # Shade weekends
    for _, row in df[df["is_weekend"] == 1].iterrows():
        ax.axvspan(row["Date"], row["Date"] + pd.Timedelta(days=1),
                   alpha=0.04, color=C["purple"])

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))
axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
plt.xticks(rotation=30, ha="right")

fig.suptitle("Figure 4 · Multivariate Time-Series Dashboard (shaded = weekends, dashed = 7-day MA)",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig4_timeseries_dashboard.png"))
plt.close()
print("  ✓  fig4_timeseries_dashboard.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §10  FIGURE 5 – DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
section("§10  FIGURE 5 – DISTRIBUTION ANALYSIS")

dist_vars = ["Steps", "Calories(kcal)", "Avg. HRV(ms)", "Min. HRV(ms)",
             "Avg. Heart Rate(bpm)", "Avg. Spo2(%)",
             "Sleep Time Ratio(%)", "Time Asleep(min)",
             "efficiency", "onset_latency", "waso", "rem_ratio"]
dist_vars = [v for v in dist_vars if v in df.columns]

fig, axes = plt.subplots(3, 4, figsize=(16, 11))
axes = axes.flatten()

for i, col in enumerate(dist_vars):
    if i >= len(axes): break
    ax = axes[i]
    data = df[col].dropna()
    color = PALETTE[i % len(PALETTE)]

    # KDE + rug
    try:
        kde_x = np.linspace(data.min(), data.max(), 300)
        kde   = stats.gaussian_kde(data)
        ax.fill_between(kde_x, kde(kde_x), alpha=0.35, color=color)
        ax.plot(kde_x, kde(kde_x), color=color, lw=2)
        ax.plot(kde_x, stats.norm.pdf(kde_x, data.mean(), data.std()),
                color="black", lw=1.5, ls="--", alpha=0.6, label="Normal")
    except Exception:
        ax.hist(data, bins=20, density=True, color=color, alpha=0.5)

    ax.axvline(data.mean(),   color=color, lw=1.5, label=f"μ={data.mean():.1f}")
    ax.axvline(data.median(), color="black", lw=1.2, ls=":", label=f"Med={data.median():.1f}")

    stat, p = shapiro(data[:50] if len(data) > 50 else data)
    sk, ku  = skew(data), kurtosis(data)
    info_txt = f"Skew={sk:.2f}\nKurt={ku:.2f}\nSW p={p:.3f}"
    ax.text(0.97, 0.95, info_txt, transform=ax.transAxes,
            ha="right", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_title(textwrap.shorten(col, 24), fontsize=9, fontweight="bold")
    ax.set_ylabel("Density", fontsize=8)
    ax.legend(fontsize=7)

for j in range(len(dist_vars), len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Figure 5 · Distribution Analysis — KDE + Normal Overlay + Shapiro-Wilk",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig5_distribution_analysis.png"))
plt.close()
print("  ✓  fig5_distribution_analysis.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §11  FIGURE 6 – CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
section("§11  FIGURE 6 – CORRELATION MATRIX")

corr_vars = [
    "Steps", "Calories(kcal)", "Avg. Heart Rate(bpm)", "Min. Heart Rate(bpm)",
    "Avg. HRV(ms)", "Min. HRV(ms)", "hrv_range",
    "Avg. Spo2(%)", "Min. Spo2(%)",
    "Sleep Time Ratio(%)", "Time Asleep(min)", "efficiency",
    "waso", "onset_latency", "rem_ratio", "deep_ratio"
]
corr_vars = [v for v in corr_vars if v in df.columns]
corr_df   = df[corr_vars].dropna()

sp_corr = corr_df.corr(method="spearman")
mask    = np.triu(np.ones_like(sp_corr, dtype=bool))

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Spearman heatmap
sns.heatmap(sp_corr, mask=mask, annot=True, fmt=".2f",
            cmap=sns.diverging_palette(230, 20, as_cmap=True),
            vmin=-1, vmax=1, ax=axes[0], linewidths=0.3,
            annot_kws={"size": 6.5}, square=True,
            cbar_kws={"shrink": 0.8, "label": "Spearman ρ"})
axes[0].set_title("A   Spearman Rank Correlation Matrix", loc="left", fontsize=11, fontweight="bold")
axes[0].tick_params(axis="x", rotation=45, labelsize=8)
axes[0].tick_params(axis="y", labelsize=8)

# Feature–target correlation bar
target_corr = sp_corr["Sleep Time Ratio(%)"].drop("Sleep Time Ratio(%)").sort_values()
bar_colors  = [C["teal"] if v > 0 else C["red"] for v in target_corr.values]
bars = axes[1].barh(range(len(target_corr)), target_corr.values,
                    color=bar_colors, edgecolor="white", height=0.7)
axes[1].set_yticks(range(len(target_corr)))
axes[1].set_yticklabels([textwrap.shorten(n, 28) for n in target_corr.index], fontsize=8)
axes[1].axvline(0, color="black", lw=0.8)
axes[1].axvline( 0.3, color=C["teal"], lw=1, ls="--", alpha=0.5)
axes[1].axvline(-0.3, color=C["red"],  lw=1, ls="--", alpha=0.5)
axes[1].set_xlabel("Spearman ρ")
axes[1].set_title("B   Correlation with Sleep Quality (Sleep Time Ratio %)",
                  loc="left", fontsize=11, fontweight="bold")
# Annotate values
for bar, v in zip(bars, target_corr.values):
    axes[1].text(v + (0.01 if v >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                 f"{v:.2f}", va="center", ha="left" if v >= 0 else "right",
                 fontsize=7.5, color="black")

legend_patches = [mpatches.Patch(color=C["teal"], label="Positive correlation"),
                  mpatches.Patch(color=C["red"],  label="Negative correlation")]
axes[1].legend(handles=legend_patches, fontsize=8)

fig.suptitle("Figure 6 · Correlation Analysis", fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig6_correlation_matrix.png"))
plt.close()
print("  ✓  fig6_correlation_matrix.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §12  FIGURE 7 – SLEEP STAGE COMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════
section("§12  FIGURE 7 – SLEEP STAGE COMPOSITION")

slp_stage_vars = [
    "Sleep Stages - Deep Sleep(min)",
    "Sleep Stages - REM(min)",
    "Sleep Stages - Light Sleep(min)",
    "Sleep Stages - Awake(min)"
]
slp_stage_vars = [v for v in slp_stage_vars if v in df.columns]
stage_colors   = [C["navy"], C["teal"], C["blue"], C["red"]]
stage_labels   = ["Deep", "REM", "Light", "Awake"]

slp_df = df[["Date", "quality_label"] + slp_stage_vars].dropna().copy()
slp_df_sorted = slp_df.sort_values("Date").reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: Stacked bar over time
bottom = np.zeros(len(slp_df_sorted))
for col, color, label in zip(slp_stage_vars, stage_colors, stage_labels):
    vals = slp_df_sorted[col].values
    axes[0].bar(range(len(slp_df_sorted)), vals, bottom=bottom,
                color=color, label=label, width=0.85)
    bottom += vals
axes[0].set_xlabel("Sleep Session (chronological)")
axes[0].set_ylabel("Duration (min)")
axes[0].set_title("A   Sleep Stage Composition Over Time", loc="left", fontweight="bold")
axes[0].legend(loc="upper right")
axes[0].axhline(420, color="gray", ls="--", lw=1, alpha=0.5, label="7h reference")

# Panel B: Boxplot by quality label
stage_ratios = ["deep_ratio", "rem_ratio", "light_ratio"]
stage_ratios = [v for v in stage_ratios if v in df.columns]
ratio_labels = ["Deep", "REM", "Light"]
positions    = np.arange(len(stage_ratios))
bp_data_good = [df[df["quality_label"] >= 0.5][v].dropna() for v in stage_ratios]
bp_data_poor = [df[df["quality_label"] <  0.5][v].dropna() for v in stage_ratios]

bp_g = axes[1].boxplot(bp_data_good, positions=positions - 0.22, widths=0.38,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=C["good"], alpha=0.65),
                        medianprops=dict(color="black", lw=2))
bp_p = axes[1].boxplot(bp_data_poor, positions=positions + 0.22, widths=0.38,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=C["poor"], alpha=0.65),
                        medianprops=dict(color="black", lw=2))
axes[1].set_xticks(positions)
axes[1].set_xticklabels(ratio_labels)
axes[1].set_ylabel("Ratio (% of total sleep)")
axes[1].set_title("B   Sleep Stage Ratios: Good vs Poor Nights", loc="left", fontweight="bold")
good_patch = mpatches.Patch(color=C["good"], alpha=0.65, label="Good (≥80%)")
poor_patch = mpatches.Patch(color=C["poor"], alpha=0.65, label="Poor (<80%)")
axes[1].legend(handles=[good_patch, poor_patch])

# Significance stars
for i, v in enumerate(stage_ratios):
    good_d = df[df["quality_label"] >= 0.5][v].dropna()
    poor_d = df[df["quality_label"] <  0.5][v].dropna()
    if len(good_d) > 3 and len(poor_d) > 3:
        _, p = mannwhitneyu(good_d, poor_d, alternative="two-sided")
        star = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
        y_max = max(good_d.max(), poor_d.max())
        axes[1].text(i, y_max + 2, star, ha="center", fontsize=10, color="black")

fig.suptitle("Figure 7 · Sleep Stage Analysis", fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig7_sleep_stage_composition.png"))
plt.close()
print("  ✓  fig7_sleep_stage_composition.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §13  FIGURE 8 – HRV vs NEXT-NIGHT SLEEP (Research Question 3)
# ═══════════════════════════════════════════════════════════════════════════════
section("§13  FIGURE 8 – HRV → SLEEP (RQ3)")

hrv_df = df[["Date", "Avg. HRV(ms)", "hrv_low_flag",
             "Steps", "next_sleep_ratio", "next_sleep_quality",
             "waso"]].dropna(subset=["Avg. HRV(ms)", "next_sleep_ratio"]).copy()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel A: Scatter HRV vs next-night sleep
quality_colors = hrv_df["next_sleep_quality"].map({0: C["poor"], 1: C["good"]})
sc = axes[0].scatter(hrv_df["Avg. HRV(ms)"], hrv_df["next_sleep_ratio"],
                     c=quality_colors, s=55, alpha=0.75, edgecolors="#FFFFFF",
                     linewidths=0.5, zorder=3)
m, b = np.polyfit(hrv_df["Avg. HRV(ms)"], hrv_df["next_sleep_ratio"], 1)
x_line = np.linspace(hrv_df["Avg. HRV(ms)"].min(), hrv_df["Avg. HRV(ms)"].max(), 200)
axes[0].plot(x_line, m*x_line + b, color="black", lw=1.8, ls="--",
             label=f"OLS fit (slope={m:.2f})", zorder=4)
r, p = spearmanr(hrv_df["Avg. HRV(ms)"], hrv_df["next_sleep_ratio"])
axes[0].text(0.04, 0.97, f"Spearman ρ = {r:.3f}\np = {p:.4f}",
             transform=axes[0].transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.4", facecolor=C["lightbg"]))
axes[0].set_xlabel("Daytime Avg. HRV (ms)")
axes[0].set_ylabel("Next-Night Sleep Quality (%)")
axes[0].set_title("A   HRV (Day D) → Sleep Quality (Night D/D+1)", loc="left", fontweight="bold")
handles = [mpatches.Patch(color=C["good"], label="Good (≥80%)"),
           mpatches.Patch(color=C["poor"], label="Poor (<80%)"),
           plt.Line2D([0],[0], color="black", ls="--", label=f"OLS slope={m:.2f}")]
axes[0].legend(handles=handles, fontsize=8)

# Panel B: Violin — HRV Low vs Normal → sleep
low_hrv_sleep  = hrv_df[hrv_df["hrv_low_flag"]==1]["next_sleep_ratio"]
high_hrv_sleep = hrv_df[hrv_df["hrv_low_flag"]==0]["next_sleep_ratio"]
vp = axes[1].violinplot([high_hrv_sleep, low_hrv_sleep], positions=[1, 2],
                         showmedians=True, showextrema=True)
for pc, c in zip(vp["bodies"], [C["good"], C["poor"]]):
    pc.set_facecolor(c); pc.set_alpha(0.65)
vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
u_stat, p_val = mannwhitneyu(high_hrv_sleep, low_hrv_sleep, alternative="greater")
axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(["Normal HRV", "Low HRV\n(<25th pct)"])
axes[1].set_ylabel("Next-Night Sleep Ratio (%)")
axes[1].set_title("B   Sleep Quality: Normal vs Low-HRV Days", loc="left", fontweight="bold")
star = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else "ns"
y_top = max(high_hrv_sleep.max(), low_hrv_sleep.max())
axes[1].annotate("", xy=(2, y_top+3), xytext=(1, y_top+3),
                 arrowprops=dict(arrowstyle="-", color="black"))
axes[1].text(1.5, y_top+5, f"MWU {star}\np={p_val:.4f}", ha="center", fontsize=9)
axes[1].text(1, high_hrv_sleep.median(), f"Mdn={high_hrv_sleep.median():.1f}",
             ha="center", fontsize=8, color=C["good"])
axes[1].text(2, low_hrv_sleep.median(),  f"Mdn={low_hrv_sleep.median():.1f}",
             ha="center", fontsize=8, color=C["poor"])

# Panel C: HRV lag correlation
hrv_lag_cols = [c for c in df.columns if "HRV" in c and "lag" in c and "Avg" in c]
if hrv_lag_cols:
    lag_rs = []
    for lc in sorted(hrv_lag_cols):
        paired = df[[lc, "next_sleep_ratio"]].dropna()
        r_val, p_val_lag = spearmanr(paired[lc], paired["next_sleep_ratio"])
        lag_rs.append((lc.split("lag")[-1], r_val, p_val_lag))
    lags_n   = [x[0] for x in lag_rs]
    lag_rhos = [x[1] for x in lag_rs]
    lag_ps   = [x[2] for x in lag_rs]
    bar_c = [C["teal"] if p<0.05 else C["gray"] for p in lag_ps]
    axes[2].bar(lags_n, lag_rhos, color=bar_c, edgecolor="white")
    axes[2].axhline(0, color="black", lw=0.8)
    for j, (r_v, p_v) in enumerate(zip(lag_rhos, lag_ps)):
        axes[2].text(j, r_v + 0.005 if r_v >= 0 else r_v - 0.015,
                     f"ρ={r_v:.3f}", ha="center", fontsize=8)
    axes[2].set_xlabel("Lag (days before sleep)")
    axes[2].set_ylabel("Spearman ρ with Next-Night Sleep %")
    axes[2].set_title("C   HRV Lag Correlation (green = p<0.05)", loc="left", fontweight="bold")
    axes[2].set_xticks(range(len(lags_n)))
    axes[2].set_xticklabels([f"Lag {n}" for n in lags_n])

fig.suptitle("Figure 8 · HRV as Predictor of Next-Night Sleep Quality (Research Question 3)",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig8_hrv_sleep_scatter.png"))
plt.close()
print("  ✓  fig8_hrv_sleep_scatter.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §14  FIGURE 9 – GOOD vs POOR SLEEP NIGHTS (multi-feature)
# ═══════════════════════════════════════════════════════════════════════════════
section("§14  FIGURE 9 – GOOD vs POOR SLEEP COMPARISON")

compare_vars = [
    ("Steps",                 "Steps (day before)"),
    ("Avg. HRV(ms)",          "Avg. HRV (ms)"),
    ("Avg. Heart Rate(bpm)",  "Avg. Heart Rate (bpm)"),
    ("Avg. Spo2(%)",          "Avg. SpO₂ (%)"),
    ("onset_latency",         "Sleep Onset Latency (min)"),
    ("waso",                  "Wake After Sleep Onset (min)"),
    ("efficiency",            "Sleep Efficiency (%)"),
    ("rem_ratio",             "REM Ratio (%)"),
]
compare_vars = [(c, l) for c, l in compare_vars if c in df.columns]

good_df = df[df["quality_label"] >= 0.5]
poor_df = df[df["quality_label"] <  0.5]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, (col, label) in enumerate(compare_vars):
    ax    = axes[i]
    gdata = good_df[col].dropna()
    pdata = poor_df[col].dropna()

    parts = ax.violinplot([gdata, pdata], positions=[1, 2],
                          showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], [C["good"], C["poor"]]):
        pc.set_facecolor(c); pc.set_alpha(0.6)
    parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2.5)

    # Individual points (jittered)
    for j_pos, data, c in [(1, gdata, C["good"]), (2, pdata, C["poor"])]:
        jitter = np.random.uniform(-0.08, 0.08, len(data))
        ax.scatter(j_pos + jitter, data, s=10, alpha=0.4, color=c, zorder=3)

    stat, p = mannwhitneyu(gdata, pdata, alternative="two-sided")
    star = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
    y_max = max(gdata.max(), pdata.max()) * 1.05
    ax.annotate("", xy=(2, y_max), xytext=(1, y_max),
                arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax.text(1.5, y_max * 1.01, f"{star}\np={p:.3f}", ha="center", fontsize=8)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Good", "Poor"], fontsize=9)
    ax.set_title(textwrap.shorten(label, 26), fontsize=9, fontweight="bold")
    ax.set_ylabel("Value", fontsize=8)

fig.suptitle("Figure 9 · Feature Distribution: Good vs Poor Sleep Nights (Mann-Whitney U)",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig9_good_vs_poor_sleep.png"))
plt.close()
print("  ✓  fig9_good_vs_poor_sleep.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §15  FIGURE 10 – STATIONARITY & AUTOCORRELATION
# ═══════════════════════════════════════════════════════════════════════════════
section("§15  FIGURE 10 – STATIONARITY & ACF/PACF")

adf_vars   = ["Steps", "Avg. HRV(ms)", "Avg. Heart Rate(bpm)", "Sleep Time Ratio(%)"]
adf_vars   = [v for v in adf_vars if v in df.columns]

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(3, len(adf_vars), figure=fig, hspace=0.5, wspace=0.4)

for j, col in enumerate(adf_vars):
    series = df[col].dropna()
    adf_r  = adfuller(series, autolag="AIC")
    stat, p_adf = adf_r[0], adf_r[1]
    status      = "Stationary" if p_adf < 0.05 else "Non-stationary"
    color       = C["green"] if p_adf < 0.05 else C["red"]

    # Row 0: time series
    ax0 = fig.add_subplot(gs[0, j])
    ax0.plot(df["Date"].iloc[:len(series)], series.values, lw=1.2, color=PALETTE[j])
    ax0.set_title(f"{textwrap.shorten(col,18)}\nADF p={p_adf:.4f} [{status}]",
                  fontsize=8, fontweight="bold", color=color)
    ax0.tick_params(axis="x", rotation=30, labelsize=6)

    # Row 1: ACF
    ax1 = fig.add_subplot(gs[1, j])
    plot_acf(series, lags=20, ax=ax1, alpha=0.05, color=PALETTE[j])
    ax1.set_title(f"ACF — {textwrap.shorten(col, 15)}", fontsize=8)
    ax1.set_xlabel("Lag"); ax1.set_ylabel("Autocorr.")

    # Row 2: PACF
    ax2 = fig.add_subplot(gs[2, j])
    plot_pacf(series, lags=20, ax=ax2, alpha=0.05, color=PALETTE[j], method="ywm")
    ax2.set_title(f"PACF — {textwrap.shorten(col, 15)}", fontsize=8)
    ax2.set_xlabel("Lag"); ax2.set_ylabel("Part. Autocorr.")

fig.suptitle("Figure 10 · Stationarity (ADF) & Autocorrelation Analysis",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig10_stationarity_acf_pacf.png"))
plt.close()
print("  ✓  fig10_stationarity_acf_pacf.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §16  SAVE PROCESSED DATASET
# ═══════════════════════════════════════════════════════════════════════════════
section("§16  SAVE PROCESSED DATASET")

df.drop(columns=["dow"], inplace=True, errors="ignore")
df_model = df.dropna(subset=["next_sleep_ratio", "next_sleep_quality"]).copy()
out_csv  = os.path.join(DATA_DIR, "processed_dataset.csv")
df_model.to_csv(out_csv, index=False)

print(f"  Saved: {out_csv}")
print(f"  Shape: {df_model.shape}")
print(f"  Target balance → Good: {(df_model['next_sleep_quality']==1).sum()}  "
      f"Poor: {(df_model['next_sleep_quality']==0).sum()}")

print("\n╔" + "═"*68 + "╗")
print("║  CODE 1 COMPLETE — all figures saved to:                           ║")
print(f"║  {FIG_DIR[:62]:<62} ║")
print("╚" + "═"*68 + "╝")
