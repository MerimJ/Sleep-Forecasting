"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MLPR-CDA · CODE 2 · MODELING, VALIDATION & EXPLAINABILITY                 ║
║  Project : Forecasting Cardiovascular Health & Sleep Quality                 ║
║            from Wearable Time-Series                                         ║
║  Author  : Merim Jusufbegovic                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run AFTER code1_eda.py (requires processed_dataset.csv)

USAGE
─────
  python code2_modeling.py

MODELS
──────
  Classical ML : Random Forest · XGBoost
  Deep Learning: LSTM · CNN-LSTM · Transformer (encoder)
  XAI          : SHAP (TreeExplainer for RF & XGBoost)

OUTPUT → ~/Downloads/MLPR-CDA/outputs/figures/
  fig11_model_comparison.png       fig16_shap_dependence.png
  fig12_roc_curves.png             fig17_confusion_matrices.png
  fig13_ts_predictions.png         fig18_feature_importance.png
  fig14_cv_results.png             fig19_learning_curves.png
  fig15_shap_summary.png           fig20_calibration_curves.png
  results_regression.csv
  results_classification.csv
"""

# ═══════════════════════════════════════════════════════════════════════════════
# §0  IMPORTS & STYLE
# ═══════════════════════════════════════════════════════════════════════════════
import os, warnings, random, textwrap
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (TimeSeriesSplit, cross_val_score,
                                     learning_curve)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, brier_score_loss)
try:
    from sklearn.metrics import calibration_curve
except ImportError:
    from sklearn.calibration import calibration_curve
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import shap
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, Dropout, Conv1D,
                                      MultiHeadAttention, LayerNormalization,
                                      GlobalAveragePooling1D, MaxPooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from scipy.stats import spearmanr, mannwhitneyu

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ─── Style ───────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.facecolor": "#FAFBFD", "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": "#E5E5E5", "grid.linestyle": "--", "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.labelcolor": "#2C3E50",
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.framealpha": 0.9,
})

C = {
    "navy": "#1B2A4A", "teal": "#0B7A75", "blue": "#2471A3",
    "orange": "#E67E22", "red": "#C0392B", "green": "#1E8449",
    "purple": "#7D3C98", "gray": "#808B96", "lightbg": "#EBF5FB",
    "rf": "#2471A3", "xgb": "#E67E22", "lstm": "#0B7A75",
    "cnnlstm": "#7D3C98", "transformer": "#C0392B",
}
MODEL_COLORS = {
    "Random Forest": C["rf"], "XGBoost": C["xgb"],
    "LSTM": C["lstm"], "CNN-LSTM": C["cnnlstm"],
    "Transformer": C["transformer"],
}

SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

def section(t):
    print(f"\n{'═'*70}\n  {t}\n{'═'*70}")

print("╔" + "═"*68 + "╗")
print("║  MLPR-CDA · CODE 2 · MODELING & VALIDATION" + " "*25 + "║")
print("╚" + "═"*68 + "╝")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  PATHS & DATA
# ═══════════════════════════════════════════════════════════════════════════════
section("§1  LOADING DATA")

DATA_DIR = os.path.expanduser("~/Downloads/MLPR-CDA")
FIG_DIR  = os.path.join(DATA_DIR, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(DATA_DIR, "processed_dataset.csv"),
                 parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
print(f"  Loaded: {df.shape}  |  {df['Date'].min().date()} → {df['Date'].max().date()}")


# ═══════════════════════════════════════════════════════════════════════════════
# §2  FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
section("§2  FEATURE SELECTION")

EXCLUDE = {
    "Date",
    # Raw sleep columns (would leak the target)
    "Sleep Time Ratio(%)", "Time Asleep(min)", "total_min",
    "efficiency", "waso", "onset_latency",
    "Sleep Stages - Awake(min)", "Sleep Stages - REM(min)",
    "Sleep Stages - Light Sleep(min)", "Sleep Stages - Deep Sleep(min)",
    "rem_ratio", "deep_ratio", "light_ratio",
    "quality_label",
    # Targets
    "next_sleep_ratio", "next_sleep_quality",
    # Redundant cyclical raw
    "month",
}
FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE
                and df[c].dtype != "object"]

TARGET_REG = "next_sleep_ratio"
TARGET_CLF = "next_sleep_quality"

df_m = df[FEATURE_COLS + [TARGET_REG, TARGET_CLF]].dropna().copy()
X    = df_m[FEATURE_COLS].values.astype(np.float32)
y_r  = df_m[TARGET_REG].values.astype(np.float32)
y_c  = df_m[TARGET_CLF].values.astype(int)

print(f"  Features : {len(FEATURE_COLS)}")
print(f"  Samples  : {len(df_m)}  |  Good: {(y_c==1).sum()}  Poor: {(y_c==0).sum()}")


# ═══════════════════════════════════════════════════════════════════════════════
# §3  TRAIN / TEST SPLIT + SCALING
# ═══════════════════════════════════════════════════════════════════════════════
section("§3  TIME-SERIES TRAIN/TEST SPLIT (80/20 chronological)")

split = int(len(df_m) * 0.80)
X_tr, X_te = X[:split], X[split:]
yr_tr, yr_te = y_r[:split], y_r[split:]
yc_tr, yc_te = y_c[:split], y_c[split:]

scaler   = RobustScaler()
X_tr_s   = scaler.fit_transform(X_tr)
X_te_s   = scaler.transform(X_te)

tscv = TimeSeriesSplit(n_splits=5)
print(f"  Train: {X_tr_s.shape}   Test: {X_te_s.shape}")

# Sequence builder for deep learning
WINDOW = 7

def make_seq(X, y, w=WINDOW):
    Xs, ys = [], []
    for i in range(w, len(X)):
        Xs.append(X[i-w:i]); ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X_full_s = np.vstack([X_tr_s, X_te_s])
yr_full  = np.concatenate([yr_tr, yr_te])
yc_full  = np.concatenate([yc_tr, yc_te])

Xs_r, ys_r = make_seq(X_full_s, yr_full)
Xs_c, ys_c = make_seq(X_full_s, yc_full)

seq_split    = int(len(Xs_r) * 0.80)
Xs_tr, Xs_te = Xs_r[:seq_split], Xs_r[seq_split:]
yr_sq_tr, yr_sq_te = ys_r[:seq_split], ys_r[seq_split:]
yc_sq_tr, yc_sq_te = ys_c[:seq_split].astype(int), ys_c[seq_split:].astype(int)

N_FEAT = X.shape[1]
print(f"  Sequence shapes — Train: {Xs_tr.shape}  Test: {Xs_te.shape}")

# Results accumulators
reg_results = []
clf_results = []

def reg_m(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:<25}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return dict(Model=name, MAE=mae, RMSE=rmse, R2=r2)

def clf_m(y_true, y_pred, y_prob, name):
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_prob)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    print(f"  {name:<25}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  "
          f"Prec={prec:.3f}  Rec={rec:.3f}  Brier={brier:.3f}")
    return dict(Model=name, Accuracy=acc, F1=f1, AUC=auc,
                Precision=prec, Recall=rec, Brier=brier)

ES = EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss")
RL = ReduceLROnPlateau(patience=7, factor=0.5, monitor="val_loss", verbose=0)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  MODEL 1 – RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════
section("§4  MODEL 1 — RANDOM FOREST")

rf_reg = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3,
                                max_features="sqrt", random_state=SEED, n_jobs=-1)
rf_reg.fit(X_tr_s, yr_tr)
rf_reg_pred = rf_reg.predict(X_te_s)
reg_results.append(reg_m(yr_te, rf_reg_pred, "Random Forest"))

rf_clf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=3,
                                 class_weight="balanced", random_state=SEED, n_jobs=-1)
rf_clf.fit(X_tr_s, yc_tr)
rf_clf_pred = rf_clf.predict(X_te_s)
rf_clf_prob = rf_clf.predict_proba(X_te_s)[:, 1]
clf_results.append(clf_m(yc_te, rf_clf_pred, rf_clf_prob, "Random Forest"))

cv_rf = cross_val_score(rf_clf, X_tr_s, yc_tr, cv=tscv, scoring="f1", n_jobs=-1)
print(f"  RF  CV-F1 (5-fold TimeSeriesSplit): {cv_rf.mean():.3f} ± {cv_rf.std():.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# §5  MODEL 2 – XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════
section("§5  MODEL 2 — XGBOOST")

xgb_reg = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.03,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_alpha=0.1, reg_lambda=1.0,
                             random_state=SEED, verbosity=0,
                             eval_metric="rmse")
xgb_reg.fit(X_tr_s, yr_tr, eval_set=[(X_te_s, yr_te)], verbose=False)
xgb_reg_pred = xgb_reg.predict(X_te_s)
reg_results.append(reg_m(yr_te, xgb_reg_pred, "XGBoost"))

sp_w = (yc_tr==0).sum() / (yc_tr==1).sum()
xgb_clf = xgb.XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.8,
                              reg_alpha=0.1, scale_pos_weight=sp_w,
                              random_state=SEED, verbosity=0,
                              eval_metric="logloss")
xgb_clf.fit(X_tr_s, yc_tr, eval_set=[(X_te_s, yc_te)], verbose=False)
xgb_clf_pred = xgb_clf.predict(X_te_s)
xgb_clf_prob = xgb_clf.predict_proba(X_te_s)[:, 1]
clf_results.append(clf_m(yc_te, xgb_clf_pred, xgb_clf_prob, "XGBoost"))

cv_xgb = cross_val_score(xgb_clf, X_tr_s, yc_tr, cv=tscv, scoring="f1", n_jobs=-1)
print(f"  XGB CV-F1: {cv_xgb.mean():.3f} ± {cv_xgb.std():.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# §6  MODEL 3 – LSTM
# ═══════════════════════════════════════════════════════════════════════════════
section("§6  MODEL 3 — LSTM")

def build_lstm(w, nf, task):
    inp = Input(shape=(w, nf))
    x   = LSTM(64, return_sequences=True)(inp)
    x   = Dropout(0.3)(x)
    x   = LSTM(32)(x)
    x   = Dropout(0.2)(x)
    x   = Dense(16, activation="relu")(x)
    out = Dense(1)(x) if task=="reg" else Dense(1, activation="sigmoid")(x)
    m   = Model(inp, out)
    m.compile(Adam(1e-3), loss="mse" if task=="reg" else "binary_crossentropy",
              metrics=["mae" if task=="reg" else "accuracy"])
    return m

lstm_reg = build_lstm(WINDOW, N_FEAT, "reg")
lstm_reg.fit(Xs_tr, yr_sq_tr, epochs=120, batch_size=16,
             validation_split=0.2, callbacks=[ES, RL], verbose=0)
lstm_reg_pred = lstm_reg.predict(Xs_te, verbose=0).flatten()
reg_results.append(reg_m(yr_sq_te, lstm_reg_pred, "LSTM"))

lstm_clf = build_lstm(WINDOW, N_FEAT, "clf")
lstm_clf.fit(Xs_tr, yc_sq_tr, epochs=120, batch_size=16,
             validation_split=0.2, callbacks=[ES, RL], verbose=0)
lstm_clf_prob = lstm_clf.predict(Xs_te, verbose=0).flatten()
lstm_clf_pred = (lstm_clf_prob >= 0.5).astype(int)
clf_results.append(clf_m(yc_sq_te, lstm_clf_pred, lstm_clf_prob, "LSTM"))


# ═══════════════════════════════════════════════════════════════════════════════
# §7  MODEL 4 – CNN-LSTM
# ═══════════════════════════════════════════════════════════════════════════════
section("§7  MODEL 4 — CNN-LSTM")

def build_cnn_lstm(w, nf, task):
    inp = Input(shape=(w, nf))
    x   = Conv1D(32, kernel_size=3, activation="relu", padding="same")(inp)
    x   = MaxPooling1D(pool_size=2, padding="same")(x)
    x   = Conv1D(16, kernel_size=2, activation="relu", padding="same")(x)
    x   = LSTM(32)(x)
    x   = Dropout(0.3)(x)
    x   = Dense(16, activation="relu")(x)
    out = Dense(1)(x) if task=="reg" else Dense(1, activation="sigmoid")(x)
    m   = Model(inp, out)
    m.compile(Adam(1e-3), loss="mse" if task=="reg" else "binary_crossentropy",
              metrics=["mae" if task=="reg" else "accuracy"])
    return m

cnn_r = build_cnn_lstm(WINDOW, N_FEAT, "reg")
cnn_r.fit(Xs_tr, yr_sq_tr, epochs=120, batch_size=16,
          validation_split=0.2, callbacks=[ES, RL], verbose=0)
cnn_r_pred = cnn_r.predict(Xs_te, verbose=0).flatten()
reg_results.append(reg_m(yr_sq_te, cnn_r_pred, "CNN-LSTM"))

cnn_c = build_cnn_lstm(WINDOW, N_FEAT, "clf")
cnn_c.fit(Xs_tr, yc_sq_tr, epochs=120, batch_size=16,
          validation_split=0.2, callbacks=[ES, RL], verbose=0)
cnn_c_prob = cnn_c.predict(Xs_te, verbose=0).flatten()
cnn_c_pred = (cnn_c_prob >= 0.5).astype(int)
clf_results.append(clf_m(yc_sq_te, cnn_c_pred, cnn_c_prob, "CNN-LSTM"))


# ═══════════════════════════════════════════════════════════════════════════════
# §8  MODEL 5 – TRANSFORMER ENCODER
# ═══════════════════════════════════════════════════════════════════════════════
section("§8  MODEL 5 — TRANSFORMER (encoder)")

def build_transformer(w, nf, task, heads=2, ff=32, drop=0.2):
    inp = Input(shape=(w, nf))
    kd  = max(1, nf // heads)
    att = MultiHeadAttention(num_heads=heads, key_dim=kd, dropout=drop)(inp, inp)
    x   = LayerNormalization(epsilon=1e-6)(att + inp)
    ff1 = Dense(ff, activation="relu")(x)
    ff2 = Dense(nf)(ff1)
    x   = LayerNormalization(epsilon=1e-6)(x + Dropout(drop)(ff2))
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(16, activation="relu")(x)
    x   = Dropout(drop)(x)
    out = Dense(1)(x) if task=="reg" else Dense(1, activation="sigmoid")(x)
    m   = Model(inp, out)
    m.compile(Adam(5e-4), loss="mse" if task=="reg" else "binary_crossentropy",
              metrics=["mae" if task=="reg" else "accuracy"])
    return m

tf_r = build_transformer(WINDOW, N_FEAT, "reg")
tf_r.fit(Xs_tr, yr_sq_tr, epochs=120, batch_size=16,
         validation_split=0.2, callbacks=[ES, RL], verbose=0)
tf_r_pred = tf_r.predict(Xs_te, verbose=0).flatten()
reg_results.append(reg_m(yr_sq_te, tf_r_pred, "Transformer"))

tf_c = build_transformer(WINDOW, N_FEAT, "clf")
tf_c.fit(Xs_tr, yc_sq_tr, epochs=120, batch_size=16,
         validation_split=0.2, callbacks=[ES, RL], verbose=0)
tf_c_prob = tf_c.predict(Xs_te, verbose=0).flatten()
tf_c_pred = (tf_c_prob >= 0.5).astype(int)
clf_results.append(clf_m(yc_sq_te, tf_c_pred, tf_c_prob, "Transformer"))


# ═══════════════════════════════════════════════════════════════════════════════
# §9  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
section("§9  RESULTS TABLES")

df_reg = pd.DataFrame(reg_results).set_index("Model")
df_clf = pd.DataFrame(clf_results).set_index("Model")

df_reg.to_csv(os.path.join(DATA_DIR, "outputs", "results_regression.csv"))
df_clf.to_csv(os.path.join(DATA_DIR, "outputs", "results_classification.csv"))

print("\n  ── Regression ──"); print(df_reg.round(4).to_string())
print("\n  ── Classification ──"); print(df_clf.round(4).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# §10  FIGURE 11 – MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
section("§10  FIGURE 11 — MODEL COMPARISON")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = df_clf.index.tolist()
mc = [MODEL_COLORS.get(m, C["gray"]) for m in models]
x  = np.arange(len(models))

# AUC-ROC
axes[0,0].bar(x, df_clf["AUC"], color=mc, edgecolor="white", linewidth=0.8)
best_auc = df_clf["AUC"].idxmax()
axes[0,0].bar(x[models.index(best_auc)], df_clf.loc[best_auc, "AUC"],
              color=MODEL_COLORS.get(best_auc, C["gray"]),
              edgecolor=C["navy"], linewidth=2.5)
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(models, rotation=20, ha="right")
axes[0,0].set_ylim(0.4, 1.0); axes[0,0].set_ylabel("AUC-ROC")
axes[0,0].set_title("A   AUC-ROC (Classification)", loc="left")
for i, v in enumerate(df_clf["AUC"]):
    axes[0,0].text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8.5)
axes[0,0].axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="Random baseline")
axes[0,0].legend()

# F1
axes[0,1].bar(x, df_clf["F1"], color=mc, edgecolor="white", linewidth=0.8)
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(models, rotation=20, ha="right")
axes[0,1].set_ylim(0.0, 1.0); axes[0,1].set_ylabel("F1 Score")
axes[0,1].set_title("B   F1 Score (Classification)", loc="left")
for i, v in enumerate(df_clf["F1"]):
    axes[0,1].text(i, v+0.01, f"{v:.3f}", ha="center", fontsize=8.5)

# RMSE
mr = df_reg.index.tolist()
mc_r = [MODEL_COLORS.get(m, C["gray"]) for m in mr]
xr   = np.arange(len(mr))
axes[1,0].bar(xr, df_reg["RMSE"], color=mc_r, edgecolor="white")
axes[1,0].set_xticks(xr); axes[1,0].set_xticklabels(mr, rotation=20, ha="right")
axes[1,0].set_ylabel("RMSE (%)"); axes[1,0].set_title("C   RMSE — Sleep Ratio Regression", loc="left")
for i, v in enumerate(df_reg["RMSE"]):
    axes[1,0].text(i, v+0.1, f"{v:.2f}", ha="center", fontsize=8.5)

# R²
axes[1,1].bar(xr, df_reg["R2"], color=mc_r, edgecolor="white")
axes[1,1].set_xticks(xr); axes[1,1].set_xticklabels(mr, rotation=20, ha="right")
axes[1,1].set_ylabel("R²"); axes[1,1].set_title("D   R² Score — Regression", loc="left")
axes[1,1].axhline(0, color="gray", ls="--", lw=1)
for i, v in enumerate(df_reg["R2"]):
    axes[1,1].text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=8.5)

fig.suptitle("Figure 11 · Model Performance Comparison — 5 Architectures",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig11_model_comparison.png"))
plt.close()
print("  ✓  fig11_model_comparison.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §11  FIGURE 12 – ROC CURVES
# ═══════════════════════════════════════════════════════════════════════════════
section("§11  FIGURE 12 — ROC CURVES")

clf_pred_dict = {
    "Random Forest": (yc_te,    rf_clf_prob),
    "XGBoost"      : (yc_te,    xgb_clf_prob),
    "LSTM"         : (yc_sq_te, lstm_clf_prob),
    "CNN-LSTM"     : (yc_sq_te, cnn_c_prob),
    "Transformer"  : (yc_sq_te, tf_c_prob),
}

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1],[0,1], "--", color="gray", lw=1.2, label="Random (AUC=0.50)")

for name, (yt, yp) in clf_pred_dict.items():
    fpr, tpr, _ = roc_curve(yt, yp)
    auc = roc_auc_score(yt, yp)
    ax.plot(fpr, tpr, lw=2.2, color=MODEL_COLORS[name],
            label=f"{name}  (AUC = {auc:.3f})")

ax.fill_between([0,1],[0,0],[0,1], alpha=0.03, color=C["gray"])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("Figure 12 · ROC Curves — Sleep Quality Classification",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
ax.text(0.5, 0.1, "Poor sleep = Positive class", ha="center",
        fontsize=9, color=C["gray"], style="italic")

plt.savefig(os.path.join(FIG_DIR, "fig12_roc_curves.png"))
plt.close()
print("  ✓  fig12_roc_curves.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §12  FIGURE 13 – PREDICTED vs ACTUAL (TIME SERIES)
# ═══════════════════════════════════════════════════════════════════════════════
section("§12  FIGURE 13 — PREDICTED vs ACTUAL")

pred_flat = {"Random Forest": rf_reg_pred, "XGBoost": xgb_reg_pred}
pred_seq  = {"LSTM": lstm_reg_pred, "CNN-LSTM": cnn_r_pred, "Transformer": tf_r_pred}

_date_col  = df_m.index if "Date" not in df_m.columns else df_m["Date"]
dates_flat = _date_col.values[split:]
n_seq_offset = len(df_m) - len(X_full_s) + seq_split
dates_seq  = _date_col.values[n_seq_offset : n_seq_offset + len(Xs_te)]

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()
idx  = 0
for name, pred in {**pred_flat, **pred_seq}.items():
    ax   = axes[idx]
    dates = dates_seq if name in pred_seq else dates_flat
    ytrue = yr_sq_te if name in pred_seq else yr_te

    ax.plot(dates, ytrue, lw=1.8, color=C["navy"],   label="Actual",    zorder=3)
    ax.plot(dates, pred,  lw=1.6, color=MODEL_COLORS[name],
            ls="--", label="Predicted", alpha=0.9, zorder=4)
    ax.fill_between(dates, ytrue, pred, alpha=0.08, color=MODEL_COLORS[name])
    ax.axhline(80, color="gray", ls=":", lw=1, alpha=0.5, label="Quality threshold (80%)")

    rmse = np.sqrt(mean_squared_error(ytrue, pred))
    r2   = r2_score(ytrue, pred)
    ax.text(0.02, 0.97, f"RMSE={rmse:.2f}  R²={r2:.3f}", transform=ax.transAxes,
            va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor=C["lightbg"]))

    ax.set_title(name, fontweight="bold")
    ax.set_ylabel("Sleep Quality (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)
    idx += 1

axes[-1].set_visible(False)
fig.suptitle("Figure 13 · Predicted vs Actual — Next-Night Sleep Quality (%)",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig13_ts_predictions.png"))
plt.close()
print("  ✓  fig13_ts_predictions.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §13  FIGURE 14 – CROSS-VALIDATION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
section("§13  FIGURE 14 — CROSS-VALIDATION FOLD RESULTS")

cv_models = {"Random Forest": rf_clf, "XGBoost": xgb_clf}
cv_metrics = {"F1": "f1", "Accuracy": "accuracy", "AUC": "roc_auc"}

fig, axes = plt.subplots(1, len(cv_metrics), figsize=(14, 5))

for ax, (metric_name, scoring) in zip(axes, cv_metrics.items()):
    fold_scores = {}
    for mname, model in cv_models.items():
        scores = cross_val_score(model, X_tr_s, yc_tr, cv=tscv,
                                 scoring=scoring, n_jobs=-1)
        fold_scores[mname] = scores

    x_pos = np.arange(tscv.n_splits) + 1
    for i, (mname, scores) in enumerate(fold_scores.items()):
        offset = (i - 0.5) * 0.3
        ax.bar(x_pos + offset, scores, 0.28,
               color=MODEL_COLORS[mname], label=mname if metric_name=="F1" else "",
               edgecolor="white", alpha=0.85)
        ax.errorbar(x_pos + offset, scores,
                    yerr=None, fmt="none", ecolor="black")

    ax.axhline(np.mean([s for ss in fold_scores.values() for s in ss]),
               color="gray", ls="--", lw=1.5, alpha=0.7, label="Overall mean")
    ax.set_xlabel("CV Fold")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} per Fold (TimeSeriesSplit)", fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_ylim(0, 1.05)
    if metric_name == "F1":
        ax.legend()

fig.suptitle("Figure 14 · Time-Series Cross-Validation Results (5 Folds)",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig14_cv_results.png"))
plt.close()
print("  ✓  fig14_cv_results.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §14  FIGURE 15 + 16 – SHAP EXPLAINABILITY (XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════
section("§14  FIGURES 15-16 — SHAP EXPLAINABILITY")

explainer   = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_te_s)
shap_df     = pd.DataFrame(shap_values, columns=FEATURE_COLS)

# ── Figure 15: SHAP Summary ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Beeswarm
plt.sca(axes[0])
shap.summary_plot(shap_values, X_te_s, feature_names=FEATURE_COLS,
                  show=False, max_display=18, plot_size=None)
axes[0].set_title("A   SHAP Beeswarm — Feature Impact on Sleep Quality\n"
                  "(red = high feature value; blue = low)",
                  loc="left", fontweight="bold", fontsize=9)

# Bar
plt.sca(axes[1])
shap.summary_plot(shap_values, X_te_s, feature_names=FEATURE_COLS,
                  plot_type="bar", show=False, max_display=18, plot_size=None,
                  color=C["teal"])
axes[1].set_title("B   Mean |SHAP| Feature Importance\n(XGBoost — Sleep Quality Classifier)",
                  loc="left", fontweight="bold", fontsize=9)

fig.suptitle("Figure 15 · SHAP Explainability — XGBoost Sleep Quality Classifier",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig15_shap_summary.png"), bbox_inches="tight")
plt.close()
print("  ✓  fig15_shap_summary.png saved")

# ── Figure 16: SHAP Dependence (HRV + Steps) ─────────────────────────────────
dep_features = []
for fname in ["Avg. HRV(ms)", "Steps", "Avg. HRV(ms)_lag1", "Avg. Heart Rate(bpm)"]:
    if fname in FEATURE_COLS:
        dep_features.append(fname)
dep_features = dep_features[:4]

if dep_features:
    fig, axes = plt.subplots(1, len(dep_features), figsize=(16, 5))
    if len(dep_features) == 1:
        axes = [axes]
    for ax, fname in zip(axes, dep_features):
        idx_f = FEATURE_COLS.index(fname)
        sc = ax.scatter(X_te_s[:, idx_f], shap_values[:, idx_f],
                        c=X_te_s[:, idx_f], cmap="RdBu_r", s=35, alpha=0.75,
                        edgecolors="none")
        plt.colorbar(sc, ax=ax, fraction=0.04, label="Feature value (scaled)")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_xlabel(textwrap.shorten(fname, 25))
        ax.set_ylabel("SHAP value")
        ax.set_title(textwrap.shorten(fname, 20), fontweight="bold")

        r_val, p_val = spearmanr(X_te_s[:, idx_f], shap_values[:, idx_f])
        ax.text(0.05, 0.97, f"ρ={r_val:.3f}\np={p_val:.4f}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor=C["lightbg"]))

    fig.suptitle("Figure 16 · SHAP Dependence Plots — Top Predictors",
                 fontsize=13, fontweight="bold")
    plt.savefig(os.path.join(FIG_DIR, "fig16_shap_dependence.png"))
    plt.close()
    print("  ✓  fig16_shap_dependence.png saved")

# ── SHAP Waterfall (single prediction) ───────────────────────────────────────
poor_idx = np.where(yc_te == 0)[0]
if len(poor_idx) > 0:
    ii = poor_idx[0]
    fig = plt.figure(figsize=(10, 6))
    exp = shap.Explanation(values=shap_values[ii],
                           base_values=explainer.expected_value,
                           data=X_te_s[ii], feature_names=FEATURE_COLS)
    shap.plots.waterfall(exp, max_display=14, show=False)
    plt.title("SHAP Waterfall — Single Poor-Sleep Prediction (XGBoost)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig16b_shap_waterfall.png"), bbox_inches="tight")
    plt.close()
    print("  ✓  fig16b_shap_waterfall.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §15  FIGURE 17 – CONFUSION MATRICES
# ═══════════════════════════════════════════════════════════════════════════════
section("§15  FIGURE 17 — CONFUSION MATRICES")

clf_all = {
    "Random Forest": (yc_te,    rf_clf_pred),
    "XGBoost"      : (yc_te,    xgb_clf_pred),
    "LSTM"         : (yc_sq_te, lstm_clf_pred),
    "CNN-LSTM"     : (yc_sq_te, cnn_c_pred),
    "Transformer"  : (yc_sq_te, tf_c_pred),
}

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, (name, (yt, yp)) in zip(axes, clf_all.items()):
    cm   = confusion_matrix(yt, yp)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Poor", "Good"])
    disp.plot(ax=ax, colorbar=False,
              cmap=mpl.colors.LinearSegmentedColormap.from_list(
                  "custom", ["white", MODEL_COLORS[name]]))
    acc = accuracy_score(yt, yp)
    f1  = f1_score(yt, yp, zero_division=0)
    ax.set_title(f"{name}\nAcc={acc:.2f}  F1={f1:.2f}", fontsize=9, fontweight="bold")

fig.suptitle("Figure 17 · Confusion Matrices — Sleep Quality Classification",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig17_confusion_matrices.png"))
plt.close()
print("  ✓  fig17_confusion_matrices.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §16  FIGURE 18 – FEATURE IMPORTANCE (RF + XGBoost)
# ═══════════════════════════════════════════════════════════════════════════════
section("§16  FIGURE 18 — FEATURE IMPORTANCE COMPARISON")

fi_rf  = pd.Series(rf_reg.feature_importances_,  index=FEATURE_COLS)
fi_xgb = pd.Series(xgb_reg.feature_importances_, index=FEATURE_COLS)
fi_shap = pd.Series(np.abs(shap_values).mean(0),  index=FEATURE_COLS)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

for ax, (fi, name, color) in zip(axes, [
    (fi_rf,   "Random Forest\n(Gini importance)",  C["rf"]),
    (fi_xgb,  "XGBoost\n(Gain importance)",        C["xgb"]),
    (fi_shap, "SHAP\n(Mean |SHAP value|)",          C["teal"]),
]):
    top = fi.sort_values(ascending=False).head(20)
    ax.barh(range(len(top)), top.values, color=color, alpha=0.8, edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([textwrap.shorten(n, 28) for n in top.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top 20 Features\n{name}", loc="left", fontsize=10, fontweight="bold")

fig.suptitle("Figure 18 · Feature Importance: RF vs XGBoost vs SHAP",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig18_feature_importance.png"))
plt.close()
print("  ✓  fig18_feature_importance.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §17  FIGURE 19 – LEARNING CURVES
# ═══════════════════════════════════════════════════════════════════════════════
section("§17  FIGURE 19 — LEARNING CURVES")

def _manual_learning_curve(model, X, y, fractions, n_splits=4):
    """Learning curve safe for heavily imbalanced small datasets."""
    from sklearn.base import clone
    from sklearn.metrics import f1_score as _f1
    tr_scores, cv_scores, abs_sizes = [], [], []
    for frac in fractions:
        n = max(int(len(X) * frac), 10)
        Xf, yf = X[:n], y[:n]
        if len(np.unique(yf)) < 2:
            continue
        # simple holdout: last 20% as val
        split_i = int(len(Xf) * 0.8)
        if split_i < 5 or (len(Xf) - split_i) < 3:
            continue
        Xtr, Xvl = Xf[:split_i], Xf[split_i:]
        ytr, yvl = yf[:split_i], yf[split_i:]
        if len(np.unique(ytr)) < 2 or len(np.unique(yvl)) < 2:
            continue
        try:
            m = clone(model); m.fit(Xtr, ytr)
            tr_scores.append(_f1(ytr, m.predict(Xtr), zero_division=0))
            cv_scores.append(_f1(yvl, m.predict(Xvl), zero_division=0))
            abs_sizes.append(n)
        except Exception:
            continue
    return np.array(abs_sizes), np.array(tr_scores), np.array(cv_scores)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (model, name, color) in zip(axes, [
    (rf_clf,  "Random Forest", C["rf"]),
    (xgb_clf, "XGBoost",       C["xgb"]),
]):
    fracs = np.linspace(0.35, 1.0, 8)
    sizes, tr_sc, cv_sc = _manual_learning_curve(model, X_tr_s, yc_tr, fracs)
    if len(sizes) == 0:
        ax.text(0.5, 0.5, "Insufficient class diversity\nfor learning curve",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(f"Learning Curve — {name}", fontweight="bold")
        continue
    ax.plot(sizes, tr_sc, "o-", color=color,      lw=2, label="Train F1")
    ax.plot(sizes, cv_sc, "s--", color=C["gray"], lw=2, label="Val F1")
    ax.fill_between(sizes, tr_sc, cv_sc, alpha=0.08, color=color)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Learning Curve — {name}", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend()
    gap = tr_sc[-1] - cv_sc[-1]
    status = "High variance" if gap > 0.15 else "Balanced" if gap > 0.05 else "Well-fitted"
    ax.text(0.98, 0.07, f"Train−Val gap: {gap:.2f}\n({status})",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round", facecolor=C["lightbg"]))

fig.suptitle("Figure 19 · Learning Curves — Bias-Variance Analysis",
             fontsize=13, fontweight="bold")
plt.savefig(os.path.join(FIG_DIR, "fig19_learning_curves.png"))
plt.close()
print("  ✓  fig19_learning_curves.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §18  FIGURE 20 – CALIBRATION CURVES
# ═══════════════════════════════════════════════════════════════════════════════
section("§18  FIGURE 20 — CALIBRATION CURVES")

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot([0,1],[0,1], "k--", lw=1.5, label="Perfect calibration")

for name, (yt, yp) in [
    ("Random Forest", (yc_te,    rf_clf_prob)),
    ("XGBoost",       (yc_te,    xgb_clf_prob)),
    ("LSTM",          (yc_sq_te, lstm_clf_prob)),
]:
    try:
        pt, pp = calibration_curve(yt, yp, n_bins=6)
        brier  = brier_score_loss(yt, yp)
        ax.plot(pp, pt, "o-", lw=2, color=MODEL_COLORS[name],
                label=f"{name}  (Brier={brier:.3f})")
    except Exception as e:
        print(f"  Calibration plot skipped for {name}: {e}")

ax.fill_between([0,1],[0,0.1],[-0.1,0.1+1], alpha=0.05, color=C["gray"])
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives (True Probability)")
ax.set_title("Figure 20 · Calibration Curves — Sleep Quality Classifiers",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
plt.savefig(os.path.join(FIG_DIR, "fig20_calibration_curves.png"))
plt.close()
print("  ✓  fig20_calibration_curves.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# §19  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
section("§19  FINAL SUMMARY")

print("\n  ── Classification ──")
print(df_clf.round(4).to_string())
print("\n  ── Regression ──")
print(df_reg.round(4).to_string())

best_clf_m = df_clf["AUC"].idxmax()
best_reg_m = df_reg["R2"].idxmax()
print(f"\n  ★  Best classifier (AUC) : {best_clf_m}  →  AUC = {df_clf.loc[best_clf_m,'AUC']:.4f}")
print(f"  ★  Best regressor  (R²)  : {best_reg_m}  →  R²  = {df_reg.loc[best_reg_m,'R2']:.4f}")

# SHAP summary for report
shap_top = pd.Series(np.abs(shap_values).mean(0), index=FEATURE_COLS).sort_values(ascending=False).head(5)
print("\n  ── Top 5 SHAP features (XGBoost) ──")
for feat, val in shap_top.items():
    print(f"    {feat:<35}  mean|SHAP| = {val:.4f}")

print("\n╔" + "═"*68 + "╗")
print("║  CODE 2 COMPLETE — all figures saved to:                           ║")
print(f"║  {FIG_DIR[:62]:<62} ║")
print("╚" + "═"*68 + "╝")
