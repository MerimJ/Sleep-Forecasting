"""
=============================================================================
CODE 2 — MODELING, VALIDATION & EXPLAINABILITY
Project : Forecasting Cardiovascular Health & Sleep Quality from
          Wearable Time-Series (MLPR / CDA)
Author  : Merim Jusufbegovic
=============================================================================

Pre-requisite: run code1_data_prep_eda.py first to generate
               ~/Downloads/MLPR-CDA/processed_dataset.csv

USAGE:
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap
                tensorflow keras statsmodels
    python code2_modeling_validation.py

Models trained:
    1. Random Forest          (classical ML, regression + classification)
    2. XGBoost                (gradient boosting, regression + classification)
    3. LSTM                   (deep learning sequence model)
    4. CNN-LSTM               (convolutional + LSTM)
    5. Transformer (encoder)  (attention-based sequence model)

Outputs (saved to ~/Downloads/MLPR-CDA/outputs/):
    model_comparison.png
    roc_curves.png
    feature_importance_rf.png
    feature_importance_xgb.png
    shap_summary.png
    shap_dependence_hrv.png
    learning_curves.png
    ts_predictions.png
    confusion_matrices.png
    calibration_curves.png
    results_table.csv
=============================================================================
"""

import os, warnings, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# scikit-learn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (TimeSeriesSplit, cross_val_score,
                                     learning_curve, StratifiedKFold)
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, calibration_curve, brier_score_loss)
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# XGBoost
import xgboost as xgb

# SHAP
import shap

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Conv1D, MaxPooling1D,
                                      Flatten, Input, MultiHeadAttention,
                                      LayerNormalization, GlobalAveragePooling1D,
                                      Reshape)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.expanduser("~/Downloads/MLPR-CDA")
OUT_DIR  = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

PROCESSED_CSV = os.path.join(DATA_DIR, "processed_dataset.csv")

print("=" * 70)
print("CODE 2 — MODELING, VALIDATION & EXPLAINABILITY")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD PROCESSED DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading processed dataset …")
df = pd.read_csv(PROCESSED_CSV, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
print(f"    Shape: {df.shape}  |  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Selecting features …")

# Exclude any leaking/raw sleep columns and targets
EXCLUDE = [
    "Date",
    "Sleep Time Ratio(%)", "Time Asleep(min)", "total_duration_min",
    "sleep_efficiency", "wake_after_sleep_onset",
    "Sleep Stages - Awake(min)", "Sleep Stages - REM(min)",
    "Sleep Stages - Light Sleep(min)", "Sleep Stages - Deep Sleep(min)",
    "rem_ratio", "deep_ratio", "light_ratio",
    "sleep_onset_latency_min", "sleep_quality_label",
    "next_sleep_ratio", "next_sleep_quality",
    "month"
]

FEATURE_COLS = [c for c in df.columns if c not in EXCLUDE
                and df[c].dtype != "object"]

print(f"    Feature count: {len(FEATURE_COLS)}")
print(f"    Features: {FEATURE_COLS}")

TARGET_REG  = "next_sleep_ratio"      # regression target
TARGET_CLF  = "next_sleep_quality"    # classification target (0=Poor, 1=Good)

df_model = df[FEATURE_COLS + [TARGET_REG, TARGET_CLF]].dropna().copy()
print(f"    Modelling rows: {len(df_model)}")

X = df_model[FEATURE_COLS].values.astype(np.float32)
y_reg = df_model[TARGET_REG].values.astype(np.float32)
y_clf = df_model[TARGET_CLF].values.astype(int)

print(f"    Class balance → Poor: {(y_clf==0).sum()}, Good: {(y_clf==1).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TIME-SERIES TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Time-series train/test split (80/20 chronological) …")

split_idx = int(len(df_model) * 0.80)
X_train, X_test = X[:split_idx], X[split_idx:]
y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
y_clf_train, y_clf_test = y_clf[:split_idx], y_clf[split_idx:]

# Scale features (fit on train only)
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"    Train: {X_train_s.shape}  |  Test: {X_test_s.shape}")

# TimeSeriesSplit for cross-validation (5 folds)
tscv = TimeSeriesSplit(n_splits=5)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def reg_metrics(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"    {label:20s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    return {"label": label, "MAE": mae, "RMSE": rmse, "R2": r2}

def clf_metrics(y_true, y_pred, y_prob, label=""):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    print(f"    {label:20s}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc:.3f}  "
          f"Prec={prec:.3f}  Rec={rec:.3f}  Brier={brier:.3f}")
    return {"label": label, "Accuracy": acc, "F1": f1, "AUC": auc,
            "Precision": prec, "Recall": rec, "Brier": brier}

def make_sequences(X, y, window=7):
    """Create sliding-window sequences for LSTM/CNN input."""
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i-window:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

WINDOW = 7    # 7-day lookback window for sequence models

# Build sequences from the full scaled dataset then re-split
X_full_s = np.vstack([X_train_s, X_test_s])
y_reg_full = np.concatenate([y_reg_train, y_reg_test])
y_clf_full = np.concatenate([y_clf_train, y_clf_test])

Xs_seq, ys_reg_seq = make_sequences(X_full_s, y_reg_full, WINDOW)
_,      ys_clf_seq = make_sequences(X_full_s, y_clf_full, WINDOW)

seq_split = int(len(Xs_seq) * 0.80)
Xs_train, Xs_test = Xs_seq[:seq_split], Xs_seq[seq_split:]
ys_reg_train_seq, ys_reg_test_seq = ys_reg_seq[:seq_split], ys_reg_seq[seq_split:]
ys_clf_train_seq, ys_clf_test_seq = ys_clf_seq[:seq_split], ys_clf_seq[seq_split:]

N_FEATURES = X.shape[1]

results_reg = []
results_clf = []

print(f"    Sequence shape: {Xs_train.shape}  (train)  |  {Xs_test.shape}  (test)")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL 1 — RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Random Forest …")

# ── 5a. Regression
rf_reg = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                random_state=SEED, n_jobs=-1)
rf_reg.fit(X_train_s, y_reg_train)
rf_reg_pred = rf_reg.predict(X_test_s)
results_reg.append(reg_metrics(y_reg_test, rf_reg_pred, "RF Regression"))

# ── 5b. Classification
rf_clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                 class_weight="balanced", random_state=SEED, n_jobs=-1)
rf_clf.fit(X_train_s, y_clf_train)
rf_clf_pred = rf_clf.predict(X_test_s)
rf_clf_prob = rf_clf.predict_proba(X_test_s)[:, 1]
results_clf.append(clf_metrics(y_clf_test, rf_clf_pred, rf_clf_prob, "RF Classification"))

# ── Cross-validation
cv_rf_f1 = cross_val_score(rf_clf, X_train_s, y_clf_train, cv=tscv,
                            scoring="f1", n_jobs=-1)
print(f"    RF CV F1 (TimeSeriesSplit): {cv_rf_f1.mean():.3f} ± {cv_rf_f1.std():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MODEL 2 — XGBOOST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] XGBoost …")

# ── 6a. Regression
xgb_reg = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=SEED, verbosity=0)
xgb_reg.fit(X_train_s, y_reg_train,
            eval_set=[(X_test_s, y_reg_test)], verbose=False)
xgb_reg_pred = xgb_reg.predict(X_test_s)
results_reg.append(reg_metrics(y_reg_test, xgb_reg_pred, "XGBoost Regression"))

# ── 6b. Classification
scale_pos = (y_clf_train == 0).sum() / (y_clf_train == 1).sum()
xgb_clf = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=scale_pos,
                              random_state=SEED, verbosity=0,
                              eval_metric="logloss")
xgb_clf.fit(X_train_s, y_clf_train,
            eval_set=[(X_test_s, y_clf_test)], verbose=False)
xgb_clf_pred = xgb_clf.predict(X_test_s)
xgb_clf_prob = xgb_clf.predict_proba(X_test_s)[:, 1]
results_clf.append(clf_metrics(y_clf_test, xgb_clf_pred, xgb_clf_prob, "XGBoost Classification"))

cv_xgb_f1 = cross_val_score(xgb_clf, X_train_s, y_clf_train, cv=tscv,
                              scoring="f1", n_jobs=-1)
print(f"    XGB CV F1: {cv_xgb_f1.mean():.3f} ± {cv_xgb_f1.std():.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MODEL 3 — LSTM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] LSTM …")

ES = EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss")
RL = ReduceLROnPlateau(patience=7, factor=0.5, monitor="val_loss")

def build_lstm(window, n_feat, task="reg"):
    inp = Input(shape=(window, n_feat))
    x   = LSTM(64, return_sequences=True)(inp)
    x   = Dropout(0.3)(x)
    x   = LSTM(32)(x)
    x   = Dropout(0.2)(x)
    x   = Dense(16, activation="relu")(x)
    if task == "reg":
        out = Dense(1)(x)
        model = Model(inp, out)
        model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    else:
        out = Dense(1, activation="sigmoid")(x)
        model = Model(inp, out)
        model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy",
                      metrics=["accuracy"])
    return model

# Regression LSTM
lstm_reg = build_lstm(WINDOW, N_FEATURES, "reg")
lstm_reg.fit(Xs_train, ys_reg_train_seq, epochs=100, batch_size=16,
             validation_split=0.2, callbacks=[ES, RL], verbose=0)
lstm_reg_pred = lstm_reg.predict(Xs_test, verbose=0).flatten()
results_reg.append(reg_metrics(ys_reg_test_seq, lstm_reg_pred, "LSTM Regression"))

# Classification LSTM
lstm_clf = build_lstm(WINDOW, N_FEATURES, "clf")
lstm_clf.fit(Xs_train, ys_clf_train_seq, epochs=100, batch_size=16,
             validation_split=0.2, callbacks=[ES, RL], verbose=0)
lstm_clf_prob = lstm_clf.predict(Xs_test, verbose=0).flatten()
lstm_clf_pred = (lstm_clf_prob >= 0.5).astype(int)
results_clf.append(clf_metrics(ys_clf_test_seq, lstm_clf_pred, lstm_clf_prob,
                               "LSTM Classification"))


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MODEL 4 — CNN-LSTM
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] CNN-LSTM …")

def build_cnn_lstm(window, n_feat, task="reg"):
    inp = Input(shape=(window, n_feat))
    x   = Conv1D(32, kernel_size=3, activation="relu", padding="same")(inp)
    x   = Conv1D(16, kernel_size=2, activation="relu", padding="same")(x)
    x   = LSTM(32)(x)
    x   = Dropout(0.3)(x)
    x   = Dense(16, activation="relu")(x)
    if task == "reg":
        out = Dense(1)(x)
        model = Model(inp, out)
        model.compile(optimizer=Adam(1e-3), loss="mse", metrics=["mae"])
    else:
        out = Dense(1, activation="sigmoid")(x)
        model = Model(inp, out)
        model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy",
                      metrics=["accuracy"])
    return model

cnn_lstm_reg = build_cnn_lstm(WINDOW, N_FEATURES, "reg")
cnn_lstm_reg.fit(Xs_train, ys_reg_train_seq, epochs=100, batch_size=16,
                 validation_split=0.2, callbacks=[ES, RL], verbose=0)
cnn_lstm_reg_pred = cnn_lstm_reg.predict(Xs_test, verbose=0).flatten()
results_reg.append(reg_metrics(ys_reg_test_seq, cnn_lstm_reg_pred, "CNN-LSTM Regression"))

cnn_lstm_clf = build_cnn_lstm(WINDOW, N_FEATURES, "clf")
cnn_lstm_clf.fit(Xs_train, ys_clf_train_seq, epochs=100, batch_size=16,
                 validation_split=0.2, callbacks=[ES, RL], verbose=0)
cnn_lstm_clf_prob = cnn_lstm_clf.predict(Xs_test, verbose=0).flatten()
cnn_lstm_clf_pred = (cnn_lstm_clf_prob >= 0.5).astype(int)
results_clf.append(clf_metrics(ys_clf_test_seq, cnn_lstm_clf_pred, cnn_lstm_clf_prob,
                               "CNN-LSTM Classification"))


# ─────────────────────────────────────────────────────────────────────────────
# 9.  MODEL 5 — TRANSFORMER (Encoder)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[9] Transformer (encoder) …")

def build_transformer(window, n_feat, task="reg",
                      num_heads=2, ff_dim=32, dropout=0.2):
    inp = Input(shape=(window, n_feat))
    # Multi-head self-attention
    x   = MultiHeadAttention(num_heads=num_heads, key_dim=n_feat // num_heads + 1,
                              dropout=dropout)(inp, inp)
    x   = LayerNormalization(epsilon=1e-6)(x + inp)
    # Feed-forward
    ff  = Dense(ff_dim, activation="relu")(x)
    ff  = Dropout(dropout)(ff)
    ff  = Dense(n_feat)(ff)
    x   = LayerNormalization(epsilon=1e-6)(x + ff)
    # Pool
    x   = GlobalAveragePooling1D()(x)
    x   = Dense(16, activation="relu")(x)
    x   = Dropout(dropout)(x)
    if task == "reg":
        out = Dense(1)(x)
        model = Model(inp, out)
        model.compile(optimizer=Adam(5e-4), loss="mse", metrics=["mae"])
    else:
        out = Dense(1, activation="sigmoid")(x)
        model = Model(inp, out)
        model.compile(optimizer=Adam(5e-4), loss="binary_crossentropy",
                      metrics=["accuracy"])
    return model

tf_reg = build_transformer(WINDOW, N_FEATURES, "reg")
tf_reg.fit(Xs_train, ys_reg_train_seq, epochs=100, batch_size=16,
           validation_split=0.2, callbacks=[ES, RL], verbose=0)
tf_reg_pred = tf_reg.predict(Xs_test, verbose=0).flatten()
results_reg.append(reg_metrics(ys_reg_test_seq, tf_reg_pred, "Transformer Regression"))

tf_clf = build_transformer(WINDOW, N_FEATURES, "clf")
tf_clf.fit(Xs_train, ys_clf_train_seq, epochs=100, batch_size=16,
           validation_split=0.2, callbacks=[ES, RL], verbose=0)
tf_clf_prob = tf_clf.predict(Xs_test, verbose=0).flatten()
tf_clf_pred = (tf_clf_prob >= 0.5).astype(int)
results_clf.append(clf_metrics(ys_clf_test_seq, tf_clf_pred, tf_clf_prob,
                               "Transformer Classification"))


# ─────────────────────────────────────────────────────────────────────────────
# 10.  RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[10] Results summary …")

df_reg_res = pd.DataFrame(results_reg).set_index("label")
df_clf_res = pd.DataFrame(results_clf).set_index("label")

print("\n── Regression metrics:")
print(df_reg_res.round(4).to_string())
print("\n── Classification metrics:")
print(df_clf_res.round(4).to_string())

df_reg_res.to_csv(os.path.join(OUT_DIR, "results_regression.csv"))
df_clf_res.to_csv(os.path.join(OUT_DIR, "results_classification.csv"))
print("    → saved results CSVs")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  MODEL COMPARISON PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[11] Model comparison plots …")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regression: RMSE + R²
x_pos = np.arange(len(df_reg_res))
axes[0].bar(x_pos - 0.2, df_reg_res["RMSE"], 0.35, label="RMSE", color="steelblue")
ax2 = axes[0].twinx()
ax2.bar(x_pos + 0.2, df_reg_res["R2"], 0.35, label="R²", color="coral", alpha=0.7)
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(df_reg_res.index, rotation=30, ha="right", fontsize=8)
axes[0].set_ylabel("RMSE", color="steelblue")
ax2.set_ylabel("R²", color="coral")
axes[0].set_title("Regression: RMSE (blue) & R² (coral)")

# Classification: AUC + F1
axes[1].bar(x_pos - 0.2, df_clf_res["AUC"], 0.35, label="AUC-ROC", color="steelblue")
axes[1].bar(x_pos + 0.2, df_clf_res["F1"],  0.35, label="F1 Score", color="coral")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(df_clf_res.index, rotation=30, ha="right", fontsize=8)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("Score")
axes[1].set_title("Classification: AUC-ROC & F1")
axes[1].legend()

plt.suptitle("Model Comparison", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "14_model_comparison.png"))
plt.close()
print("    → saved 14_model_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  ROC CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[12] ROC curves …")

clf_probs = {
    "Random Forest"  : (y_clf_test,       rf_clf_prob),
    "XGBoost"        : (y_clf_test,       xgb_clf_prob),
    "LSTM"           : (ys_clf_test_seq,  lstm_clf_prob),
    "CNN-LSTM"       : (ys_clf_test_seq,  cnn_lstm_clf_prob),
    "Transformer"    : (ys_clf_test_seq,  tf_clf_prob),
}

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["steelblue", "coral", "green", "purple", "orange"]
for (name, (y_t, y_p)), col in zip(clf_probs.items(), colors):
    fpr, tpr, _ = roc_curve(y_t, y_p)
    auc = roc_auc_score(y_t, y_p)
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Sleep Quality Classification")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "15_roc_curves.png"))
plt.close()
print("    → saved 15_roc_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  CONFUSION MATRICES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[13] Confusion matrices …")

clf_preds = {
    "Random Forest"  : (y_clf_test,      rf_clf_pred),
    "XGBoost"        : (y_clf_test,      xgb_clf_pred),
    "LSTM"           : (ys_clf_test_seq, lstm_clf_pred),
    "CNN-LSTM"       : (ys_clf_test_seq, cnn_lstm_clf_pred),
    "Transformer"    : (ys_clf_test_seq, tf_clf_pred),
}

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, (name, (y_t, y_p)) in zip(axes, clf_preds.items()):
    cm = confusion_matrix(y_t, y_p)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Poor", "Good"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=9)
plt.suptitle("Confusion Matrices — Sleep Quality Classification", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "16_confusion_matrices.png"))
plt.close()
print("    → saved 16_confusion_matrices.png")


# ─────────────────────────────────────────────────────────────────────────────
# 14.  TIME-SERIES PREDICTION PLOT  (best regression model)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[14] Time-series prediction plots …")

# Best model by RMSE
best_reg_label = df_reg_res["RMSE"].idxmin()
print(f"    Best regression model: {best_reg_label}")

pred_map = {
    "RF Regression"          : rf_reg_pred,
    "XGBoost Regression"     : xgb_reg_pred,
    "LSTM Regression"        : lstm_reg_pred,
    "CNN-LSTM Regression"    : cnn_lstm_reg_pred,
    "Transformer Regression" : tf_reg_pred,
}

# Use XGBoost dates for flat models (no sequence window offset)
test_dates_flat = df_model["Date"].values[split_idx:]
# Sequence models have WINDOW extra offset
offset = len(df_model) - len(X_full_s) + seq_split  # start date for seq test
test_dates_seq = df_model["Date"].values[offset : offset + len(Xs_test)]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for i, (name, pred) in enumerate(pred_map.items()):
    if i >= len(axes): break
    is_seq = name in ["LSTM Regression", "CNN-LSTM Regression", "Transformer Regression"]
    dates  = test_dates_seq if is_seq else test_dates_flat
    y_true = ys_reg_test_seq if is_seq else y_reg_test
    axes[i].plot(dates, y_true, label="Actual",    color="steelblue", lw=1.5)
    axes[i].plot(dates, pred,   label="Predicted", color="crimson",   lw=1.5, linestyle="--")
    axes[i].set_title(name, fontsize=9)
    axes[i].set_ylabel("Sleep Time Ratio (%)")
    axes[i].legend(fontsize=8)
    axes[i].tick_params(axis="x", rotation=30)

axes[-1].set_visible(False)
plt.suptitle("Predicted vs Actual — Sleep Time Ratio (%)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "17_ts_predictions.png"))
plt.close()
print("    → saved 17_ts_predictions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 15.  FEATURE IMPORTANCE — RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[15] Feature importance (RF & XGBoost) …")

fi_rf = pd.Series(rf_reg.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 7))
fi_rf.head(20).plot(kind="barh", ax=ax, color="steelblue")
ax.invert_yaxis()
ax.set_title("Random Forest — Top 20 Feature Importances (Regression)")
ax.set_xlabel("Importance (Gini)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "18_feature_importance_rf.png"))
plt.close()
print("    → saved 18_feature_importance_rf.png")

fi_xgb = pd.Series(xgb_reg.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 7))
fi_xgb.head(20).plot(kind="barh", ax=ax, color="coral")
ax.invert_yaxis()
ax.set_title("XGBoost — Top 20 Feature Importances (Regression)")
ax.set_xlabel("Importance (gain)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "19_feature_importance_xgb.png"))
plt.close()
print("    → saved 19_feature_importance_xgb.png")


# ─────────────────────────────────────────────────────────────────────────────
# 16.  SHAP EXPLAINABILITY  (XGBoost — Classification)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[16] SHAP analysis (XGBoost classifier) …")

explainer = shap.TreeExplainer(xgb_clf)
shap_values = explainer.shap_values(X_test_s)

# Summary beeswarm
fig = plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_test_s, feature_names=FEATURE_COLS,
                  show=False, max_display=20)
plt.title("SHAP Summary (XGBoost — Sleep Quality Classification)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "20_shap_summary.png"), bbox_inches="tight")
plt.close()
print("    → saved 20_shap_summary.png")

# Bar plot
fig = plt.figure(figsize=(9, 6))
shap.summary_plot(shap_values, X_test_s, feature_names=FEATURE_COLS,
                  plot_type="bar", show=False, max_display=20)
plt.title("SHAP Feature Importance (mean |SHAP|)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "21_shap_bar.png"), bbox_inches="tight")
plt.close()
print("    → saved 21_shap_bar.png")

# Dependence plot for HRV (research question 3)
hrv_idx = FEATURE_COLS.index("Avg. HRV(ms)") if "Avg. HRV(ms)" in FEATURE_COLS else 0
fig, ax = plt.subplots(figsize=(8, 5))
shap.dependence_plot(hrv_idx, shap_values, X_test_s,
                     feature_names=FEATURE_COLS, ax=ax, show=False)
ax.set_title("SHAP Dependence — Avg. HRV (ms)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "22_shap_dependence_hrv.png"), bbox_inches="tight")
plt.close()
print("    → saved 22_shap_dependence_hrv.png")

# Waterfall for a single "Poor sleep" prediction
poor_idx = np.where(y_clf_test == 0)[0]
if len(poor_idx) > 0:
    idx = poor_idx[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    shap_exp = shap.Explanation(values=shap_values[idx],
                                base_values=explainer.expected_value,
                                data=X_test_s[idx],
                                feature_names=FEATURE_COLS)
    shap.plots.waterfall(shap_exp, max_display=15, show=False)
    plt.title(f"SHAP Waterfall — Single Poor-Sleep Prediction (test idx {idx})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "23_shap_waterfall_poor.png"), bbox_inches="tight")
    plt.close()
    print("    → saved 23_shap_waterfall_poor.png")


# ─────────────────────────────────────────────────────────────────────────────
# 17.  LEARNING CURVES  (RF Classifier)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[17] Learning curves …")

train_sizes, train_scores, val_scores = learning_curve(
    rf_clf, X_train_s, y_clf_train, cv=tscv, scoring="f1",
    train_sizes=np.linspace(0.2, 1.0, 8), n_jobs=-1
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(train_sizes,
                train_scores.mean(1) - train_scores.std(1),
                train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="steelblue")
ax.fill_between(train_sizes,
                val_scores.mean(1) - val_scores.std(1),
                val_scores.mean(1) + val_scores.std(1), alpha=0.15, color="crimson")
ax.plot(train_sizes, train_scores.mean(1), "o-", color="steelblue", label="Train F1")
ax.plot(train_sizes, val_scores.mean(1),   "o-", color="crimson",   label="CV Val F1")
ax.set_xlabel("Training set size")
ax.set_ylabel("F1 Score")
ax.set_title("Learning Curves — Random Forest Classifier")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "24_learning_curves.png"))
plt.close()
print("    → saved 24_learning_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 18.  CALIBRATION CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n[18] Calibration curves …")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

cal_probs_dict = {
    "Random Forest" : (y_clf_test,      rf_clf_prob),
    "XGBoost"       : (y_clf_test,      xgb_clf_prob),
    "LSTM"          : (ys_clf_test_seq, lstm_clf_prob),
}

cols = ["steelblue", "coral", "green"]
for (name, (y_t, y_p)), col in zip(cal_probs_dict.items(), cols):
    try:
        prob_true, prob_pred = calibration_curve(y_t, y_p, n_bins=5)
        ax.plot(prob_pred, prob_true, "o-", color=col, lw=2, label=name)
    except Exception:
        pass

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Calibration Curves — Sleep Quality Classification")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "25_calibration_curves.png"))
plt.close()
print("    → saved 25_calibration_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# 19.  HRV ← → FRAGMENTATION  (Research Question 3 — model-based)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[19] HRV lag analysis (RQ3) …")

hrv_cols_lag = [c for c in FEATURE_COLS if "HRV" in c and "lag" in c]
if hrv_cols_lag:
    # For each lag column, measure Spearman correlation with next-night quality
    lag_corrs = {}
    for col in sorted(hrv_cols_lag):
        r, p = stats.spearmanr(df_model[col].dropna(),
                               df_model["next_sleep_ratio"].loc[df_model[col].dropna().index])
        lag_corrs[col] = {"rho": r, "p": p}
    lag_df = pd.DataFrame(lag_corrs).T
    print("    HRV lag correlations with next-night sleep ratio:")
    print(lag_df.round(4).to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    colors_lag = ["green" if p < 0.05 else "gray" for p in lag_df["p"]]
    ax.bar(lag_df.index, lag_df["rho"], color=colors_lag)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("HRV Lag Correlations with Next-Night Sleep Ratio\n(green = p<0.05)")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "26_hrv_lag_correlation.png"))
    plt.close()
    print("    → saved 26_hrv_lag_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# 20.  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print("\n── Regression (predict next-night Sleep Time Ratio %):")
print(df_reg_res.round(4).to_string())

print("\n── Classification (predict Good/Poor sleep quality):")
print(df_clf_res.round(4).to_string())

best_clf = df_clf_res["AUC"].idxmax()
best_reg = df_reg_res["R2"].idxmax()
print(f"\n★  Best classifier (AUC): {best_clf}  →  AUC={df_clf_res.loc[best_clf,'AUC']:.4f}")
print(f"★  Best regressor  (R²) : {best_reg}  →  R²={df_reg_res.loc[best_reg,'R2']:.4f}")

print("\n" + "=" * 70)
print("CODE 2 COMPLETE — all outputs saved to:", OUT_DIR)
print("=" * 70)
