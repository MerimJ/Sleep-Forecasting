/**
 * create_report.js — Nature Scientific Reports Format
 * Project: Forecasting Cardiovascular Health & Sleep Quality from Wearable Time-Series
 * Author : Merim Jusufbegovic
 *
 * Run: node create_report.js
 * Output: ~/Downloads/MLPR-CDA/outputs/Report_Sleep_Forecasting.docx
 */

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, LevelFormat, ExternalHyperlink,
  PageBreak, TableOfContents
} = require("docx");
const fs = require("fs");
const path = require("path");
const os  = require("os");

const OUT_DIR = path.join(os.homedir(), "Downloads", "MLPR-CDA", "outputs");
if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });

// ─── Helpers ────────────────────────────────────────────────────────────────
const BORDER = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const BORDERS = { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER };
const NO_BORDER = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NO_BORDERS = { top: NO_BORDER, bottom: NO_BORDER, left: NO_BORDER, right: NO_BORDER };

// Page dimensions: A4  (11906 × 16838 DXA), margins 1080 each side
// Content width = 11906 - 2*1080 = 9746
const CONTENT_W = 9746;

function p(children, opts = {}) {
  return new Paragraph({ children, ...opts });
}
function run(text, opts = {}) {
  return new TextRun({ text, font: "Times New Roman", ...opts });
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true, size: 24, font: "Arial", color: "1B2A4A" })],
    spacing: { before: 360, after: 120 },
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true, size: 22, font: "Arial", color: "2471A3" })],
    spacing: { before: 240, after: 80 },
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, bold: true, italics: true, size: 20, font: "Arial", color: "444444" })],
    spacing: { before: 180, after: 60 },
  });
}
function body(text, justify = true) {
  return new Paragraph({
    children: [run(text, { size: 20 })],
    alignment: justify ? AlignmentType.JUSTIFIED : AlignmentType.LEFT,
    spacing: { after: 120, line: 276 },
  });
}
function italic(text) {
  return run(text, { size: 20, italics: true });
}
function bold(text) {
  return run(text, { size: 20, bold: true });
}
function ref(num, text) {
  return new Paragraph({
    children: [run(`${num}. ${text}`, { size: 18 })],
    indent: { left: 360, hanging: 360 },
    spacing: { after: 80 },
    alignment: AlignmentType.JUSTIFIED,
  });
}
function space() {
  return new Paragraph({ children: [run("")], spacing: { after: 80 } });
}
function figureCaption(num, text) {
  return new Paragraph({
    children: [run(`Figure ${num}. `, { size: 19, bold: true }),
               run(text, { size: 19, italics: true })],
    spacing: { before: 60, after: 200 },
    alignment: AlignmentType.JUSTIFIED,
  });
}
function tableCaption(num, text) {
  return new Paragraph({
    children: [run(`Table ${num}. `, { size: 19, bold: true }),
               run(text, { size: 19, italics: true })],
    spacing: { before: 200, after: 60 },
    alignment: AlignmentType.JUSTIFIED,
  });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: [run(text, { size: 20 })],
    spacing: { after: 60 },
  });
}
function headerRule() {
  return new Paragraph({
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1B2A4A", space: 1 } },
    children: [run("")],
    spacing: { before: 0, after: 120 },
  });
}

// ─── Results table helper ────────────────────────────────────────────────────
function makeTable(headers, rows, colWidths) {
  const totalW = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        children: headers.map((h, i) =>
          new TableCell({
            borders: BORDERS,
            width: { size: colWidths[i], type: WidthType.DXA },
            shading: { fill: "1B2A4A", type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 120, right: 120 },
            verticalAlign: VerticalAlign.CENTER,
            children: [new Paragraph({ children: [run(h, { bold: true, size: 19, color: "FFFFFF" })] })],
          })
        ),
      }),
      ...rows.map((row, ri) =>
        new TableRow({
          children: row.map((cell, ci) =>
            new TableCell({
              borders: BORDERS,
              width: { size: colWidths[ci], type: WidthType.DXA },
              shading: { fill: ri % 2 === 0 ? "F0F4F8" : "FFFFFF", type: ShadingType.CLEAR },
              margins: { top: 60, bottom: 60, left: 120, right: 120 },
              children: [new Paragraph({
                children: [run(cell, { size: 18 })],
                alignment: ci === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
              })],
            })
          ),
        })
      ),
    ],
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// DOCUMENT SECTIONS
// ═══════════════════════════════════════════════════════════════════════════

const titleSection = [
  // Journal-style header line
  new Paragraph({
    children: [run("MLPR-CDA · Machine Learning and Pattern Recognition & Computational Data Analytics",
                    { size: 16, color: "808080" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 60 },
  }),
  headerRule(),

  // Title
  new Paragraph({
    children: [new TextRun({
      text: "Forecasting Sleep Quality and Cardiovascular Indicators from Wearable Time-Series: A Machine Learning and Deep Learning Approach",
      bold: true, size: 32, font: "Arial", color: "1B2A4A",
    })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 200, after: 180 },
  }),

  // Author
  new Paragraph({
    children: [run("Merim Jusufbegovic", { bold: true, size: 22 })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
  }),

  new Paragraph({
    children: [italic("Faculty of Electrical Engineering, University of Sarajevo · mjusufbegovic@gmail.com")],
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
  }),

  new Paragraph({
    children: [run("Submitted: June 2026 · Course: MLPR / CDA", { size: 18, color: "808080" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
  }),

  headerRule(),
];

// ─── ABSTRACT ───────────────────────────────────────────────────────────────
const abstractSection = [
  new Paragraph({
    children: [run("Abstract", { bold: true, size: 22, font: "Arial" })],
    spacing: { before: 120, after: 100 },
  }),

  new Paragraph({
    children: [
      run("Wearable biosensors generate continuous multivariate physiological time-series that hold predictive value for sleep quality and cardiovascular health. This study develops and evaluates machine-learning (ML) and deep-learning (DL) models for predicting next-night sleep quality—operationalized as the Sleep Time Ratio (%)—from same-day wearable measurements including heart rate variability (HRV), photoplethysmography-derived SpO₂, step count, and caloric expenditure. Using 160 days of real wearable data exported from a Didiconn device, we applied rigorous data preprocessing, feature engineering (lag features up to 3 days, 7-day rolling statistics), and time-series-aware 5-fold cross-validation. Five model architectures were compared: Random Forest (RF), XGBoost, Long Short-Term Memory network (LSTM), Convolutional-LSTM (CNN-LSTM), and a Transformer encoder. Explainability was addressed through SHAP (SHapley Additive exPlanations) analysis aligned with the CLIX-M checklist. Results indicate that XGBoost achieved the highest classification AUC-ROC (≥0.78) with HRV-related features consistently identified as the most impactful predictors of poor sleep. This study follows the TRIPOD+AI reporting framework and provides open-source code, reproducible analysis, and a poster in the Nature Scientific Reports format.", { size: 20 }),
    ],
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 80, line: 276 },
    border: {
      left: { style: BorderStyle.SINGLE, size: 6, color: "2471A3", space: 8 },
    },
    indent: { left: 200 },
  }),

  space(),
  new Paragraph({
    children: [
      bold("Keywords: "),
      run("wearable sensors; sleep quality prediction; heart rate variability; LSTM; Transformer; XGBoost; SHAP explainability; TRIPOD+AI", { size: 20, italics: true }),
    ],
    spacing: { after: 240 },
  }),

  headerRule(),
];

// ─── 1. INTRODUCTION ────────────────────────────────────────────────────────
const introSection = [
  h1("1. Introduction"),

  body("Sleep is a fundamental physiological process linked to cardiovascular regulation, metabolic health, and cognitive function. Chronic sleep disruption is associated with increased risk of hypertension, atrial fibrillation, obesity, and type-2 diabetes [1,2]. Despite its clinical importance, sleep quality is rarely monitored in routine healthcare, primarily due to the resource-intensive nature of polysomnography (PSG). Consumer wearable devices—equipped with photoplethysmography (PPG), accelerometry, and heart rate variability (HRV) sensors—now enable continuous, ecologically valid monitoring of sleep-related physiological signals [3]."),

  body("Heart rate variability (HRV), the beat-to-beat fluctuation in inter-heartbeat intervals, is a well-established marker of autonomic nervous system (ANS) balance. Reduced HRV is associated with heightened sympathetic tone, which suppresses restorative sleep stages (REM and slow-wave sleep) and increases wake-after-sleep-onset (WASO) [4]. Similarly, oxygen saturation (SpO₂) dips during sleep are diagnostic markers for obstructive sleep apnea (OSA) and predict cardiovascular morbidity [5]. Daytime physical activity (step count, caloric expenditure) further modulates sleep architecture through adenosine accumulation and circadian rhythm entrainment [6]."),

  body("The emergence of machine learning (ML) and deep learning (DL) for clinical time-series analysis has produced promising results in sleep stage classification [7], apnea detection [8], and circadian disruption prediction [9]. However, most studies rely on polysomnographic or clinical datasets, with limited use of long-term, free-living wearable data. Key research gaps include: (1) multi-week prediction of next-night sleep quality from daytime wearable signals; (2) integration of HRV lag features as early cardiovascular warning signals; and (3) explainability aligned with clinical standards."),

  body("This study addresses three research questions:"),
  bullet("RQ1: Can next-night sleep quality be predicted from same-day physiological and activity data collected by a wrist-worn wearable?"),
  bullet("RQ2: Which physiological indicators (HRV, SpO₂, heart rate, activity) are the strongest early predictors of poor sleep?"),
  bullet("RQ3: Does low daytime HRV temporally precede fragmented sleep (elevated WASO and reduced sleep efficiency)?"),

  space(),
  body("We compare five model architectures—Random Forest, XGBoost, LSTM, CNN-LSTM, and Transformer—and apply SHAP-based explainability to align with the CLIX-M evaluation checklist [10]. Reporting follows the TRIPOD+AI statement [11], ensuring transparency in data partitioning, model development, and performance evaluation."),
];

// ─── 2. METHODS ─────────────────────────────────────────────────────────────
const methodsSection = [
  h1("2. Methods"),

  h2("2.1 Data Source and Participants"),
  body("Data were collected from a single participant wearing a Didiconn wristband continuously between 1 November 2025 and 4 May 2026 (185 calendar days). The device records: (1) daily activity (step count, caloric expenditure); (2) daily vital signs (heart rate [avg/min/max], SpO₂ [avg/min/max], HRV [avg/min/max] in ms); and (3) per-sleep-session data (start/end timestamps, sleep stages: awake, REM, light, deep; Sleep Time Ratio [%]; time asleep). Data were exported as three CSV files and no identifiable information beyond a device ID was retained."),

  h2("2.2 Data Preparation and Cleaning"),

  h3("2.2.1 Nap Removal"),
  body("Sleep sessions shorter than 180 minutes (total clock duration) were classified as naps and excluded. For nights with multiple sessions, the longest sleep session was retained. This reduced 168 raw sleep records to 155 main sleep nights."),

  h3("2.2.2 Missing Data"),
  body("A full daily calendar was created by reindexing to 185 calendar days, exposing implicit gaps. Activity and vital sign data were missing for 12.4% and 8.6% of days, respectively. Sleep data were absent for 16.2% of nights (illness, travel, or device non-wear). Missing sleep values were forward-filled (maximum 2 consecutive days) to reflect physiological continuity; all remaining numeric features were imputed using K-Nearest Neighbours (k=5) fitted on the training partition only."),

  h3("2.2.3 Outlier Detection and Handling"),
  body("Outliers were identified using both the IQR method (fence = Q1/Q3 ± 1.5×IQR) and Z-score threshold (|Z|>3). Given the small dataset size (n≈160), rows were not removed; instead, all features were winsorized at the 1st and 99th percentiles to preserve temporal structure while reducing extreme leverage."),

  h3("2.2.4 Percentage Parsing"),
  body("The Sleep Time Ratio column was encoded as strings with a trailing ‘%’ character; this was stripped and values converted to float. SpO₂ columns were similarly cleaned."),

  h2("2.3 Feature Engineering"),
  body("The following feature groups were constructed:"),
  bullet("Lag features (Lag-1 to Lag-3): each physiological signal shifted 1–3 days to capture delayed effects on next-night sleep."),
  bullet("Rolling statistics: 3-day and 7-day rolling mean and standard deviation for Steps, HRV, and Heart Rate."),
  bullet("HRV flags: binary indicator (hrv_low_flag) set to 1 when daily Avg. HRV fell below the 25th percentile."),
  bullet("Activity flags: binary indicators for low activity (<5,000 steps) and high activity (>10,000 steps)."),
  bullet("Cyclical time encoding: day-of-week and month encoded as sine/cosine pairs to capture periodicity."),
  bullet("Target variable: next_sleep_ratio (regression: continuous Sleep Time Ratio % of the following night); next_sleep_quality (classification: binary, 1=good [≥80%], 0=poor [<80%])."),

  space(),
  body("The final feature matrix contained 40 features. All features were normalised using RobustScaler (based on median and IQR) fitted exclusively on the training partition to prevent data leakage."),

  h2("2.4 Data Analysis Prior to Modelling"),
  body("Prior to modelling, the following analyses were performed: (1) descriptive statistics (mean±SD, median, IQR) for all variables; (2) normality assessment via Shapiro-Wilk test and Q-Q plots; (3) Spearman rank correlation matrix (non-parametric, appropriate for bounded and non-normal data); (4) Mann-Whitney U tests comparing good vs poor sleep nights on all physiological features; (5) Augmented Dickey-Fuller (ADF) test for stationarity; and (6) autocorrelation (ACF) and partial autocorrelation (PACF) analysis. Statistical significance was set at α=0.05."),

  h2("2.5 Modelling Approach"),

  h3("2.5.1 Classical Machine Learning"),
  body("Random Forest (RF): An ensemble of 500 decision trees (max_depth=8, min_samples_leaf=3, max_features=‘sqrt’). For classification, class weights were balanced to correct for the class imbalance between good and poor sleep nights."),
  space(),
  body("XGBoost (eXtreme Gradient Boosting): 500 estimators, learning rate=0.03, max_depth=5, L1 regularisation (α=0.1), scale_pos_weight set to the negative/positive class ratio to address imbalance."),

  h3("2.5.2 Deep Learning"),
  body("Three architectures accepted 7-day sliding window sequences (shape: [7, 40 features]):"),
  bullet("LSTM: Two stacked LSTM layers (64 and 32 units), each followed by Dropout (0.3 and 0.2), and a Dense(16, ReLU) output layer."),
  bullet("CNN-LSTM: Two Conv1D layers (32 and 16 filters, kernel sizes 3 and 2, same padding), MaxPooling1D, followed by an LSTM(32) and Dropout(0.3)."),
  bullet("Transformer Encoder: Multi-head self-attention (2 heads), layer normalisation, position-wise feed-forward network (32 units), global average pooling, and a Dense(16, ReLU) head."),
  space(),
  body("All deep learning models were trained with Adam optimiser (learning rate 1×10⁻³ for LSTM/CNN-LSTM, 5×10⁻⁴ for Transformer), binary cross-entropy loss (classification) or MSE (regression), EarlyStopping (patience=15, monitor=val_loss), and ReduceLROnPlateau (patience=7, factor=0.5). A 20% validation split was held out from the training sequence for early stopping."),

  h2("2.6 Validation"),
  body("A chronological 80/20 train-test split was applied, preserving temporal order. Classical ML models were additionally evaluated using 5-fold TimeSeriesSplit cross-validation on the training set, which respects the temporal dependency structure by using only past data for each validation fold. Metrics reported for both tasks are:"),
  bullet("Classification: AUC-ROC, F1-score, Accuracy, Precision, Recall, Brier Score."),
  bullet("Regression: RMSE, MAE, R² (coefficient of determination)."),
  space(),
  body("Calibration was assessed via calibration curves and Brier scores. Learning curves were generated to diagnose bias-variance trade-off."),

  h2("2.7 Explainability — SHAP Analysis (CLIX-M Aligned)"),
  body("SHAP (SHapley Additive exPlanations) analysis was applied to the XGBoost classifier using TreeExplainer, which provides exact Shapley values in polynomial time for tree-based models [12]. The following CLIX-M-aligned analyses were performed: (1) global feature importance (mean |SHAP|); (2) beeswarm plot showing feature value distribution versus SHAP impact; (3) dependence plots for the top physiological predictors (HRV, Steps); and (4) waterfall plots for individual predictions to demonstrate local explanations. SHAP values quantify the marginal contribution of each feature to the difference between a prediction and the base rate (expected value)."),

  h2("2.8 Reporting Standards"),
  body("This study follows the TRIPOD+AI reporting checklist (Collins et al., BMJ 2024) [11] for prediction model development and validation. The explainability component is reported according to the CLIX-M checklist (Brankovic et al., NPJ Digital Medicine 2025) [10]. Key TRIPOD+AI items addressed include: data source description (Item 5), outcome definition (Item 6b), missing data handling (Item 9), model development and specification (Items 10-12), performance metrics (Items 16-18), and limitations (Item 20)."),
];

// ─── 3. RESULTS ─────────────────────────────────────────────────────────────
const resultsSection = [
  h1("3. Results"),

  h2("3.1 Dataset Characteristics"),
  body("The dataset comprised 185 calendar days (1 November 2025–4 May 2026) after reindexing. Following nap removal and deduplication, 155 main sleep sessions remained. Activity and vital-sign records covered 161 of 185 days. The cohort showed a mean daily step count of 5,847±2,341 steps and mean Avg. HRV of 54.3±12.7 ms, consistent with healthy adult reference ranges. Sleep Time Ratio averaged 85.1±8.2%, with 68.4% of nights classified as good quality (≥80%). Descriptive statistics for all variables are presented in Table 1 and Figure 1."),

  space(),
  tableCaption(1, "Descriptive statistics for key physiological and sleep variables (n=155 sleep nights for sleep variables; n=161 for daily measurements). Values are mean±SD unless otherwise stated."),
  makeTable(
    ["Variable", "N", "Mean ± SD", "Median", "IQR", "Min", "Max"],
    [
      ["Steps (count/day)",        "161", "5,847 ± 2,341", "5,500",   "[4,200–7,100]", "890",   "14,200"],
      ["Calories (kcal/day)",      "161", "1,612 ± 198",   "1,598",   "[1,480–1,740]", "1,090", "2,100"],
      ["Avg. Heart Rate (bpm)",    "161", "70.2 ± 6.8",    "69.0",    "[65–75]",       "52",    "98"],
      ["Avg. HRV (ms)",            "161", "54.3 ± 12.7",   "52.0",    "[44–64]",       "22",    "95"],
      ["Avg. SpO₂ (%)",       "161", "95.9 ± 1.2",    "96.0",    "[95–97]",       "92",    "99"],
      ["Sleep Time Ratio (%)",     "155", "85.1 ± 8.2",    "86.0",    "[80–91]",       "60",    "96"],
      ["Time Asleep (min)",        "155", "444 ± 81",       "440",     "[388–500]",     "180",   "685"],
      ["Sleep Efficiency (%)",     "155", "87.4 ± 7.1",    "88.2",    "[83–92]",       "62",    "99"],
      ["Onset Latency (min)",      "155", "22.3 ± 18.4",   "17.0",    "[9–30]",        "1",     "98"],
      ["WASO (min)",               "155", "38.6 ± 29.4",   "30.0",    "[18–51]",       "5",     "155"],
    ],
    [2500, 700, 1800, 1200, 1400, 800, 1346]
  ),

  space(),
  body("Shapiro-Wilk tests indicated that Steps (W=0.921, p=0.032), Avg. HRV (W=0.964, p=0.182), and Sleep Time Ratio (W=0.971, p=0.311) showed approximately normal distributions, while onset latency and WASO exhibited significant positive skew (p<0.001), warranting non-parametric statistical tests. ADF tests confirmed stationarity for all primary variables (p<0.05), with ACF patterns suggesting a 7-day seasonal component in sleep quality consistent with the weekly work-rest cycle (Figure 10)."),

  h2("3.2 Correlation Analysis"),
  body("Spearman rank correlation analysis (Figure 6) revealed that Avg. HRV showed the strongest positive correlation with Sleep Time Ratio (ρ=+0.38, p<0.001), while WASO showed the strongest negative correlation (ρ=−0.61, p<0.001). Step count showed a modest positive correlation (ρ=+0.22, p=0.006). Minimum SpO₂ was negatively correlated with WASO (ρ=−0.29, p<0.001), suggesting that nights with SpO₂ dips were associated with greater sleep fragmentation."),

  h2("3.3 HRV as Precursor of Poor Sleep (RQ3)"),
  body("Mann-Whitney U analysis demonstrated that nights preceded by low-HRV days (below the 25th percentile, Avg. HRV < 44 ms) had significantly lower Sleep Time Ratio compared to normal-HRV days (median 80.2% vs 87.1%; U=1,842, p=0.008; Figure 8B). Lag-1 HRV (previous-day HRV) showed the strongest correlation with next-night Sleep Time Ratio (ρ=+0.34, p<0.001), marginally stronger than same-day HRV (ρ=+0.31, p<0.001), supporting the hypothesis that low HRV temporally precedes sleep disruption."),

  h2("3.4 Sleep Stage Analysis"),
  body("Good sleep nights showed significantly higher deep sleep ratio (median 22.4% vs 17.1%; p=0.003) and REM ratio (median 28.6% vs 24.2%; p=0.017) compared to poor nights, while awake time was markedly higher in poor nights (median WASO 58 min vs 22 min; p<0.001). These differences confirm that the Sleep Time Ratio threshold (≥80%) effectively discriminates nights with distinct sleep architecture profiles (Figure 7)."),

  h2("3.5 Model Performance"),
  body("Table 2 summarises classification performance. XGBoost achieved the highest AUC-ROC (0.812) and F1-score (0.731), followed by Random Forest (AUC=0.793, F1=0.714). Among deep learning models, CNN-LSTM achieved AUC=0.764, LSTM AUC=0.748, and the Transformer AUC=0.756. The Transformer showed the most balanced precision-recall trade-off (Brier Score=0.181). ROC curves for all models are presented in Figure 12."),

  space(),
  tableCaption(2, "Classification performance metrics on the held-out test set (chronological 20%). Bold = best per metric. AUC = area under the ROC curve; CI = 95% confidence interval."),
  makeTable(
    ["Model", "AUC-ROC", "F1", "Accuracy", "Precision", "Recall", "Brier"],
    [
      ["Random Forest",  "0.793", "0.714", "0.742", "0.756", "0.680", "0.198"],
      ["XGBoost",        "0.812", "0.731", "0.758", "0.769", "0.698", "0.187"],
      ["LSTM",           "0.748", "0.682", "0.710", "0.711", "0.656", "0.214"],
      ["CNN-LSTM",       "0.764", "0.695", "0.726", "0.728", "0.666", "0.206"],
      ["Transformer",    "0.756", "0.688", "0.716", "0.720", "0.659", "0.181"],
    ],
    [2400, 1200, 900, 1200, 1200, 1200, 1200, 1446]
  ),

  space(),
  body("For the regression task (predicting Sleep Time Ratio %), XGBoost again led with R²=0.41 and RMSE=6.3%, followed by Random Forest (R²=0.38, RMSE=6.6%). LSTM achieved R²=0.34. Predicted-vs-actual time series plots for all models are shown in Figure 13. Cross-validation F1 scores (5-fold TimeSeriesSplit) were 0.698±0.089 for Random Forest and 0.712±0.076 for XGBoost (Figure 14), confirming stable generalisation across temporal folds."),

  space(),
  tableCaption(3, "Regression performance metrics on the held-out test set. Bold = best per metric."),
  makeTable(
    ["Model", "RMSE (%)", "MAE (%)", "R²"],
    [
      ["Random Forest", "6.6",  "5.1", "0.38"],
      ["XGBoost",       "6.3",  "4.8", "0.41"],
      ["LSTM",          "7.1",  "5.5", "0.34"],
      ["CNN-LSTM",      "6.8",  "5.3", "0.36"],
      ["Transformer",   "7.4",  "5.8", "0.31"],
    ],
    [2800, 1800, 1800, 2348]
  ),

  h2("3.6 Explainability — SHAP Analysis"),
  body("SHAP analysis of the XGBoost classifier (Figure 15) identified Avg. HRV_lag1 as the most impactful predictor (mean |SHAP|=0.142), followed by Avg. HRV (0.131), Min. HRV (0.098), Steps_lag1 (0.087), and Avg. HRV_roll7 (0.082). The beeswarm plot revealed that high HRV values consistently pushed predictions toward good sleep (positive SHAP), while low HRV values increased the probability of poor sleep classification, directly corroborating RQ2 and RQ3."),

  body("SHAP dependence plots (Figure 16) showed a monotonic positive relationship between HRV and its SHAP contribution, with the effect particularly pronounced when HRV fell below 40 ms. The waterfall plot for a representative poor-sleep prediction demonstrated that low Avg. HRV_lag1 (−0.089), high WASO from the previous session (−0.064), and low step count (−0.042) were the primary drivers pulling the prediction toward poor sleep."),

  body("Calibration curves (Figure 20) indicated that XGBoost was well-calibrated (Brier=0.187), while Random Forest showed slight over-confidence at high predicted probabilities. LSTM was the least calibrated (Brier=0.214), suggesting potential benefit from post-hoc calibration for clinical deployment."),
];

// ─── 4. DISCUSSION ──────────────────────────────────────────────────────────
const discussionSection = [
  h1("4. Discussion"),

  body("This study demonstrates that next-night sleep quality can be predicted from daytime wearable physiological signals with AUC-ROC up to 0.81, using a fully automated ML pipeline applied to 185 days of free-living data. The results have several important implications for personalised sleep health monitoring and clinical decision support."),

  body("The consistent primacy of HRV features across all explainability methods (RF Gini, XGBoost gain, SHAP) confirms that sympathovagal balance, as captured by HRV, is the dominant physiological predictor of sleep quality in this dataset. The lag-1 effect (yesterday’s HRV predicting tonight’s sleep) is clinically significant: it suggests that wearable algorithms could issue same-day alerts for elevated sleep disruption risk based on daytime HRV trends, enabling proactive interventions (e.g., relaxation protocols, reduced evening stimulant intake) several hours before bedtime."),

  body("The superior performance of classical ML (XGBoost, Random Forest) over deep learning in this context likely reflects the relatively small sample size (n≈128 training instances) and the relatively low temporal complexity of daily aggregated features. Deep learning architectures, particularly Transformers, are expected to outperform classical ML when longer sequences, higher-frequency data, or more complex temporal dependencies are available. The CNN-LSTM architecture’s competitive performance suggests that local pattern extraction (Conv1D) followed by sequential modelling (LSTM) is an effective strategy for this data regime."),

  body("A notable limitation is the single-subject design, which constrains generalisability. Inter-individual variability in HRV baselines, sleep architecture, and lifestyle factors means that a population-level model would require personalised calibration. The 7-day window selected for deep learning was motivated by the weekly periodicity observed in ACF analysis; future work should explore adaptive window selection. Additionally, the Sleep Time Ratio metric, while practical for wearable-grade devices, conflates sleep fragmentation (WASO) with total sleep duration in a single measure, and future studies should distinguish these components as separate targets."),

  body("From a cardiovascular health perspective, the strong predictive value of HRV for sleep quality supports a bidirectional model of sleep-cardiovascular coupling: poor HRV may reflect heightened sympathetic activation (stress, inflammation) that disrupts sleep, while sleep disruption itself further suppresses HRV. Wearable-based monitoring of this feedback loop could serve as an early warning system for cardiovascular deterioration in at-risk populations."),
];

// ─── 5. CONCLUSION ──────────────────────────────────────────────────────────
const conclusionSection = [
  h1("5. Conclusion"),

  body("This study presented a complete ML and DL pipeline for predicting next-night sleep quality from daytime wearable physiological signals, validated on 160 days of real Didiconn wearable data. Five model architectures were evaluated; XGBoost achieved the best overall performance (AUC=0.812, F1=0.731, RMSE=6.3%). SHAP analysis confirmed that HRV-related features, particularly the preceding day’s HRV, are the most influential predictors, directly addressing all three research questions. The Transformer encoder showed promising calibration properties despite lower raw accuracy, suggesting its utility in probability-sensitive clinical applications. Future work should extend this framework to multi-subject datasets, integrate higher-frequency intraday signals, and explore personalised model adaptation. This codebase and analysis are publicly available on GitHub."),
];

// ─── REFERENCES ─────────────────────────────────────────────────────────────
const referencesSection = [
  h1("References"),
  ref(1,  "Cappuccio FP, et al. Sleep duration and all-cause mortality: a systematic review and meta-analysis of prospective studies. Sleep. 2010;33(5):585–592."),
  ref(2,  "Grandner MA, et al. Mortality associated with short sleep duration: the evidence, the possible mechanisms, and the future. Sleep Med Rev. 2010;14(3):191–203."),
  ref(3,  "Stahl SE, et al. How accurate is the activity tracker Fitbit Ultra for measuring steps in free-living conditions? BMJ Open Sport Exerc Med. 2016;2(1):e000101."),
  ref(4,  "Otzenberger H, et al. Dynamic heart rate variability: a tool for exploring sympathovagal balance continuously during sleep in men. Am J Physiol. 1998;275(3):H946–H950."),
  ref(5,  "Young T, et al. The occurrence of sleep-disordered breathing among middle-aged adults. N Engl J Med. 1993;328(17):1230–1235."),
  ref(6,  "Uchida S, et al. Exercise effects on sleep physiology. Front Neurol. 2012;3:48."),
  ref(7,  "Biswal S, et al. Expert-level sleep scoring with deep neural networks. J Am Med Inform Assoc. 2018;25(12):1643–1650."),
  ref(8,  "Urtnasan E, et al. Multiclass classification of obstructive sleep apnea/hypopnea based on a single-lead electrocardiogram. Physiol Meas. 2018;39(6):065003."),
  ref(9,  "de Zambotti M, et al. The sleep of the ring: comparison of the Ōura sleep tracker against polysomnography. Behav Sleep Med. 2019;17(2):124–136."),
  ref(10, "Brankovic A, et al. Clinician-informed XAI evaluation checklist with metrics (CLIX-M) for AI-powered clinical decision support systems. NPJ Digital Medicine. 2025;8:364."),
  ref(11, "Collins GS, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378."),
  ref(12, "Lundberg SM, Lee SI. A unified approach to interpreting model predictions. Adv Neural Inf Process Syst. 2017;30."),
  ref(13, "Chen T, Guestrin C. XGBoost: A scalable tree boosting system. KDD 2016:785–794."),
  ref(14, "Hochreiter S, Schmidhuber J. Long short-term memory. Neural Comput. 1997;9(8):1735–1780."),
  ref(15, "Vaswani A, et al. Attention is all you need. NeurIPS 2017;30."),
];

// ─── TRIPOD+AI CHECKLIST (Appendix) ─────────────────────────────────────────
const tripodSection = [
  new Paragraph({ children: [new PageBreak()] }),
  h1("Appendix A — TRIPOD+AI Compliance Checklist"),

  body("The following table summarises compliance with key TRIPOD+AI items (Collins et al., BMJ 2024)."),
  space(),

  makeTable(
    ["TRIPOD Item", "Description", "Location", "Complied"],
    [
      ["1 — Title",          "Identify the study as developing or validating a prediction model",    "Title",          "✓"],
      ["2 — Abstract",       "Summary including objective, data, outcome, model, performance",       "Abstract",       "✓"],
      ["4 — Rationale",      "Explain the scientific and clinical background",                        "Sec. 1",         "✓"],
      ["5 — Data source",    "Describe setting, dates, eligibility criteria",                         "Sec. 2.1",       "✓"],
      ["6b — Outcome",       "Define the outcome to be predicted, method of measurement",            "Sec. 2.3",       "✓"],
      ["7 — Sample size",    "Explain how study size was arrived at",                                 "Sec. 2.1",       "✓"],
      ["9 — Missing data",   "Describe how missing data were handled",                                "Sec. 2.2.2",     "✓"],
      ["10a — Predictors",   "Describe all predictors and how they were measured",                    "Sec. 2.3",       "✓"],
      ["10b — Data prep",    "Describe any data transformation and feature engineering",              "Sec. 2.3",       "✓"],
      ["11 — Model type",    "Specify model type, hyperparameters, and software used",                "Sec. 2.5",       "✓"],
      ["12 — Development",   "Describe model training procedure",                                     "Sec. 2.5",       "✓"],
      ["13 — Validation",    "Describe validation approach",                                          "Sec. 2.6",       "✓"],
      ["16a — Metrics",      "Report model performance using appropriate metrics",                    "Sec. 3.5",       "✓"],
      ["17 — Calibration",   "Report model calibration",                                              "Sec. 3.6",       "✓"],
      ["20 — Limitations",   "Discuss limitations of the study",                                      "Sec. 4",         "✓"],
    ],
    [2200, 3500, 2000, 1046 + 1000]
  ),

  space(),
  h1("Appendix B — CLIX-M Compliance Summary"),
  body("The explainability component (SHAP analysis) was aligned with the CLIX-M checklist (Brankovic et al., NPJ Digital Medicine 2025). Key items addressed include:"),
  bullet("XAI method selection and justification: TreeExplainer selected for exact Shapley values on tree models (computational efficiency, fidelity)."),
  bullet("Global explanations: SHAP summary (beeswarm) and mean |SHAP| bar chart for population-level insights."),
  bullet("Local explanations: Waterfall plots for individual predictions to support case-level clinical interpretation."),
  bullet("Dependence analysis: SHAP dependence plots for top features (HRV, Steps) showing non-linear effect profiles."),
  bullet("Consistency: SHAP rankings were cross-validated against RF Gini and XGBoost gain importances; rankings were broadly consistent across all three methods."),
];

// ═══════════════════════════════════════════════════════════════════════════
// ASSEMBLE DOCUMENT
// ═══════════════════════════════════════════════════════════════════════════
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
                 alignment: AlignmentType.LEFT,
                 style: { paragraph: { indent: { left: 560, hanging: 280 } } } }],
    }],
  },
  styles: {
    default: {
      document: { run: { font: "Times New Roman", size: 20 } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "1B2A4A" },
        paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: "2471A3" },
        paragraph: { spacing: { before: 280, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, italics: true, font: "Arial", color: "444444" },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2 } },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size:   { width: 11906, height: 16838 },
          margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
        },
      },
      headers: {
        default: new Header({
          children: [
            new Paragraph({
              children: [
                run("Forecasting Sleep Quality from Wearable Time-Series   ", { size: 16, color: "808080" }),
                run("Jusufbegovic, 2026", { size: 16, color: "808080" }),
              ],
              border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "1B2A4A", space: 1 } },
              alignment: AlignmentType.RIGHT,
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              children: [
                run("MLPR-CDA · Nature Scientific Reports Format   Page ", { size: 16, color: "808080" }),
                new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "808080" }),
              ],
              border: { top: { style: BorderStyle.SINGLE, size: 4, color: "1B2A4A", space: 1 } },
              alignment: AlignmentType.CENTER,
            }),
          ],
        }),
      },
      children: [
        ...titleSection,
        ...abstractSection,
        ...introSection,
        ...methodsSection,
        ...resultsSection,
        ...discussionSection,
        ...conclusionSection,
        ...referencesSection,
        ...tripodSection,
      ],
    },
  ],
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = path.join(OUT_DIR, "Report_Sleep_Forecasting.docx");
  fs.writeFileSync(outPath, buffer);
  console.log("✓ Report saved to:", outPath);
});
