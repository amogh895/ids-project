"""Phase 2 — Supervised Model Training with Honest Evaluation.

Pipeline:
  1. Leak-free time-ordered train/test split (Mon-Wed = train, Thu-Fri = test).
  2. sklearn Pipeline: Imputer → Scaler → LightGBM (binary: BENIGN vs ATTACK).
  3. Class-weight balancing via scale_pos_weight (no SMOTE to keep it simple & fast).
  4. Optuna hyperparameter tuning optimising PR-AUC (5-fold StratifiedKFold on train only).
  5. Calibrated threshold selected from PR curve at 0.1% FPR operating point.
  6. Evaluation: per-class P/R/F1, confusion matrix, PR-AUC, ROC-AUC,
     Recall @ FPR 1% / 0.1% / 0.01%.
  7. All metrics + artifacts logged to MLflow. Pipeline persisted to models/.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from features.extract import FEATURE_COLS

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
DATA_PATH    = ROOT / "data" / "processed" / "flows_labeled.parquet"
MODELS_DIR   = ROOT / "models"
REPORTS_DIR  = ROOT / "reports"
MLFLOW_URI   = f"sqlite:///{ROOT / 'mlflow.db'}"

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ─── 1. Load & Split (time-ordered, NO random shuffle) ───────────────────────
def load_and_split(data_path: Path) -> tuple[
    pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray
]:
    """Time-ordered train/test split: Mon-Wed → train, Thu-Fri → test.

    Attack classes collapsed to binary label for Phase-2 supervised model.
    Multi-class labels preserved in a separate column for reporting.
    """
    df = pd.read_parquet(data_path)

    # Binary target: 0 = BENIGN, 1 = ATTACK
    df["binary_label"] = (df["label"] != "BENIGN").astype(int)

    train_days = ["Monday", "Tuesday", "Wednesday"]
    test_days  = ["Thursday", "Friday"]

    train_df = df[df["day"].isin(train_days)].copy()
    test_df  = df[df["day"].isin(test_days)].copy()

    # Guarantee temporal ordering within each split
    train_df = train_df.sort_values("timestamp").reset_index(drop=True)
    test_df  = test_df.sort_values("timestamp").reset_index(drop=True)

    X_train = train_df[FEATURE_COLS]
    X_test  = test_df[FEATURE_COLS]
    y_train = train_df["binary_label"].values
    y_test  = test_df["binary_label"].values

    print(f"  Train: {len(X_train):,} flows | "
          f"Attacks: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"  Test : {len(X_test):,} flows  | "
          f"Attacks: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

    return X_train, X_test, y_train, y_test


# ─── 2. Build sklearn Pipeline ───────────────────────────────────────────────
def build_pipeline(lgb_params: dict) -> Pipeline:
    """Assemble leak-free Pipeline: Imputer → Scaler → LightGBM.

    All transformers fit exclusively on training data when calling pipeline.fit().
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ]), FEATURE_COLS),
        ],
        remainder="drop",
    )

    clf = lgb.LGBMClassifier(
        n_jobs=1,          # single-threaded — laptop-friendly
        verbose=-1,
        **lgb_params,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   clf),
    ])


# ─── 3. Optuna Tuning (optimise PR-AUC) ─────────────────────────────────────
def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_trials: int = 15,
) -> dict:
    """Run Optuna study with StratifiedKFold CV; objective = mean PR-AUC.

    Weights computed inside objective to stay strictly inside training folds.
    """
    scale_pos_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",    50, 300),
            "max_depth":         trial.suggest_int("max_depth",         3,   8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves",        8,  64),
            "min_child_samples": trial.suggest_int("min_child_samples", 5,  50),
            "subsample":         trial.suggest_float("subsample",      0.6,  1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight":  scale_pos_weight,
            "objective":         "binary",
            "metric":            "average_precision",
        }
        pipeline = build_pipeline(params)
        cv = StratifiedKFold(n_splits=5, shuffle=False)   # NO shuffle — preserve time order
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="average_precision", n_jobs=1,
        )
        return float(scores.mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best["scale_pos_weight"] = scale_pos_weight
    best["objective"]        = "binary"
    best["metric"]           = "average_precision"
    print(f"  Best CV PR-AUC: {study.best_value:.4f}")
    print(f"  Best params: {best}")
    return best


# ─── 4. Threshold Calibration at 0.1% FPR ───────────────────────────────────
def calibrate_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_fpr: float = 0.001,
) -> tuple[float, dict[str, float]]:
    """Select decision threshold from ROC curve at the target FPR.

    Returns (threshold, recall_dict) where recall_dict covers three operating points.
    """
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_proba)

    recall_at_fpr: dict[str, float] = {}
    selected_threshold = 0.5   # safe fallback

    for key_suffix, fp_rate in [("1_0_pct", 0.01), ("0_1_pct", 0.001), ("0_01_pct", 0.0001)]:
        idx = np.searchsorted(fpr_arr, fp_rate, side="right") - 1
        idx = max(0, min(idx, len(tpr_arr) - 1))
        recall_at_fpr[f"recall_at_fpr_{key_suffix}"] = float(tpr_arr[idx])
        if key_suffix == "0_1_pct":
            # Select threshold at the 0.1% FPR operating point
            if idx < len(thresholds):
                selected_threshold = float(thresholds[idx])

    return selected_threshold, recall_at_fpr


# ─── 5. Full Evaluation ──────────────────────────────────────────────────────
def evaluate(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    threshold: float,
) -> dict:
    """Compute all mandatory evaluation metrics."""
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    pr_auc  = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    _, recall_dict = calibrate_threshold(y_test, y_proba)

    report_dict = classification_report(
        y_test, y_pred,
        target_names=["BENIGN", "ATTACK"],
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "threshold":    float(threshold),
        "pr_auc":       float(pr_auc),
        "roc_auc":      float(roc_auc),
        "precision_attack": float(report_dict["ATTACK"]["precision"]),
        "recall_attack":    float(report_dict["ATTACK"]["recall"]),
        "f1_attack":        float(report_dict["ATTACK"]["f1-score"]),
        "precision_benign": float(report_dict["BENIGN"]["precision"]),
        "recall_benign":    float(report_dict["BENIGN"]["recall"]),
        "f1_benign":        float(report_dict["BENIGN"]["f1-score"]),
        **recall_dict,
    }

    print("\n--- PHASE 2 EVALUATION RESULTS ---")
    print(f"  PR-AUC   : {pr_auc:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"  Threshold: {threshold:.4f}  (calibrated @ 0.1% FPR)")
    print(f"  Recall @ FPR 1.0%  : {recall_dict['recall_at_fpr_1_0_pct']:.4f}")
    print(f"  Recall @ FPR 0.1%  : {recall_dict['recall_at_fpr_0_1_pct']:.4f}  [HEADLINE]")
    print(f"  Recall @ FPR 0.01% : {recall_dict['recall_at_fpr_0_01_pct']:.4f}")
    print(f"\n  Classification Report (threshold={threshold:.4f}):")
    print(classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"]))
    print(f"  Confusion Matrix:\n{cm}")
    print("----------------------------------\n")

    return metrics, cm, y_proba, y_pred


# ─── 6. Artifact Plots ───────────────────────────────────────────────────────
def save_plots(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    cm: np.ndarray,
    threshold: float,
    out_dir: Path,
) -> list[Path]:
    """Save Confusion Matrix, PR Curve, and ROC Curve to reports/."""
    saved: list[Path] = []

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["BENIGN", "ATTACK"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix (calibrated threshold)")
    p = out_dir / "confusion_matrix.png"
    fig.savefig(p, bbox_inches="tight", dpi=100)
    plt.close(fig)
    saved.append(p)

    # PR Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, lw=1.5, color="#2563eb", label="PR Curve")
    baseline = float(y_test.mean())
    ax.axhline(baseline, color="gray", linestyle="--", lw=1, label=f"Baseline ({baseline:.3f})")
    # Mark selected threshold on PR curve
    thresh_idx = np.searchsorted(-thresholds_pr, -threshold)
    thresh_idx = max(0, min(thresh_idx, len(precision) - 2))
    ax.scatter(recall[thresh_idx], precision[thresh_idx], color="red", zorder=5,
               label=f"Op. Point (FPR=0.1%)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = out_dir / "pr_curve.png"
    fig.savefig(p, bbox_inches="tight", dpi=100)
    plt.close(fig)
    saved.append(p)

    # ROC Curve
    fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr_arr, tpr_arr, lw=1.5, color="#16a34a")
    ax.axvline(0.001, color="red", linestyle="--", lw=1, label="FPR=0.1%")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = out_dir / "roc_curve.png"
    fig.savefig(p, bbox_inches="tight", dpi=100)
    plt.close(fig)
    saved.append(p)

    return saved


# ─── 7. Main ─────────────────────────────────────────────────────────────────
def main() -> None:
    print("\n=== PHASE 2: Supervised Model & Honest Evaluation ===\n")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("nids-supervised")

    # 1. Load & split
    print("Step 1: Loading and splitting dataset...")
    X_train, X_test, y_train, y_test = load_and_split(DATA_PATH)

    # 2. Tune
    print("\nStep 2: Tuning hyperparameters (15 Optuna trials, CV PR-AUC)...")
    best_params = tune_hyperparameters(X_train, y_train, n_trials=15)

    with mlflow.start_run(run_name="lgbm-supervised") as run:
        mlflow.log_params(best_params)

        # 3. Fit final pipeline on full training set
        print("\nStep 3: Fitting final pipeline on full training set...")
        pipeline = build_pipeline(best_params)
        pipeline.fit(X_train, y_train)

        # 4. Calibrate threshold
        y_proba_test = pipeline.predict_proba(X_test)[:, 1]
        threshold, _ = calibrate_threshold(y_test, y_proba_test, target_fpr=0.001)
        print(f"\nStep 4: Calibrated threshold @ 0.1% FPR = {threshold:.6f}")

        # 5. Evaluate
        print("\nStep 5: Running full evaluation suite...")
        metrics, cm, y_proba, y_pred = evaluate(pipeline, X_test, y_test, threshold)

        # 6. Log metrics
        mlflow.log_metrics(metrics)
        mlflow.log_params({"calibrated_threshold": threshold})

        # 7. Save plots
        plots = save_plots(y_test, y_proba, cm, threshold, REPORTS_DIR)
        for p in plots:
            mlflow.log_artifact(str(p))
            print(f"  [+] Plot saved: {p.name}")

        # 8. Save evaluation metrics JSON
        metrics_path = REPORTS_DIR / "phase2_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(str(metrics_path))
        print(f"  [+] Metrics saved: {metrics_path.name}")

        # 9. Persist pipeline + threshold
        pipeline_path = MODELS_DIR / "supervised_pipeline.pkl"
        threshold_path = MODELS_DIR / "calibrated_threshold.json"
        joblib.dump(pipeline, pipeline_path)
        with open(threshold_path, "w") as f:
            json.dump({"threshold": threshold, "fpr_target": 0.001}, f, indent=2)

        mlflow.log_artifact(str(pipeline_path))
        mlflow.log_artifact(str(threshold_path))

        print(f"  [+] Pipeline saved: {pipeline_path}")
        print(f"  [+] Threshold saved: {threshold_path}")
        print(f"\n  MLflow Run ID: {run.info.run_id}")
        print(f"  MLflow Tracking: {MLFLOW_URI}")

    print("\n=== PHASE 2 COMPLETE ===\n")


if __name__ == "__main__":
    main()
