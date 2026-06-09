"""
step2_train.py — Extract features and train the model
Run: python step2_train.py   (expect 5–15 minutes)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Loads the URLs downloaded in Step 1
2. Converts every URL into numbers (feature extraction)
3. Splits data into train / validation / test sets
4. Trains 3 different ML models and compares them
5. Finds the best decision threshold (fixes the 30% problem)
6. Saves the best model to model/phish_model.pkl

LESSON — Why 3 splits (train / validation / test)?
  • Train      — model LEARNS from this (70%)
  • Validation — we TUNE the threshold on this (15%). Model hasn't seen it.
  • Test        — FINAL honest score (15%). Neither model nor threshold has seen it.
  Using only 1 split gives you an over-optimistic score. This 3-way split
  gives you a score you can honestly put on your resume.

LESSON — Why compare 3 models?
  No algorithm is always best. We let the data decide which wins.
  Logistic Regression → simple baseline, trains in seconds
  Random Forest       → ensemble of decision trees, usually good
  Gradient Boosting   → builds trees sequentially, often the winner

LESSON — What is threshold tuning and why does it fix the 30% problem?
  Every model outputs a probability (0.0 – 1.0).
  By default we say: if probability > 0.5 → phishing.
  But 0.5 is arbitrary! Maybe 0.35 catches more phishing, or 0.6
  reduces false positives on legitimate sites. We find the threshold
  that maximises F1 score on the validation set.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from features import extract_features

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_DIR  = 'data'
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)


# ── FEATURE EXTRACTION ─────────────────────────────────────────────────────────

def extract_all(urls: list) -> pd.DataFrame:
    """
    Run extract_features() on every URL and collect results into a DataFrame.
    Prints progress every 1,000 URLs.
    """
    print(f"  Extracting features from {len(urls):,} URLs...")
    rows = []
    total = len(urls)
    for i, url in enumerate(urls):
        if i > 0 and i % 1000 == 0:
            pct = round(i / total * 100)
            print(f"  Progress: {i:>6,}/{total:,}  ({pct}%)")
        rows.append(extract_features(str(url)))
    print(f"  Done. {len(rows):,} feature vectors created.")
    return pd.DataFrame(rows)


# ── MODEL EVALUATION ───────────────────────────────────────────────────────────

def print_eval(name: str, model, X: pd.DataFrame, y: np.ndarray, threshold: float) -> float:
    """Print a full evaluation report and return the ROC AUC score."""
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    auc   = roc_auc_score(y, proba)

    print(f"\n{'─'*58}")
    print(f"  {name}  |  threshold={threshold:.3f}")
    print(f"{'─'*58}")
    print(classification_report(y, preds, target_names=['Legit', 'Phishing']))
    print(f"  ROC AUC: {auc:.4f}")

    # Confusion matrix broken down into plain English
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    print(f"\n  Confusion matrix breakdown:")
    print(f"    Phishing correctly blocked:     {tp:>6,}  ✓")
    print(f"    Phishing slipped through:       {fn:>6,}  ← want low")
    print(f"    Legit sites falsely blocked:    {fp:>6,}  ← want low")
    print(f"    Legit sites correctly allowed:  {tn:>6,}  ✓")
    return auc


# ── THRESHOLD TUNING ───────────────────────────────────────────────────────────

def tune_threshold(model, X_val: pd.DataFrame, y_val: np.ndarray) -> float:
    """
    Find the probability threshold that maximises F1 on the validation set.

    F1 = 2 × (Precision × Recall) / (Precision + Recall)
    It balances:
      Precision — "when we say phishing, are we right?"
      Recall    — "do we catch most of the actual phishing?"

    This is the key fix for false positives on legitimate modern websites.
    """
    proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, proba)

    # F1 at each threshold
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1[:-1]))  # last point has no threshold
    best_t   = float(thresholds[best_idx])

    # Compare default vs tuned
    def score(t):
        preds = (proba >= t).astype(int)
        return round((preds == y_val).mean() * 100, 1)

    print(f"\n  Threshold tuning on validation set:")
    print(f"    Default (0.500): {score(0.5):>5}% accuracy")
    print(f"    Tuned   ({best_t:.3f}): {score(best_t):>5}% accuracy  ← improvement")
    print(f"    Best F1 score: {f1[best_idx]:.4f}  |  "
          f"Precision: {precision[best_idx]:.3f}  |  Recall: {recall[best_idx]:.3f}")

    return best_t


# ── FEATURE IMPORTANCE ─────────────────────────────────────────────────────────

def show_feature_importance(model, feature_names: list):
    """Print a simple bar chart of the top 10 most important features."""
    clf = model
    if hasattr(model, 'named_steps'):           # unwrap Pipeline
        clf = model.named_steps.get('clf', model)

    if not hasattr(clf, 'feature_importances_'):
        return

    imp = pd.DataFrame({
        'feature':    feature_names,
        'importance': clf.feature_importances_,
    }).sort_values('importance', ascending=False)

    print("\n  Top 10 features the model relies on most:")
    print(f"  {'Feature':<30} Importance")
    print(f"  {'─'*50}")
    for _, row in imp.head(10).iterrows():
        bar  = '█' * int(row['importance'] * 300)
        print(f"  {row['feature']:<30} {bar:<20} {row['importance']:.4f}")

    imp.to_csv(os.path.join(MODEL_DIR, 'feature_importance.csv'), index=False)
    print(f"\n  Full importance table saved to {MODEL_DIR}/feature_importance.csv")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  STEP 2 — Training the phishing detection model")
    print("=" * 58)

    # ── Load data ────────────────────────────────────────────────────────────
    raw_path = os.path.join(DATA_DIR, 'urls_raw.csv')
    if not os.path.exists(raw_path):
        print(f"\nERROR: {raw_path} not found. Run step1_get_data.py first.")
        return

    df = pd.read_csv(raw_path)
    print(f"\nLoaded {len(df):,} URLs")
    print(df['label'].value_counts().rename({1: 'phishing', 0: 'legit'}).to_string())

    # ── Extract features ─────────────────────────────────────────────────────
    print("\nExtracting features from all URLs...")
    print("(This is the slowest part — grab a coffee)\n")
    X = extract_all(df['url'].tolist())
    y = df['label'].values

    # Replace NaN (should only be domain_age_days in Tier 2) with -1
    X = X.fillna(-1)

    # Save features so you can retrain quickly without re-extracting
    feat_path = os.path.join(DATA_DIR, 'features.csv')
    X.to_csv(feat_path, index=False)
    print(f"\nFeatures saved to {feat_path}")
    print(f"Features per URL: {X.shape[1]}")
    print(f"Feature names: {list(X.columns)}")

    # ── 3-way train/validation/test split ────────────────────────────────────
    # stratify=y ensures phishing/legit ratio is preserved in every split
    print("\nSplitting data: 70% train / 15% validation / 15% test")
    X_tv, X_test,  y_tv, y_test  = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.176, random_state=42, stratify=y_tv
    )  # 0.176 × 0.85 ≈ 0.15 of total

    print(f"  Train:      {len(X_train):,}")
    print(f"  Validation: {len(X_val):,}")
    print(f"  Test:       {len(X_test):,}")

    # ── Define 3 models ──────────────────────────────────────────────────────
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),      # LR needs normalised features
            ('clf',    LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            random_state=42, n_jobs=-1        # n_jobs=-1 uses all CPU cores
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            subsample=0.8, random_state=42
        ),
    }

    # ── Train and evaluate all models ────────────────────────────────────────
    print("\nTraining all 3 models (comparing at default threshold=0.5)...")
    results = {}

    for name, model in models.items():
        print(f"\n  Training: {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = round(time.time() - t0, 1)
        print(f"  Trained in {elapsed}s")
        auc = print_eval(name, model, X_test, y_test, threshold=0.5)
        results[name] = {'model': model, 'auc': auc}

    # ── Pick winner ──────────────────────────────────────────────────────────
    best_name  = max(results, key=lambda k: results[k]['auc'])
    best_model = results[best_name]['model']

    print(f"\n{'='*58}")
    print(f"  Winner: {best_name}  (AUC: {results[best_name]['auc']:.4f})")
    print(f"{'='*58}")

    # ── Tune threshold ───────────────────────────────────────────────────────
    print(f"\nTuning decision threshold for {best_name}...")
    best_threshold = tune_threshold(best_model, X_val, y_val)

    # ── Final honest evaluation ──────────────────────────────────────────────
    print(f"\nFINAL EVALUATION — {best_name} with tuned threshold:")
    final_auc = print_eval(best_name, best_model, X_test, y_test, best_threshold)

    # ── Feature importance ───────────────────────────────────────────────────
    show_feature_importance(best_model, list(X.columns))

    # ── Save model and metadata ──────────────────────────────────────────────
    # We save the metadata alongside the model so prediction code always
    # uses the correct threshold and feature names — no guessing needed.
    metadata = {
        'model_name':    best_name,
        'threshold':     best_threshold,
        'feature_names': list(X.columns),
        'auc':           final_auc,
        'n_training':    len(X_train),
    }

    joblib.dump(best_model, os.path.join(MODEL_DIR, 'phish_model.pkl'))
    joblib.dump(metadata,   os.path.join(MODEL_DIR, 'metadata.pkl'))

    print(f"\nModel saved:    {MODEL_DIR}/phish_model.pkl")
    print(f"Metadata saved: {MODEL_DIR}/metadata.pkl")
    print(f"\nDone! Next step: python step3_test.py")


if __name__ == '__main__':
    main()
