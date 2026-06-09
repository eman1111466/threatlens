"""
step3_test.py — Test the model on any URL
Run: python step3_test.py                         (demo mode)
Run: python step3_test.py https://suspicious.com  (single URL)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loads the model saved in Step 2 and predicts whether a URL is
phishing or legitimate, with a confidence score and risk level.

This file also becomes the prediction engine for Phase 2
(the Threat Intelligence Platform). The Flask API will import
the predict() function directly.

LESSON — Why do we save threshold + feature names in metadata.pkl?
  If we hardcoded these values here, we'd have to manually update them
  every time we retrain. By loading from metadata.pkl, this file works
  correctly regardless of what threshold or features were chosen during training.
  This is called "keeping training and serving in sync."
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys
import os
import joblib
import pandas as pd
from features import extract_features

MODEL_DIR = 'model'


# ── LOAD ───────────────────────────────────────────────────────────────────────

def load_model():
    """Load the trained model and its metadata from disk."""
    model_path = os.path.join(MODEL_DIR, 'phish_model.pkl')
    meta_path  = os.path.join(MODEL_DIR, 'metadata.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model found at {model_path}. Run step2_train.py first."
        )

    model    = joblib.load(model_path)
    metadata = joblib.load(meta_path)
    return model, metadata


# ── PREDICT ────────────────────────────────────────────────────────────────────

def predict(url: str, model, metadata: dict) -> dict:
    """
    Classify a single URL as phishing or legit.

    This function is designed to be imported by the Flask API in Phase 2:
        from step3_test import predict, load_model

    Returns:
        {
            'url':         the input URL
            'prediction':  'PHISHING' or 'LEGIT'
            'confidence':  probability × 100 (0–100 %)
            'risk':        'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW'
            'is_phishing': True | False
            'features':    dict of extracted feature values (useful for Phase 2 dashboard)
        }
    """
    feats = extract_features(url)

    # Build DataFrame with columns in the exact order used during training
    X = pd.DataFrame([feats])[metadata['feature_names']]
    X = X.fillna(-1)

    # Model outputs a probability 0.0–1.0 (not just 0/1)
    proba     = float(model.predict_proba(X)[0][1])
    threshold = metadata['threshold']
    is_phish  = proba >= threshold

    # Map probability to a human-readable risk level
    if   proba >= 0.85: risk = 'CRITICAL'
    elif proba >= 0.65: risk = 'HIGH'
    elif proba >= 0.40: risk = 'MEDIUM'
    else:               risk = 'LOW'

    return {
        'url':         url,
        'prediction':  'PHISHING' if is_phish else 'LEGIT',
        'confidence':  round(proba * 100, 1),
        'risk':        risk,
        'is_phishing': is_phish,
        'features':    feats,   # Phase 2 will display these in the dashboard
    }


# ── DISPLAY ────────────────────────────────────────────────────────────────────

RISK_ICONS = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}

def print_result(r: dict, expected: str = None):
    icon      = RISK_ICONS.get(r['risk'], '⚪')
    url_short = r['url'][:65] + '...' if len(r['url']) > 65 else r['url']
    verdict   = r['prediction']

    print(f"\n  {icon} [{r['risk']:<8}] {verdict:<10} {r['confidence']:5.1f}%")
    print(f"     {url_short}")

    if expected:
        match  = verdict.lower() == expected.lower()
        marker = '✓ correct' if match else '✗ WRONG'
        print(f"     Expected: {expected.upper()}  →  {marker}")


# ── DEMO MODE ──────────────────────────────────────────────────────────────────

TEST_URLS = [
    # URL                                                              expected
    ('https://google.com',                                            'legit'),
    ('https://github.com/torvalds/linux',                             'legit'),
    ('https://stackoverflow.com/questions/tagged/python',             'legit'),
    ('https://amazon.com/dp/B08N5KWB9H',                              'legit'),
    ('https://maybank2u.com.my/m2u/common/login.do',                  'legit'),
    ('http://paypa1-secure-login.tk/account/verify?id=abc123',        'phishing'),
    ('http://192.168.1.1/admin/secure-login.php',                     'phishing'),
    ('http://amazon-account-suspended.ml/signin/verify-identity',     'phishing'),
    ('http://apple-id-locked.gq/verify-identity/confirm-now',         'phishing'),
    ('http://secure-banking-update-required.xyz/login?session=xyz',   'phishing'),
]


def demo_mode(model, metadata):
    print(f"\n{'='*62}")
    print(f"  PhishProof — Demo  |  {len(TEST_URLS)} test URLs")
    print(f"  Model: {metadata['model_name']:<22} AUC: {metadata['auc']:.4f}")
    print(f"  Threshold: {metadata['threshold']:.3f}  (trained on {metadata['n_training']:,} URLs)")
    print(f"{'='*62}")

    correct = 0
    for url, expected in TEST_URLS:
        r = predict(url, model, metadata)
        print_result(r, expected)
        if r['prediction'].lower() == expected:
            correct += 1

    total = len(TEST_URLS)
    pct   = round(correct / total * 100)
    print(f"\n  Score: {correct}/{total}  ({pct}%)")
    print(f"\n  Tip: run  python step3_test.py <url>  to test any URL.")


# ── SINGLE URL MODE ────────────────────────────────────────────────────────────

def single_mode(url: str, model, metadata):
    print(f"\n  Analysing: {url}")
    r = predict(url, model, metadata)
    print_result(r)

    # Show the most informative features that drove the prediction
    interesting = {
        'url_length': r['features'].get('url_length'),
        'has_https': r['features'].get('has_https'),
        'has_ip': r['features'].get('has_ip'),
        'suspicious_tld': r['features'].get('suspicious_tld'),
        'n_suspicious_keywords': r['features'].get('n_suspicious_keywords'),
        'brand_impersonation': r['features'].get('brand_impersonation'),
        'domain_entropy': r['features'].get('domain_entropy'),
        'num_subdomains': r['features'].get('num_subdomains'),
    }
    print(f"\n  Key features:")
    for feat, val in interesting.items():
        print(f"    {feat:<30} {val}")
    print()


# ── ENTRY POINT ────────────────────────────────────────────────────────────────

def main():
    print("Loading model...")
    model, metadata = load_model()

    if len(sys.argv) > 1:
        single_mode(sys.argv[1], model, metadata)
    else:
        demo_mode(model, metadata)


if __name__ == '__main__':
    main()
