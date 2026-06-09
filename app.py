"""
app.py — ThreatLens Flask Backend
Run: python app.py
Then open: http://localhost:5000

API Endpoints:
  POST /api/scan    — scan a URL, returns prediction + confidence
  GET  /api/stats   — overall stats (total scanned, phishing %, model info)
  GET  /api/recent  — last 20 scan results
"""

import os
import sqlite3
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from features import extract_features

app  = Flask(__name__)
CORS(app)

# ── Trusted domain whitelist ───────────────────────────────────────────────────
# These domains bypass the ML model and are always classified as LEGIT.
# This is standard practice in production security tools — ML + whitelist together.
# Add any domain you know is legitimate that the model keeps misclassifying.
TRUSTED_DOMAINS = {
    # Malaysian banks
    'maybank.com', 'maybank2u.com.my', 'maybank2e.com', 'etiqa.com.my',
    'cimb.com.my', 'cimbclicks.com.my', 'cimbbank.com',
    'rhbgroup.com', 'rhbgroup.com.my', 'rhbonline.com.my',
    'publicbank.com.my', 'pbebank.com', 'pbenterprise.com.my',
    'hlb.com.my', 'hongleongbank.com',
    'affinbank.com.my', 'affinhwangam.com',
    'bankislam.com.my', 'bsn.com.my',
    'ambank.com.my', 'ambankgroup.com',
    'tnb.com.my', 'mytnb.com.my',
    # Global tech
    'google.com', 'youtube.com', 'gmail.com', 'google.com.my',
    'microsoft.com', 'office.com', 'live.com', 'outlook.com',
    'apple.com', 'icloud.com',
    'github.com', 'gitlab.com',
    'stackoverflow.com', 'linkedin.com',
    'facebook.com', 'instagram.com', 'twitter.com',
    'amazon.com', 'amazon.com.my',
    # Malaysian sites
    'gov.my', 'hasil.gov.my', 'lhdn.gov.my', 'perkeso.gov.my',
    'kwsp.gov.my', 'epf.gov.my', 'jhev.gov.my',
    'thestar.com.my', 'malaymail.com', 'bernama.com',
    'mudah.my', 'lazada.com.my', 'shopee.com.my',
    'grab.com', 'airasia.com', 'petronas.com.my',
}

def is_trusted(url: str) -> bool:
    """Return True if the URL belongs to a known trusted domain."""
    try:
        import tldextract
        ext = tldextract.extract(url)
        registered = ext.registered_domain.lower()
        full = (ext.subdomain + '.' + registered).lstrip('.').lower()
        # Match exact registered domain OR any subdomain of a trusted domain
        return any(
            registered == t or registered.endswith('.' + t) or t == full
            for t in TRUSTED_DOMAINS
        )
    except Exception:
        return False

# ── Load model once at startup ─────────────────────────────────────────────────
print("Loading PhishProof model...")
model    = joblib.load('model/phish_model.pkl')
metadata = joblib.load('model/metadata.pkl')
print(f"Model: {metadata['model_name']}  |  AUC: {metadata['auc']:.4f}  |  Threshold: {metadata['threshold']:.3f}")

# ── Database setup ─────────────────────────────────────────────────────────────
DB_PATH = os.path.join('data', 'threats.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS scans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                url         TEXT    NOT NULL,
                prediction  TEXT    NOT NULL,
                confidence  REAL    NOT NULL,
                risk        TEXT    NOT NULL,
                is_phishing INTEGER NOT NULL,
                scanned_at  TEXT    NOT NULL
            )
        ''')
        conn.commit()

# ── Prediction logic ───────────────────────────────────────────────────────────
def run_prediction(url: str) -> dict:
    feats     = extract_features(url)
    X         = pd.DataFrame([feats])[metadata['feature_names']].fillna(-1)
    proba     = float(model.predict_proba(X)[0][1])
    threshold = metadata['threshold']
    is_phish  = proba >= threshold

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
        'scanned_at':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the React dashboard."""
    return render_template('index.html')


@app.route('/api/scan', methods=['POST'])
def scan():
    """
    Scan a URL and return the prediction.
    Body: { "url": "https://example.com" }
    """
    body = request.get_json(silent=True) or {}
    url  = body.get('url', '').strip()

    if not url:
        return jsonify({'error': 'URL is required'}), 400

    # Check whitelist first — trusted domains skip the ML model entirely
    if is_trusted(url):
        result = {
            'url':         url,
            'prediction':  'LEGIT',
            'confidence':  0.0,
            'risk':        'LOW',
            'is_phishing': False,
            'scanned_at':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'note':        'whitelisted domain'
        }
    else:
        result = run_prediction(url)

    # Store in database
    with get_db() as conn:
        conn.execute(
            '''INSERT INTO scans
               (url, prediction, confidence, risk, is_phishing, scanned_at)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (result['url'], result['prediction'], result['confidence'],
             result['risk'], int(result['is_phishing']), result['scanned_at'])
        )
        conn.commit()

    return jsonify(result)


@app.route('/api/stats')
def stats():
    """Return overall platform statistics."""
    with get_db() as conn:
        total    = conn.execute('SELECT COUNT(*) FROM scans').fetchone()[0]
        phishing = conn.execute('SELECT COUNT(*) FROM scans WHERE is_phishing=1').fetchone()[0]
        legit    = conn.execute('SELECT COUNT(*) FROM scans WHERE is_phishing=0').fetchone()[0]
        by_risk  = {
            row['risk']: row['cnt']
            for row in conn.execute(
                'SELECT risk, COUNT(*) as cnt FROM scans WHERE is_phishing=1 GROUP BY risk'
            ).fetchall()
        }

    return jsonify({
        'total':      total,
        'phishing':   phishing,
        'legit':      legit,
        'by_risk':    by_risk,
        'model_name': metadata['model_name'],
        'auc':        round(metadata['auc'], 4),
        'threshold':  round(metadata['threshold'], 3),
    })


@app.route('/api/recent')
def recent():
    """Return the last N scan results."""
    limit = request.args.get('limit', 20, type=int)
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM scans ORDER BY id DESC LIMIT ?', (limit,)
        ).fetchall()
    return jsonify([dict(row) for row in rows])


# ── Entry point ────────────────────────────────────────────────────────────────
os.makedirs('data', exist_ok=True)
init_db()   # always runs — works for both 'python app.py' and gunicorn

if __name__ == '__main__':
    print('\n  ThreatLens is running → http://localhost:5000\n')
    app.run(debug=True, port=5000)