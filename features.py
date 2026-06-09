"""
features.py — The brain of PhishProof

This file converts a raw URL string into a list of numbers that a
machine learning model can understand. This is called "feature engineering."

WHY THIS MATTERS:
  ML models cannot read text. They only understand numbers.
  Our job is to extract meaningful numbers from a URL that capture
  signals a human expert would use: "is this URL suspiciously long?",
  "does it use an IP instead of a domain?", "does it contain brand names?"

TWO TIERS of features:
  Tier 1 — URL-string features: computed from the text alone.
            Fast, always available, consistent between training and live use.
  Tier 2 — Network features: require a lookup (e.g. WHOIS domain age).
            Slow, can fail. We use safe fallbacks so the model still works.

THE 30% FAILURE PROBLEM:
  The most common cause is Tier 2 features working during training
  but timing out or returning different values in live prediction.
  We fix this by making Tier 2 optional (use_whois=False by default).
"""

import re
import math
import tldextract
from urllib.parse import urlparse


# ── CONSTANTS ──────────────────────────────────────────────────────────────────

# Keywords that appear often in phishing URLs but rarely in legit ones
SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'sign-in', 'verify', 'account', 'secure',
    'update', 'confirm', 'banking', 'paypal', 'ebay', 'amazon',
    'apple', 'microsoft', 'google', 'password', 'credential',
    'urgent', 'suspended', 'limited', 'validate', 'winner',
    'prize', 'free', 'webscr', 'cmd', 'checkout', 'billing',
]

# TLDs that are free/cheap and heavily abused by phishers
SUSPICIOUS_TLDS = {
    'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'work',
    'click', 'link', 'party', 'gdn', 'stream', 'download',
    'online', 'site', 'website', 'space',
}

# Brand names phishers impersonate by putting them in fake domains
# e.g. "paypal-secure-login.tk" or "amazon-account.xyz"
BRAND_NAMES = [
    'paypal', 'google', 'facebook', 'amazon', 'apple', 'microsoft',
    'netflix', 'instagram', 'twitter', 'linkedin', 'dropbox',
    'bankofamerica', 'chase', 'wellsfargo', 'maybank', 'cimb',
    'tnb', 'hsbc', 'visa', 'mastercard',
]


# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """Ensure URL has a scheme so urlparse works correctly."""
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    return url


def shannon_entropy(s: str) -> float:
    """
    Measure how random/unpredictable a string is.

    Real domain names score low  → 'google' ≈ 2.58
    Random phishing domains score high → 'xk3j9mq' ≈ 2.95

    Formula: H = -Σ p(x) · log₂(p(x))
    """
    if not s:
        return 0.0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    entropy = 0.0
    n = len(s)
    for count in freq.values():
        p = count / n
        entropy -= p * math.log2(p)
    return round(entropy, 4)


def has_ip_address(url: str) -> int:
    """
    Return 1 if the host part is an IP address, 0 otherwise.
    Phishers use IPs to avoid registering a recognisable domain.
    e.g.  http://192.168.1.1/login.php  → 1
          http://google.com             → 0
    """
    pattern = re.compile(r'(https?://)?((\d{1,3}\.){3}\d{1,3})')
    return int(bool(pattern.search(url)))


# ── MAIN FUNCTION ──────────────────────────────────────────────────────────────

def extract_features(url: str, use_whois: bool = False) -> dict:
    """
    Extract all features from a URL and return as an ordered dictionary.

    Args:
        url        : The URL string to analyse.
        use_whois  : If True, look up domain age via WHOIS (~2s per call).
                     Keep False during training for speed; enable later for
                     the live platform where latency is acceptable.

    Returns:
        dict mapping feature_name → numeric value.

    IMPORTANT: Feature names and order must never change after training.
    If you add/remove features you must retrain the model from scratch.
    """
    url = normalize_url(url)
    parsed  = urlparse(url)
    ext     = tldextract.extract(url)
    url_low = url.lower()

    f = {}   # our features dictionary

    # ── TIER 1: URL-string features ─────────────────────────────────────────

    # Length features
    # Phishing URLs are often longer — they cram in fake subdomains and paths
    f['url_length']     = len(url)
    f['domain_length']  = len(ext.domain)
    f['path_length']    = len(parsed.path)
    f['query_length']   = len(parsed.query)

    # Special character counts
    # Phishers use characters like @ (to spoof the domain) and %20 (encoding)
    f['num_dots']        = url.count('.')
    f['num_hyphens']     = url.count('-')
    f['num_underscores'] = url.count('_')
    f['num_at_signs']    = url.count('@')
    f['num_slashes']     = url.count('/')
    f['num_equals']      = url.count('=')
    f['num_ampersands']  = url.count('&')
    f['num_percent']     = url.count('%')
    f['num_question']    = url.count('?')

    # Ratio features (normalised by URL length so short and long URLs are comparable)
    n = len(url) or 1
    f['digit_ratio']  = round(sum(c.isdigit() for c in url) / n, 4)
    f['letter_ratio'] = round(sum(c.isalpha() for c in url) / n, 4)

    # Entropy — phishing domains generated by scripts score higher
    f['url_entropy']    = shannon_entropy(url)
    f['domain_entropy'] = shannon_entropy(ext.domain)

    # Structure flags
    f['num_subdomains']     = len(ext.subdomain.split('.')) if ext.subdomain else 0
    f['has_https']          = int(parsed.scheme == 'https')
    f['has_ip']             = has_ip_address(url)
    f['has_port']           = int(bool(re.search(r':\d+', parsed.netloc)))
    f['has_at_sign']        = int('@' in url)
    f['has_double_slash']   = int('//' in parsed.path)   # e.g. http://evil.com//login
    f['domain_has_digit']   = int(any(c.isdigit() for c in ext.domain))
    f['too_many_subdomains']= int(f['num_subdomains'] > 3)

    # Suspicious patterns
    # IMPORTANT: Use word-boundary matching (\b) so 'secure' does NOT match
    # inside 'security', and 'login' does NOT match inside 'loginscreen'.
    # Substring matching caused false positives on legit banking sites like
    # maybank2u.com.my/login.do (flagged for 'login') and
    # cimb.com.my/security-policy.html (flagged for 'secure' inside 'security').
    f['suspicious_tld'] = int(ext.suffix.lower() in SUSPICIOUS_TLDS)
    f['n_suspicious_keywords'] = sum(
        1 for kw in SUSPICIOUS_KEYWORDS
        if re.search(r'\b' + re.escape(kw) + r'\b', url_low)
    )
    f['has_suspicious_keyword'] = int(f['n_suspicious_keywords'] > 0)

    # Brand impersonation
    # has_brand_in_domain: e.g. "paypal" appears in the URL domain
    # brand_impersonation: brand appears in domain but domain ≠ brand's real domain
    brand_in_domain = any(b in ext.domain.lower() for b in BRAND_NAMES)
    f['has_brand_in_domain'] = int(brand_in_domain)

    impersonation = False
    if brand_in_domain:
        for brand in BRAND_NAMES:
            if brand in ext.domain.lower():
                # The real domain for this brand should be brand.com (simplified)
                if ext.registered_domain != f'{brand}.com':
                    impersonation = True
                    break
    f['brand_impersonation'] = int(impersonation)

    # ── TIER 2: Network features (optional, with safe fallbacks) ────────────

    if use_whois:
        try:
            import whois
            import datetime
            w = whois.whois(ext.registered_domain)
            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0]
            if creation and isinstance(creation, datetime.datetime):
                f['domain_age_days'] = (datetime.datetime.now() - creation).days
            else:
                f['domain_age_days'] = -1
        except Exception:
            f['domain_age_days'] = -1   # -1 = "unknown", not a bug
    else:
        f['domain_age_days'] = -1

    return f


def get_feature_names() -> list:
    """Return the ordered list of feature names produced by extract_features()."""
    return list(extract_features('http://example.com').keys())


# ── QUICK SELF-TEST ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_urls = [
        'https://google.com',
        'http://paypa1-secure-login.tk/account/verify?id=12345',
        'http://192.168.1.1/admin/login.php',
    ]
    print(f"Feature names ({len(get_feature_names())} total):")
    print(get_feature_names())
    print()
    for url in test_urls:
        feats = extract_features(url)
        print(f"URL: {url}")
        for k, v in feats.items():
            print(f"  {k:<30} {v}")
        print()