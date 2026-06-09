"""
step1_get_data.py — Build training dataset
Run: python step1_get_data.py

This version works fully offline using an embedded dataset.
It also tries to download real data first — if that works, great.
If downloads are blocked, it uses the embedded dataset instead.

EMBEDDED DATASET:
  ~6,000 legit URLs  — 150 well-known domains × varied realistic paths
  ~6,000 phishing URLs — realistic synthetic patterns covering:
    typosquatting, suspicious TLDs, IP-based, brand impersonation,
    subdomain tricks, deceptive paths, both HTTP and HTTPS phishing
"""

import os
import io
import gzip
import random
import requests
import urllib3
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_DIR   = 'data'
MAX_URLS   = 6_000
SSL_VERIFY = False
HEADERS    = {'User-Agent': 'PhishProof-Research/1.0 (academic project)'}
os.makedirs(DATA_DIR, exist_ok=True)

random.seed(42)

# ── EMBEDDED LEGIT DOMAINS ──────────────────────────────────────────────────

LEGIT_DOMAINS = [
    # Search & Social
    'google.com','youtube.com','facebook.com','twitter.com','instagram.com',
    'linkedin.com','reddit.com','pinterest.com','tiktok.com','snapchat.com',
    'whatsapp.com','telegram.org','discord.com','twitch.tv','tumblr.com',
    # Tech & Dev
    'github.com','stackoverflow.com','gitlab.com','bitbucket.org','npmjs.com',
    'pypi.org','docs.python.org','developer.mozilla.org','w3schools.com',
    'css-tricks.com','dev.to','hashnode.com','medium.com','hackernoon.com',
    'digitalocean.com','heroku.com','netlify.com','vercel.com','cloudflare.com',
    # Shopping & E-commerce
    'amazon.com','ebay.com','etsy.com','shopify.com','aliexpress.com',
    'walmart.com','target.com','bestbuy.com','newegg.com','wayfair.com',
    # Finance
    'paypal.com','stripe.com','wise.com','revolut.com','squareup.com',
    'chase.com','bankofamerica.com','wellsfargo.com','citibank.com',
    # Streaming & Entertainment
    'netflix.com','spotify.com','apple.com','microsoft.com','adobe.com',
    'hulu.com','disneyplus.com','hbomax.com','primevideo.com','crunchyroll.com',
    # Productivity & Tools
    'dropbox.com','slack.com','zoom.us','notion.so','trello.com','asana.com',
    'atlassian.com','salesforce.com','hubspot.com','mailchimp.com',
    'google.com','drive.google.com','docs.google.com','sheets.google.com',
    # News & Reference
    'bbc.com','cnn.com','reuters.com','nytimes.com','theguardian.com',
    'apnews.com','bloomberg.com','forbes.com','techcrunch.com','wired.com',
    'wikipedia.org','britannica.com','investopedia.com','webmd.com',
    # Cloud & Infra
    'aws.amazon.com','azure.microsoft.com','cloud.google.com',
    'digitalocean.com','linode.com','vultr.com',
    # Malaysia-specific
    'maybank2u.com.my','cimbclicks.com.my','rhbgroup.com','publicbank.com.my',
    'tnb.com.my','thestar.com.my','malaymail.com','bernama.com',
    'mudah.my','lazada.com.my','shopee.com.my','grab.com','airasia.com',
    # Government & Education
    'gov.uk','usa.gov','canada.ca','who.int','un.org','europa.eu',
    'mit.edu','stanford.edu','harvard.edu','coursera.org','edx.org','udemy.com',
    # Security & Infra
    'cloudflare.com','letsencrypt.org','shodan.io','virustotal.com',
    'haveibeenpwned.com','mozilla.org','ubuntu.com','debian.org',
]

LEGIT_PATHS = [
    '',
    '/about', '/about-us', '/contact', '/help', '/faq', '/support',
    '/products', '/services', '/pricing', '/features', '/plans',
    '/blog', '/news', '/articles', '/press', '/resources',
    '/login', '/signin', '/register', '/signup', '/auth',
    '/dashboard', '/settings', '/profile', '/account', '/preferences',
    '/docs', '/documentation', '/api', '/developer', '/guides',
    '/search?q=python', '/search?q=security+tools', '/search?q=tutorial',
    '/search?q=how+to+install', '/search?q=best+practices',
    '/category/technology', '/category/news', '/category/finance',
    '/tag/python', '/tag/security', '/tag/webdev', '/tag/machine-learning',
    '/dp/B08N5KWB9H', '/dp/A1B2C3D4E5', '/dp/XR7K9M2W',
    '/questions/tagged/python', '/questions/12345/how-to-use-flask',
    '/questions/tagged/machine-learning', '/questions/98765/ssl-error-fix',
    '/torvalds/linux', '/django/django', '/python/cpython', '/keras-team/keras',
    '/2024/01/article-title', '/2024/03/blog-post-security',
    '/en/about', '/en/products', '/en/contact',
    '/product/laptop-pro-2024', '/product/wireless-headphones',
    '/user/johndoe', '/user/profile', '/user/settings',
    '/support/ticket/12345', '/support/faq/billing',
    '/m2u/common/login.do', '/ibanking/personal/login',
    '/index.html', '/home', '/main', '/overview',
    '?page=1', '?page=2', '?tab=overview', '?lang=en',
    '?ref=homepage', '?source=nav&medium=cpc',
    '/dp/B07XJ8C8F5?ref=sr_1_1', '/s?k=python+book',
    '/wiki/Cybersecurity', '/wiki/Machine_learning',
    '/en-us/docs/web/javascript', '/en-us/learn/python',
    '/download/latest', '/download/stable',
    '/releases/tag/v2.1.0', '/issues/1234',
    '/privacy', '/terms', '/cookies', '/legal',
    '/careers', '/jobs', '/team', '/investors',
]


def build_legit_urls() -> list:
    """
    Build legit URL list by combining domains with realistic paths.
    Each domain gets a different path so no two are identical.
    """
    urls = []
    n_paths = len(LEGIT_PATHS)
    for i, domain in enumerate(LEGIT_DOMAINS):
        # Give each domain several path variations
        for j in range(6):
            path = LEGIT_PATHS[(i * 6 + j) % n_paths]
            urls.append(f'https://{domain}{path}')
        # Add a www. variant
        urls.append(f'https://www.{domain}')
    return list(set(urls))


# ── EMBEDDED PHISHING PATTERNS ──────────────────────────────────────────────

PHISH_BRANDS   = ['paypal','amazon','apple','microsoft','google','facebook',
                  'netflix','instagram','twitter','linkedin','dropbox',
                  'bankofamerica','chase','wellsfargo','maybank','cimb',
                  'hsbc','visa','mastercard','ebay','steam','spotify']
PHISH_ACTIONS  = ['verify','login','signin','confirm','update','validate',
                  'secure','suspended','unlock','reset','recover','billing',
                  'account','identity','payment','access','authorize']
PHISH_TLDS     = ['tk','ml','ga','cf','gq','xyz','top','work','click',
                  'link','online','site','website','space','info','biz']
PHISH_WORDS    = ['secure','alert','urgent','limited','official','support',
                  'service','customer','portal','centre','center','now']


def build_phishing_urls(n: int = 6000) -> list:
    """
    Generate realistic phishing URL patterns.
    Covers the main techniques attackers actually use.
    """
    urls = set()
    rng  = random.Random(42)

    for i in range(n * 3):   # overshoot so we have enough after dedup
        brand  = rng.choice(PHISH_BRANDS)
        action = rng.choice(PHISH_ACTIONS)
        tld    = rng.choice(PHISH_TLDS)
        word   = rng.choice(PHISH_WORDS)
        token  = f'{rng.randint(1000,9999)}'
        scheme = rng.choice(['http', 'https'])   # phishing now uses HTTPS too
        ip     = f'192.168.{rng.randint(1,254)}.{rng.randint(1,254)}'

        pattern = rng.randint(0, 9)

        if pattern == 0:
            # Typosquatting: paypa1.com, arnazon.net
            typo = brand[:-1] + str(rng.randint(0,9))
            urls.add(f'{scheme}://{typo}.{tld}/{action}/{word}?id={token}')

        elif pattern == 1:
            # Brand-action domain: paypal-verify.tk
            urls.add(f'{scheme}://{brand}-{action}.{tld}/{word}/confirm?token={token}')

        elif pattern == 2:
            # Action-brand domain: verify-paypal-secure.xyz
            urls.add(f'{scheme}://{action}-{brand}-{word}.{tld}/account?ref={token}')

        elif pattern == 3:
            # Subdomain trick: paypal.com.evil.tk
            urls.add(f'{scheme}://{brand}.com.{word}-{action}.{tld}/login?id={token}')

        elif pattern == 4:
            # Deep deceptive path: /secure/account/verify/billing/confirm
            urls.add(f'{scheme}://{word}-{brand}-{action}.{tld}/secure/account/{action}/billing?token={token}')

        elif pattern == 5:
            # IP-based URL
            urls.add(f'http://{ip}/admin/{brand}/{action}.php?session={token}')

        elif pattern == 6:
            # Lookalike with hyphen noise
            urls.add(f'{scheme}://{brand}-{word}-{action}-now.{tld}/{action}?id={token}&step=2')

        elif pattern == 7:
            # URL encoding tricks (%20, %40)
            urls.add(f'{scheme}://{word}-{brand}.{tld}/{action}%20page?user={token}%40mail.com')

        elif pattern == 8:
            # Redirect/tracking style
            urls.add(f'{scheme}://{action}-{brand}.{tld}/redirect?url={brand}.com&token={token}')

        elif pattern == 9:
            # Long subdomain chain
            urls.add(f'{scheme}://{action}.{word}.{brand}-support.{tld}/account/billing?id={token}')

        if len(urls) >= n:
            break

    return list(urls)[:n]


# ── ONLINE DOWNLOAD ATTEMPTS (optional enhancement) ──────────────────────────

def try_download_phishing() -> list:
    """Try to get real phishing URLs from online feeds."""
    sources = [
        ('OpenPhish',  'https://openphish.com/feed.txt'),
        ('URLhaus',    'https://urlhaus.abuse.ch/downloads/text_recent/'),
    ]
    urls = []
    for name, url in sources:
        try:
            print(f'  {name}: downloading...')
            r = requests.get(url, headers=HEADERS, timeout=20, verify=SSL_VERIFY)
            r.raise_for_status()
            fetched = [l.strip() for l in r.text.splitlines()
                       if l.strip().startswith('http') and not l.startswith('#')]
            print(f'  {name}: {len(fetched):,} URLs')
            urls.extend(fetched)
        except Exception as e:
            print(f'  {name}: blocked or unavailable ({e.__class__.__name__})')
    return urls


def try_download_legit() -> list:
    """Try to get real legit domain lists from online sources."""
    sources = [
        ('Majestic Million',
         'https://downloads.majestic.com/majestic_million.csv', 'majestic'),
        ('Cisco Umbrella',
         'https://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.gz', 'umbrella'),
    ]
    for name, url, fmt in sources:
        try:
            print(f'  {name}: downloading...')
            r = requests.get(url, headers=HEADERS, timeout=60, verify=SSL_VERIFY)
            r.raise_for_status()
            if fmt == 'majestic':
                df = pd.read_csv(io.StringIO(r.text))
                domains = df['Domain'].dropna().head(MAX_URLS * 2).tolist()
            else:
                content = gzip.decompress(r.content).decode('utf-8', errors='ignore')
                df = pd.read_csv(io.StringIO(content), header=None, names=['rank','domain'])
                domains = df['domain'].dropna().head(MAX_URLS * 2).tolist()
            urls = []
            for i, d in enumerate(domains):
                path = LEGIT_PATHS[i % len(LEGIT_PATHS)]
                urls.append(f'https://{d}{path}')
            print(f'  {name}: {len(urls):,} URLs')
            return urls
        except Exception as e:
            print(f'  {name}: blocked or unavailable ({e.__class__.__name__})')
    return []


# ── LOCAL FILE READERS ────────────────────────────────────────────────────────

def read_urlhaus_local() -> list:
    """
    Read phishing URLs from a locally saved URLhaus CSV file.
    Place the file at: data/urlhaus.csv
    Download it from: https://urlhaus.abuse.ch/downloads/csv_recent/

    URLhaus format: ALL lines start with # including the header line.
    Header looks like: # id,dateadded,url,url_status,...
    Data lines look like: "3861479","2026-06-09",...
    """
    path = os.path.join(DATA_DIR, 'urlhaus.csv')
    if not os.path.exists(path):
        return []
    print(f'  Found data/urlhaus.csv — reading...')

    header = None
    data_lines = []

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('"'):
                # Actual data row — starts with a quoted ID like "3861479"
                data_lines.append(line)
            elif line.startswith('# id,'):
                # Header line — strip the leading "# " to get column names
                header = line[2:].strip()

    if not data_lines:
        print(f'  No data rows found in urlhaus.csv')
        return []

    # Reconstruct a clean CSV with proper header + data
    col_names = header if header else 'id,dateadded,url,url_status,last_online,threat,tags,urlhaus_link,reporter'
    csv_text  = col_names + '\n' + ''.join(data_lines)
    df   = pd.read_csv(io.StringIO(csv_text))
    urls = df['url'].dropna().tolist()
    urls = [u for u in urls if str(u).startswith('http')]
    print(f'  URLhaus local: {len(urls):,} real phishing URLs')
    return urls


def read_majestic_local() -> list:
    """
    Read legit domains from a locally saved Majestic Million CSV.
    Place the file at: data/majestic.csv
    Download it from: https://majestic.com/reports/majestic-million

    Columns include: GlobalRank, Domain, TLD, ...
    We use the 'Domain' column and add varied realistic paths.
    """
    path = os.path.join(DATA_DIR, 'majestic.csv')
    if not os.path.exists(path):
        return []
    print(f'  Found data/majestic.csv — reading...')
    df = pd.read_csv(path)
    domains = df['Domain'].dropna().head(MAX_URLS * 2).tolist()
    urls = []
    for i, domain in enumerate(domains):
        path_suffix = LEGIT_PATHS[i % len(LEGIT_PATHS)]
        urls.append(f'https://{domain}{path_suffix}')
    print(f'  Majestic local: {len(urls):,} real legit URLs')
    return urls


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print('=' * 55)
    print('  STEP 1 — Building training dataset')
    print('=' * 55)

    # --- Phishing URLs ---
    # Priority: local file → online download → synthetic fallback
    print('\nPhishing URLs:')
    phishing = read_urlhaus_local()
    if len(phishing) < 500:
        phishing = try_download_phishing()
    if len(phishing) < 500:
        print(f'  No real data found. Using embedded synthetic dataset.')
        print(f'  TIP: Save urlhaus.abuse.ch/downloads/csv_recent/ as data/urlhaus.csv')
        phishing = build_phishing_urls(MAX_URLS)
    print(f'  Total phishing: {len(phishing):,}')

    # --- Legit URLs ---
    # Priority: local file → online download → synthetic fallback
    print('\nLegit URLs:')
    legit = read_majestic_local()
    if len(legit) < 500:
        legit = try_download_legit()
    if len(legit) < 500:
        print(f'  No real data found. Using embedded legit dataset.')
        print(f'  TIP: Save majestic.com/reports/majestic-million as data/majestic.csv')
        legit = build_legit_urls()
    print(f'  Total legit: {len(legit):,}')

    # --- Balance ---
    phishing = list(set(phishing))
    legit    = list(set(legit))
    n = min(len(phishing), len(legit), MAX_URLS)
    phishing = phishing[:n]
    legit    = legit[:n]

    print(f'\nBalanced dataset: {n:,} phishing + {n:,} legit = {n*2:,} total URLs')

    df = pd.concat([
        pd.DataFrame({'url': phishing, 'label': 1}),
        pd.DataFrame({'url': legit,    'label': 0}),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    out = os.path.join(DATA_DIR, 'urls_raw.csv')
    df.to_csv(out, index=False)

    print(f'\nSaved to: {out}')
    print(df['label'].value_counts().rename({1:'phishing', 0:'legit'}).to_string())

    # Show a few examples so you can verify they look realistic
    print('\nSample legit URLs:')
    for u in df[df.label==0]['url'].head(3).tolist():
        print(f'  {u}')
    print('Sample phishing URLs:')
    for u in df[df.label==1]['url'].head(3).tolist():
        print(f'  {u}')

    print('\nDone! Next step: python step2_train.py')


if __name__ == '__main__':
    main()