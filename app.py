import streamlit as st
import pickle
import re
import io
import csv
import html
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from collections import Counter
from datetime import datetime

# ── langdetect: optional, degrade gracefully if missing ──────────────────────
try:
    from langdetect import detect as _lang_detect
    from langdetect.lang_detect_exception import LangDetectException
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="CineSense AI | Movie Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS & THEME
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Main app canvas */
.stApp {
    background: radial-gradient(circle at 50% 0%, #171c2c 0%, #0d101b 50%, #080a12 100%) !important;
    color: #e5ebf8 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.block-container {
    max-width: 1250px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: rgba(13, 16, 27, 0.88) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

/* Hero Section */
.hero {
    text-align: center;
    padding: 44px 28px 36px;
    border: 1px solid rgba(245, 197, 24, 0.28);
    border-radius: 24px;
    background: radial-gradient(ellipse at 50% -20%, rgba(245, 197, 24, 0.15) 0%, rgba(20, 25, 41, 0.8) 75%);
    backdrop-filter: blur(20px);
    box-shadow: 0 20px 50px -15px rgba(0, 0, 0, 0.7), 0 0 35px rgba(245, 197, 24, 0.1);
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: "";
    position: absolute;
    top: 0; left: 15%; right: 15%; height: 2px;
    background: linear-gradient(90deg, transparent, #f5c518, #ff8c00, #f5c518, transparent);
}

.hero-top-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 99px;
    font-size: 11.5px;
    font-weight: 750;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    background: rgba(245, 197, 24, 0.12);
    border: 1px solid rgba(245, 197, 24, 0.35);
    color: #ffd859;
    margin-bottom: 12px;
    box-shadow: 0 0 15px rgba(245, 197, 24, 0.15);
}

.hero h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 52px !important;
    font-weight: 900 !important;
    letter-spacing: -1.2px !important;
    background: linear-gradient(135deg, #FFF6D1 0%, #F5C518 50%, #FF9E1B 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    margin: 0 0 6px 0 !important;
    text-shadow: 0 8px 30px rgba(245, 197, 24, 0.25) !important;
}

.hero .subtitle {
    color: #e2e8f7;
    font-size: 19px;
    font-weight: 600;
    margin-bottom: 8px;
    letter-spacing: -0.2px;
}

.hero p {
    color: #9aa7c2;
    font-size: 14.5px;
    line-height: 1.65;
    max-width: 680px;
    margin: 0 auto 18px;
}

.hero-badges {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #dbe3f3;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: all 0.25s ease;
}

.hero-chip:hover {
    transform: translateY(-2px);
    border-color: rgba(245, 197, 24, 0.4);
    color: #ffffff;
}

.hero-chip.accent {
    border-color: rgba(245, 197, 24, 0.4);
    background: rgba(245, 197, 24, 0.1);
    color: #ffd859;
}

/* Info Cards */
.info-card {
    background: rgba(19, 24, 38, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 20px 14px;
    text-align: center;
    min-height: 125px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    backdrop-filter: blur(12px);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 8px 24px -6px rgba(0,0,0,0.4);
}

.info-card:hover {
    transform: translateY(-4px);
    border-color: rgba(245, 197, 24, 0.45);
    box-shadow: 0 16px 36px -8px rgba(245, 197, 24, 0.18);
    background: rgba(26, 33, 52, 0.9);
}

.info-card h3 {
    font-family: 'Outfit', sans-serif;
    color: #f5c518;
    font-size: 17px;
    font-weight: 700;
    margin: 0 0 6px;
}

.info-card p {
    color: #a7b4cf;
    font-size: 12.5px;
    line-height: 1.5;
    margin: 0;
}

/* Stat Cards in Sidebar */
.stat-card {
    background: rgba(23, 28, 45, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 16px 12px;
    text-align: center;
    transition: all 0.2s ease;
    box-shadow: 0 6px 18px rgba(0,0,0,0.3);
}

.stat-card:hover {
    border-color: rgba(245, 197, 24, 0.3);
    transform: translateY(-2px);
}

.stat-card .stat-num {
    font-family: 'Outfit', sans-serif;
    font-size: 30px;
    font-weight: 800;
    color: #f5c518;
    line-height: 1.1;
}

.stat-card .stat-lbl {
    font-size: 12px;
    font-weight: 600;
    color: #94a1bc;
    margin-top: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Section Title */
.section-title {
    font-family: 'Outfit', sans-serif;
    color: #f5c518;
    font-size: 26px;
    font-weight: 800;
    margin-top: 24px;
    margin-bottom: 8px;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Primary Buttons */
div.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 14px 20px;
    background: linear-gradient(135deg, #f5c518 0%, #e8a900 100%);
    color: #0b0e14;
    font-family: 'Outfit', sans-serif;
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 0.5px;
    min-height: 48px;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 6px 20px -4px rgba(245, 197, 24, 0.4);
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #ffd859 0%, #f5b700 100%);
    transform: translateY(-2px);
    box-shadow: 0 10px 28px -4px rgba(245, 197, 24, 0.55);
    color: #05070a;
}

div.stButton > button:active {
    transform: translateY(0);
}

/* Text Area */
.stTextArea textarea {
    background: rgba(18, 22, 35, 0.8) !important;
    color: #f5f8ff !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 16px !important;
    font-size: 15.5px !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    transition: all 0.2s ease !important;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.4) !important;
}

.stTextArea textarea:focus {
    border-color: #f5c518 !important;
    box-shadow: 0 0 0 2px rgba(245, 197, 24, 0.25), inset 0 2px 8px rgba(0,0,0,0.4) !important;
}

/* Result Cards */
.result-card {
    padding: 34px 26px;
    border-radius: 22px;
    text-align: center;
    margin-top: 18px;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(16px);
}

.result-card.positive {
    background: radial-gradient(circle at 50% 0%, rgba(0, 230, 118, 0.2) 0%, rgba(13, 38, 29, 0.85) 90%);
    border: 1.5px solid #00e676;
    box-shadow: 0 16px 45px -10px rgba(0, 230, 118, 0.25), 0 0 30px rgba(0, 230, 118, 0.1);
}

.result-card.negative {
    background: radial-gradient(circle at 50% 0%, rgba(255, 23, 68, 0.2) 0%, rgba(45, 16, 22, 0.85) 90%);
    border: 1.5px solid #ff1744;
    box-shadow: 0 16px 45px -10px rgba(255, 23, 68, 0.25), 0 0 30px rgba(255, 23, 68, 0.1);
}

.verdict-tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 99px;
    font-size: 11.5px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
}

.positive .verdict-tag {
    background: rgba(0, 230, 118, 0.15);
    border: 1px solid rgba(0, 230, 118, 0.4);
    color: #69f0ae;
}

.negative .verdict-tag {
    background: rgba(255, 23, 68, 0.15);
    border: 1px solid rgba(255, 23, 68, 0.4);
    color: #ff8a9a;
}

.result-title {
    font-family: 'Outfit', sans-serif;
    font-size: 34px;
    font-weight: 900;
    letter-spacing: -0.5px;
    margin-bottom: 8px;
}

.positive .result-title { color: #8affc6; text-shadow: 0 0 25px rgba(0, 230, 118, 0.4); }
.negative .result-title { color: #ff99a8; text-shadow: 0 0 25px rgba(255, 23, 68, 0.4); }

.result-desc {
    font-size: 15.5px;
    color: #d1d8e8;
    line-height: 1.6;
}

.result-desc strong { color: #ffffff; }

/* Intensity Badge */
.intensity-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 99px;
    font-family: 'Outfit', sans-serif;
    font-size: 13.5px;
    font-weight: 750;
    margin-top: 14px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.3);
}

/* Confidence Bar Wrapper */
.conf-bar-wrap {
    background: rgba(20, 25, 40, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 20px 24px;
    margin-top: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}

.conf-bar-wrap h4 {
    font-family: 'Outfit', sans-serif;
    color: #f5c518;
    margin: 0 0 12px;
    font-size: 16.5px;
    font-weight: 700;
}

/* Keyword Pills Row */
.kw-pill-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
    margin-top: 8px;
}

.kw-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 12.5px;
    font-weight: 600;
    transition: all 0.2s ease;
}

.kw-pill-pos {
    background: rgba(0, 230, 118, 0.12);
    border: 1px solid rgba(0, 230, 118, 0.35);
    color: #69f0ae;
}

.kw-pill-neg {
    background: rgba(255, 23, 68, 0.12);
    border: 1px solid rgba(255, 23, 68, 0.35);
    color: #ff8a9a;
}

/* Pulsing Live Box */
.live-box {
    background: rgba(18, 23, 37, 0.85);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 16px 20px;
    margin-top: 12px;
    font-size: 14.5px;
    color: #d8e1f5;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.35);
}

.pulse-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background-color: #00e676;
    box-shadow: 0 0 10px #00e676;
    animation: pulseAnim 1.8s infinite;
}

@keyframes pulseAnim {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0.7); }
    70% { transform: scale(1.15); box-shadow: 0 0 0 8px rgba(0, 230, 118, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 230, 118, 0); }
}

/* Review Highlighting */
.kw-positive {
    background: rgba(0, 230, 118, 0.22);
    color: #a5f5d0;
    border-radius: 5px;
    padding: 2px 6px;
    font-weight: 650;
    border-bottom: 2px solid #00e676;
}

.kw-negative {
    background: rgba(255, 23, 68, 0.22);
    color: #ffb8c2;
    border-radius: 5px;
    padding: 2px 6px;
    font-weight: 650;
    border-bottom: 2px solid #ff1744;
}

/* History Rows */
.hist-row {
    background: rgba(20, 25, 40, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 14px 18px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 14px;
    color: #d8e1f5;
    transition: all 0.2s ease;
}

.hist-row:hover {
    border-color: rgba(245, 197, 24, 0.3);
    background: rgba(25, 32, 50, 0.85);
    transform: translateX(4px);
}

.hist-row .badge-pos { color: #69f0ae; font-weight: 700; }
.hist-row .badge-neg { color: #ff8a9a; font-weight: 700; }

/* Tabs */
button[data-baseweb="tab"] {
    color: #a5b2cc !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 15.5px !important;
    font-weight: 600 !important;
    padding-top: 10px !important;
    padding-bottom: 10px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #f5c518 !important;
    border-bottom: 2.5px solid #f5c518 !important;
    font-weight: 750 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #8b97b0;
    font-size: 13.5px;
    line-height: 2;
    margin-top: 60px;
    padding-top: 24px;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
}

@media (max-width: 768px) {
    .hero h1 { font-size: 36px !important; }
    .result-card { font-size: 20px; padding: 22px 14px; }
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# PLOTLY THEME DEFAULTS
# =====================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14, 18, 30, 0.6)",
    font=dict(color="#d1d6e4", family="Plus Jakarta Sans, sans-serif"),
    margin=dict(l=40, r=20, t=40, b=40),
)


# =====================================================
# LOAD TRAINED MODEL & ARTIFACTS
# =====================================================
MODEL_PATH = Path(__file__).resolve().parent / "sentiment_analysis.pkl"


@st.cache_resource
def load_model_and_artifacts():
    with open(MODEL_PATH, "rb") as f:
        mdl = pickle.load(f)
    tfidf = mdl.named_steps["tfidf"]
    clf = mdl.named_steps["clf"]
    feat_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]
    return mdl, tfidf, clf, feat_names, coefs


try:
    model, tfidf_step, clf_step, FEATURE_NAMES, CLF_COEFS = load_model_and_artifacts()
except Exception as e:
    st.error("Unable to load the trained model. Ensure sentiment_analysis.pkl exists.")
    st.code(str(e))
    st.stop()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

# ── 1. Language detection ────────────────────────────────────────────────────
def detect_language(text: str) -> str | None:
    """Return ISO language code or None if unavailable / uncertain."""
    if not LANGDETECT_OK or len(text.strip()) < 10:
        return None
    try:
        return _lang_detect(text)
    except Exception:
        return None


# ── 2. Sentiment intensity label ─────────────────────────────────────────────
INTENSITY_MAP = {
    "positive": [
        (95, "Overwhelmingly Positive", "#00e676", "#003320"),
        (85, "Strongly Positive",       "#35c58b", "#123c30"),
        (70, "Moderately Positive",     "#80e8b0", "#1a3d2b"),
        (55, "Mildly Positive",         "#b8f0d0", "#203530"),
        (0,  "Barely Positive",         "#d4f5e2", "#202e28"),
    ],
    "negative": [
        (95, "Overwhelmingly Negative", "#ff1744", "#3a0a10"),
        (85, "Strongly Negative",       "#f06a7a", "#47232b"),
        (70, "Moderately Negative",     "#f5a0aa", "#3a1a20"),
        (55, "Mildly Negative",         "#f5c0c5", "#302022"),
        (0,  "Barely Negative",         "#f5d5d7", "#2a2022"),
    ],
}

def get_intensity(label: str, confidence: float) -> tuple[str, str, str]:
    """Return (intensity_label, text_color, bg_color)."""
    for threshold, lbl, fg, bg in INTENSITY_MAP.get(label, []):
        if confidence >= threshold:
            return lbl, fg, bg
    return label.capitalize(), "#f5c518", "#202536"


# ── 3. Confidence score & prediction ─────────────────────────────────────────
def get_confidence(text: str) -> tuple[str, float]:
    raw_score = float(model.decision_function([text])[0])
    conf_raw  = 1.0 / (1.0 + np.exp(-abs(raw_score)))
    confidence = round(conf_raw * 100, 1)
    label = "positive" if raw_score > 0 else "negative"
    return label, confidence


# ── 4. Keyword extraction (Optimized with cached features) ───────────────────
def get_top_keywords(text: str, n: int = 10) -> list[tuple[str, float, str]]:
    try:
        X       = tfidf_step.transform([text])
        cx      = X.tocsr()
        nz_idx  = cx.indices
        nz_vals = np.array(cx.data, dtype=float)
        if len(nz_idx) == 0:
            return []
        scores  = nz_vals * CLF_COEFS[nz_idx]
        top_n   = min(n, len(scores))
        top_idx = np.argsort(np.abs(scores))[::-1][:top_n]
        results = []
        for i in top_idx:
            word      = FEATURE_NAMES[nz_idx[i]]
            score     = scores[i]
            direction = "positive" if score > 0 else "negative"
            results.append((word, abs(score), direction))
        return results
    except Exception:
        return []


# ── 5. Review highlighter (Safe HTML-escaped) ────────────────────────────────
def highlight_review(text: str, keywords: list) -> str:
    pos_words = {kw.lower() for kw, _, d in keywords if d == "positive"}
    neg_words = {kw.lower() for kw, _, d in keywords if d == "negative"}
    sorted_words = sorted(pos_words | neg_words, key=len, reverse=True)

    if not sorted_words:
        return html.escape(text)

    pattern = r'\b(' + '|'.join(re.escape(w) for w in sorted_words) + r')\b'
    parts   = re.split(pattern, text, flags=re.IGNORECASE)
    output  = []
    for part in parts:
        lw = part.lower()
        if lw in pos_words:
            output.append(f'<span class="kw-positive">{html.escape(part)}</span>')
        elif lw in neg_words:
            output.append(f'<span class="kw-negative">{html.escape(part)}</span>')
        else:
            output.append(html.escape(part))
    return "".join(output)


# ── 6. Full analysis ──────────────────────────────────────────────────────────
def analyze_review(text: str) -> dict:
    label, conf       = get_confidence(text)
    kws               = get_top_keywords(text)
    intensity, fg, bg = get_intensity(label, conf)
    lang              = detect_language(text)
    return {
        "label":      label,
        "confidence": conf,
        "intensity":  intensity,
        "int_fg":     fg,
        "int_bg":     bg,
        "keywords":   kws,
        "text":       text,
        "lang":       lang,
        "time":       datetime.now().strftime("%H:%M:%S"),
    }


# ── 7. Render result card ─────────────────────────────────────────────────────
def render_result_card(result: dict, prefix: str = ""):
    label     = result["label"]
    conf      = result["confidence"]
    intensity = result["intensity"]
    int_fg    = result["int_fg"]
    int_bg    = result["int_bg"]
    kws       = result["keywords"]
    bar_color = "#00e676" if label == "positive" else "#ff1744"
    emoji     = "✨ 😊" if label == "positive" else "⚠️ 😞"
    css_cls   = "positive" if label == "positive" else "negative"
    verdict_badge = "POSITIVE SENTIMENT DETECTED" if label == "positive" else "NEGATIVE SENTIMENT DETECTED"

    # Language warning
    lang = result.get("lang")
    if lang and lang != "en":
        lang_names = {
            "hi": "Hindi", "fr": "French", "de": "German", "es": "Spanish",
            "it": "Italian", "pt": "Portuguese", "zh-cn": "Chinese",
            "ja": "Japanese", "ar": "Arabic", "ru": "Russian",
        }
        lang_display = lang_names.get(lang, lang.upper())
        st.markdown(f"""
        <div class="lang-warn">
            ⚠️ <strong>Non-English text detected ({html.escape(lang_display)})</strong> — This model was trained on
            English IMDb reviews. Results may be inaccurate for non-English input.
        </div>""", unsafe_allow_html=True)

    # Main result card with verdict tag and intensity badge
    st.markdown(f"""
    <div class="result-card {css_cls}">
        <div class="verdict-tag">{verdict_badge}</div>
        <div class="result-title">{emoji} {intensity}</div>
        <div class="result-desc">
            The NLP Pipeline classifies this review as <strong>{label.upper()}</strong>
            with <strong>{conf}%</strong> statistical confidence.
        </div>
        <div>
            <span class="intensity-badge" style="background:{int_bg};color:{int_fg};border:1px solid {int_fg}60;">
                ★ {intensity}
            </span>
        </div>
    </div>""", unsafe_allow_html=True)

    # Confidence bar
    conf_label = (
        "🔥 Very High Confidence" if conf >= 90 else
        "✅ High Confidence"       if conf >= 75 else
        "⚠️ Moderate Confidence"   if conf >= 60 else
        "❓ Borderline Confidence"
    )
    st.markdown(f"""
    <div class="conf-bar-wrap">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
            <h4 style="margin:0;">📊 Statistical Confidence</h4>
            <span style="font-family:'Outfit',sans-serif;font-weight:800;font-size:18px;color:#f5c518;">{conf}%</span>
        </div>
        <div style="background:#171c2c;border-radius:99px;height:14px;overflow:hidden;box-shadow:inset 0 2px 6px rgba(0,0,0,0.5);">
            <div style="width:{conf}%;background:{bar_color};height:100%;border-radius:99px;
                        box-shadow:0 0 14px {bar_color};transition:width 0.6s ease;"></div>
        </div>
        <div style="font-size:12.5px;color:#94a1bc;margin-top:8px;">{conf_label}</div>
    </div>""", unsafe_allow_html=True)

    # Keyword pills + bar chart
    if kws:
        st.markdown("#### 🔍 Influential Keyword Signals")

        # Visual Keyword Pills
        pill_tags = []
        for word, score, direction in kws[:8]:
            p_cls = "kw-pill-pos" if direction == "positive" else "kw-pill-neg"
            arrow = "▲" if direction == "positive" else "▼"
            pill_tags.append(f'<span class="kw-pill {p_cls}">{arrow} {html.escape(word)} ({score:.2f})</span>')
        st.markdown(f'<div class="kw-pill-container">{"".join(pill_tags)}</div>', unsafe_allow_html=True)

        words  = [k[0] for k in kws]
        scores = [k[1] for k in kws]
        dirs   = [k[2] for k in kws]
        colors = ["#00e676" if d == "positive" else "#ff1744" for d in dirs]

        fig = go.Figure(go.Bar(
            x=scores[::-1],
            y=words[::-1],
            orientation="h",
            marker=dict(color=colors[::-1], line=dict(color="rgba(255,255,255,0.15)", width=1)),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=max(260, len(kws) * 28),
            xaxis=dict(title="Weight / Influence Magnitude", gridcolor="rgba(255,255,255,0.06)", color="#8b93a8"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="#d1d6e4"),
            title=dict(text="Feature Contribution to Hyperplane Margin", font=dict(color="#f5c518", size=14)),
        )
        chart_key = f"kw_chart_{prefix}_{result.get('time', '')}_{abs(hash(result.get('text', '')))}"
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

        # Highlighted text
        highlighted = highlight_review(result["text"], kws)
        st.markdown(
            f'<div style="background:rgba(18,22,35,0.85);border:1px solid rgba(255,255,255,0.1);border-radius:14px;'
            f'padding:18px 20px;line-height:2.0;font-size:15.5px;box-shadow:0 6px 20px rgba(0,0,0,0.35);">{highlighted}</div>',
            unsafe_allow_html=True
        )
        st.caption("🟢 Green glow = pushed towards Positive  |  🔴 Red glow = pushed towards Negative")
    else:
        st.markdown("#### 🔍 Influential Keywords")
        st.info("No strong vocabulary keywords detected in the model dictionary for this text.")
        st.markdown(
            f'<div style="background:rgba(18,22,35,0.85);border:1px solid rgba(255,255,255,0.1);border-radius:14px;'
            f'padding:18px 20px;line-height:2.0;font-size:15.5px;">{html.escape(result["text"])}</div>',
            unsafe_allow_html=True
        )


# ── 8. Live preview helper ────────────────────────────────────────────────────
def live_preview(text: str):
    """Return a lightweight live sentiment estimate (no keyword extraction)."""
    if not text.strip() or len(text.split()) < 2:
        return None
    label, conf = get_confidence(text)
    intensity, fg, _ = get_intensity(label, conf)
    emoji = "😊" if label == "positive" else "😞"
    color = "#00e676" if label == "positive" else "#ff1744"
    return emoji, label, conf, intensity, fg, color


# =====================================================
# SESSION STATE
# =====================================================
for key, default in [
    ("history", []),
    ("review_input", ""),
    ("single_result", None),
    ("compare_results", None),
    ("batch_results", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 🎬 CineSense AI")
    st.markdown("---")

    history_sidebar = st.session_state.history
    total    = len(history_sidebar)
    pos_cnt  = sum(1 for r in history_sidebar if r["label"] == "positive")
    neg_cnt  = total - pos_cnt
    avg_conf = round(float(np.mean([r["confidence"] for r in history_sidebar])), 1) if total else 0

    st.markdown("### 📈 Session Stats")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-num">{total}</div>
            <div class="stat-lbl">Total Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-num">{avg_conf}%</div>
            <div class="stat-lbl">Avg Confidence</div>
        </div>""", unsafe_allow_html=True)

    st.write("")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-num" style="color:#35c58b;">{pos_cnt}</div>
            <div class="stat-lbl">Positive</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-card">
            <div class="stat-num" style="color:#f06a7a;">{neg_cnt}</div>
            <div class="stat-lbl">Negative</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    if total > 0:
        st.markdown("### 🥧 Sentiment Split")
        st.progress(pos_cnt / total, text=f"😊 Positive ({pos_cnt})")
        st.progress(neg_cnt / total, text=f"😞 Negative ({neg_cnt})")

        # Mini confidence trend in sidebar
        if total >= 2:
            st.markdown("### 📉 Confidence Trend")
            confs  = [r["confidence"] for r in history_sidebar]
            labels = [r["label"] for r in history_sidebar]
            colors_trend = ["#35c58b" if l == "positive" else "#f06a7a" for l in labels]
            fig_side = go.Figure(go.Scatter(
                y=confs, mode="lines+markers",
                line=dict(color="#f5c518", width=2),
                marker=dict(color=colors_trend, size=8),
                hovertemplate="Review %{x+1}: %{y}%<extra></extra>",
            ))
            fig_side.update_layout(
                **{**PLOTLY_LAYOUT, "margin": dict(l=10, r=10, t=10, b=10)},
                height=160,
                xaxis=dict(visible=False),
                yaxis=dict(range=[48, 102], gridcolor="#2e3347", color="#8b93a8", ticksuffix="%"),
                showlegend=False,
            )
            st.plotly_chart(fig_side, use_container_width=True, key="sidebar_trend")

    st.markdown("---")

    if st.button("🗑️ Clear History", key="clear_hist"):
        st.session_state.history = []
        st.session_state.single_result = None
        st.session_state.compare_results = None
        st.session_state.batch_results = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:12px;color:#8b93a8;line-height:1.8;">
    <strong style="color:#f5c518;">Model:</strong> LinearSVC<br>
    <strong style="color:#f5c518;">Features:</strong> TF-IDF (~95k Vocabulary)<br>
    <strong style="color:#f5c518;">Dataset:</strong> IMDb 50k Reviews<br>
    <strong style="color:#f5c518;">Accuracy:</strong> ~89–91%<br>
    <strong style="color:#f5c518;">Framework:</strong> Scikit-learn + Plotly
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# HERO
# =====================================================
st.markdown("""
<div class="hero">
    <div class="hero-top-badge">🎬 NEXT-GEN NLP SENTIMENT INTELLIGENCE</div>
    <h1>CineSense AI</h1>
    <div class="subtitle">IMDb Movie Review Sentiment Engine</div>
    <p>
        An enterprise-grade Natural Language Processing framework trained on 50,000 IMDb movie reviews.
        Featuring LinearSVC maximum-margin classification, 95k-word TF-IDF representations,
        real-time keystroke sentiment radar &amp; comprehensive comparative analytics.
    </p>
    <div class="hero-badges">
        <span class="hero-chip accent">🧠 LinearSVC Core</span>
        <span class="hero-chip">🔤 94,814 Vocab Features</span>
        <span class="hero-chip accent">🎯 90.0% Test Accuracy</span>
        <span class="hero-chip">⚡ Real-Time Radar</span>
        <span class="hero-chip">🍿 IMDb 50k Benchmark</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================
# TECH STACK CARDS
# =====================================================
col1, col2, col3, col4, col5 = st.columns(5)
cards = [
    ("🧠", "LinearSVC",        "Max-margin SVM classifier"),
    ("🔤", "TF-IDF Matrix",    "94.8k feature vocabulary"),
    ("📊", "Plotly Engine",    "Interactive GPU charts"),
    ("🌡️", "5-Level Scale",    "Granular intensity mapping"),
    ("🌍", "LangDetect",      "Multi-lingual awareness shield"),
]
for col, (icon, title, desc) in zip([col1, col2, col3, col4, col5], cards):
    with col:
        st.markdown(f"""
        <div class="info-card">
            <h3>{icon} {title}</h3>
            <p>{desc}</p>
        </div>""", unsafe_allow_html=True)


# =====================================================
# MAIN TABS
# =====================================================
st.write("")
tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Single Review",
    "⚖️ Compare Reviews",
    "📦 Batch Analysis",
    "📜 History & Analytics",
])


# ══════════════════════════════════════════════════════
# TAB 1 — SINGLE REVIEW
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📝 Analyze a Movie Review</div>', unsafe_allow_html=True)
    st.write("Enter your own review or pick a sample below.")

    SAMPLES = {
        "😊 Positive Sample": (
            "This movie was absolutely fantastic! The performances were outstanding, "
            "the cinematography was breathtaking, and the storyline kept me hooked "
            "from start to finish. A true masterpiece of modern cinema."
        ),
        "😞 Negative Sample": (
            "This film was a complete disappointment. The plot was predictable and dull, "
            "the characters were poorly written, and the pacing was unbearably slow. "
            "I regret wasting two hours on this."
        ),
        "😐 Mixed Sample": (
            "The visuals were impressive and the music score was excellent, "
            "but the story felt hollow and the ending was anticlimactic. "
            "Not the worst film, but not worth the price of admission."
        ),
    }

    sc1, sc2, sc3 = st.columns(3)
    for col, (lbl, txt) in zip([sc1, sc2, sc3], SAMPLES.items()):
        with col:
            if st.button(lbl, key=f"sample_{lbl}"):
                st.session_state.review_input = txt
                st.session_state.single_result = None
                st.rerun()

    # ── Text input ──
    review = st.text_area(
        "Enter your movie review",
        key="review_input",
        height=180,
        placeholder="Example: The movie was wonderful, with excellent acting and a captivating story...",
    )

    word_count = len(review.split()) if review.strip() else 0
    char_count = len(review)

    m1, m2, m3 = st.columns(3)
    with m1: st.caption(f"📝 Words: {word_count}")
    with m2: st.caption(f"🔡 Characters: {char_count}")
    with m3:
        hint = "—"
        if word_count > 0:
            pw = {"great","amazing","fantastic","excellent","wonderful","brilliant","loved","best","outstanding","masterpiece"}
            nw = {"terrible","awful","boring","worst","bad","waste","poor","dull","disappointing","regret"}
            tokens = set(re.findall(r'\b[a-z]+\b', review.lower()))
            ph, nh = len(tokens & pw), len(tokens & nw)
            hint = ("🟢 Likely Positive" if ph > nh else
                    "🔴 Likely Negative" if nh > ph else "🔵 Mixed/Neutral")
        st.caption(f"Quick hint: {hint}")

    # ── LIVE SENTIMENT PREVIEW ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚡ Live Keystroke Radar")
    st.caption("Real-time sentiment inference dynamically updating on every keystroke.")

    if word_count >= 3:
        live = live_preview(review)
        if live:
            emoji_l, label_l, conf_l, intensity_l, fg_l, color_l = live

            # Language check in live preview too
            lang_l = detect_language(review)
            if lang_l and lang_l != "en":
                st.markdown(f"""<div class="lang-warn">
                    ⚠️ Non-English text detected — predictions may be inaccurate.
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="live-box">
                <span class="pulse-indicator"></span>
                <span style="font-size:30px;">{emoji_l}</span>
                <div>
                    <div style="font-family:'Outfit',sans-serif;font-size:17px;font-weight:800;color:{color_l};">{intensity_l}</div>
                    <div style="font-size:12.5px;color:#94a1bc;">Live Confidence: <strong style="color:#ffffff;">{conf_l}%</strong> &nbsp;|&nbsp; Category: <strong style="color:#ffffff;">{label_l.capitalize()}</strong></div>
                </div>
                <div style="margin-left:auto;background:#151a2a;border-radius:99px;
                            width:130px;height:10px;overflow:hidden;box-shadow:inset 0 1px 4px rgba(0,0,0,0.5);">
                    <div style="width:{conf_l}%;background:{color_l};height:100%;border-radius:99px;box-shadow:0 0 10px {color_l};"></div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="live-box" style="color:#64708d;">
            <span style="font-size:22px;">⌨️</span>
            <span>Type at least 3 words to activate the real-time sentiment radar…</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Analyze button ──
    if st.button("🔍 Analyze Sentiment", key="analyze_btn"):
        if not review.strip():
            st.warning("Please enter a movie review before analyzing.")
        elif word_count < 3:
            st.warning("Please enter at least a few words for a meaningful analysis.")
        else:
            with st.spinner("Analyzing sentiment…"):
                result = analyze_review(review)
                st.session_state.history.append(result)
                st.session_state.single_result = result

    # Render persisted single result so interactive Plotly actions don't clear it
    if st.session_state.single_result is not None:
        st.markdown("### 🎯 Full Analysis Result")
        render_result_card(st.session_state.single_result, prefix="single")
        st.caption("Note: This is a machine learning prediction and may not always be accurate.")


# ══════════════════════════════════════════════════════
# TAB 2 — COMPARISON MODE
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">⚖️ Compare Two Reviews</div>', unsafe_allow_html=True)
    st.write("Analyze two reviews side-by-side and compare sentiment, intensity & confidence.")

    cmp_c1, cmp_c2 = st.columns(2)
    with cmp_c1:
        st.markdown("#### 📄 Review A")
        rev_a = st.text_area("Review A", height=160, key="rev_a",
                             placeholder="Enter first review here…", label_visibility="collapsed")
    with cmp_c2:
        st.markdown("#### 📄 Review B")
        rev_b = st.text_area("Review B", height=160, key="rev_b",
                             placeholder="Enter second review here…", label_visibility="collapsed")

    if st.button("⚖️ Compare Both Reviews", key="compare_btn"):
        if not rev_a.strip() or not rev_b.strip():
            st.warning("Please enter text in both review fields.")
        else:
            with st.spinner("Comparing…"):
                res_a = analyze_review(rev_a)
                res_b = analyze_review(rev_b)
                st.session_state.history.extend([res_a, res_b])
                st.session_state.compare_results = (res_a, res_b)

    # Render persisted compare results so chart interactions don't reset state
    if st.session_state.compare_results is not None:
        res_a, res_b = st.session_state.compare_results

        st.markdown("### 🎯 Comparison Results")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### Review A")
            render_result_card(res_a, prefix="cmp_a")
        with r2:
            st.markdown("#### Review B")
            render_result_card(res_b, prefix="cmp_b")

        # ── Comparison chart ──
        st.markdown("---")
        st.markdown("#### 📊 Head-to-Head Comparison Chart")

        categories = ["Confidence", "Word Count", "Positivity Score"]
        wc_a = min(len(res_a["text"].split()), 100)
        wc_b = min(len(res_b["text"].split()), 100)
        pos_score_a = res_a["confidence"] if res_a["label"] == "positive" else 100 - res_a["confidence"]
        pos_score_b = res_b["confidence"] if res_b["label"] == "positive" else 100 - res_b["confidence"]

        fig_radar = go.Figure()
        for name, vals, color in [
            ("Review A", [res_a["confidence"], wc_a, pos_score_a], "#f5c518"),
            ("Review B", [res_b["confidence"], wc_b, pos_score_b], "#35c58b"),
        ]:
            fig_radar.add_trace(go.Bar(name=name, x=categories, y=vals, marker_color=color))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            barmode="group",
            height=320,
            yaxis=dict(range=[0, 105], gridcolor="#2e3347", ticksuffix="%"),
            legend=dict(font=dict(color="#d1d6e4")),
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="cmp_chart")

        # Metric summary
        summ1, summ2, summ3 = st.columns(3)
        winner = "A" if res_a["confidence"] > res_b["confidence"] else ("B" if res_b["confidence"] > res_a["confidence"] else "Tie")
        with summ1:
            st.metric("🏆 Higher Confidence", f"Review {winner}",
                      delta=f"{abs(res_a['confidence'] - res_b['confidence']):.1f}%")
        with summ2:
            st.metric("Review A", res_a["intensity"], delta=f"{res_a['confidence']}%")
        with summ3:
            st.metric("Review B", res_b["intensity"], delta=f"{res_b['confidence']}%")


# ══════════════════════════════════════════════════════
# TAB 3 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📦 Batch Sentiment Analysis</div>', unsafe_allow_html=True)
    st.write("Analyze multiple reviews at once — type them or upload a CSV.")

    batch_method = st.radio(
        "Input method",
        ["✏️ Type Reviews (one per line)", "📁 Upload CSV File"],
        horizontal=True, key="batch_method",
    )

    batch_reviews = []

    if batch_method == "✏️ Type Reviews (one per line)":
        batch_text = st.text_area(
            "Enter reviews (one per line)", height=200, key="batch_text",
            placeholder=(
                "The acting was incredible and the plot was gripping.\n"
                "Worst movie I've ever seen, complete waste of time.\n"
                "Decent film with some good moments but overall average."
            ),
        )
        if batch_text.strip():
            batch_reviews = [r.strip() for r in batch_text.split("\n") if r.strip()]
    else:
        uploaded = st.file_uploader("Upload CSV (must contain review text)", type=["csv"], key="batch_csv")
        if uploaded:
            try:
                raw_bytes = uploaded.read()
                try:
                    content = raw_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    content = raw_bytes.decode("latin-1")

                reader = csv.DictReader(io.StringIO(content))
                fieldnames = reader.fieldnames or []
                col_map = {f.strip().lower(): f for f in fieldnames}

                review_col = None
                for candidate in ["review", "reviews", "text", "content", "comment", "feedback"]:
                    if candidate in col_map:
                        review_col = col_map[candidate]
                        break

                if not review_col:
                    st.error(f"Could not find a review column in CSV. Found columns: {', '.join(fieldnames)}. Please ensure one column is named 'review' or 'text'.")
                else:
                    batch_reviews = [
                        val for row in reader
                        if (val := (row.get(review_col) or "").strip())
                    ]
                    st.success(f"Loaded {len(batch_reviews)} review(s) from column '{review_col}'.")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    if batch_reviews:
        st.info(f"Ready to analyze **{len(batch_reviews)}** review(s).")

    if st.button("🚀 Run Batch Analysis", key="batch_btn"):
        if not batch_reviews:
            st.warning("Please enter or upload some reviews first.")
        else:
            results  = []
            progress = st.progress(0, text="Analyzing reviews…")
            for i, rev in enumerate(batch_reviews):
                r = analyze_review(rev)
                results.append(r)
                st.session_state.history.append(r)
                progress.progress((i + 1) / len(batch_reviews), text=f"Analyzing {i+1}/{len(batch_reviews)}…")
            progress.empty()
            st.session_state.batch_results = results

    # Render persisted batch results so clicking Download CSV does NOT wipe away results
    if st.session_state.batch_results:
        results  = st.session_state.batch_results
        total_b  = len(results)
        pos_b    = sum(1 for r in results if r["label"] == "positive")
        neg_b    = total_b - pos_b
        avg_c    = round(float(np.mean([r["confidence"] for r in results])), 1)

        # Summary metrics
        st.markdown("### 📊 Batch Summary")
        bm1, bm2, bm3, bm4 = st.columns(4)
        bm1.metric("Total Reviews", total_b)
        bm2.metric("😊 Positive", pos_b, delta=f"{round(pos_b/total_b*100,1)}%")
        bm3.metric("😞 Negative", neg_b, delta=f"{round(neg_b/total_b*100,1)}%")
        bm4.metric("Avg Confidence", f"{avg_c}%")

        # ── Plotly batch charts ──────────────────────────────────────────
        st.markdown("### 📈 Batch Charts")
        bc1, bc2 = st.columns(2)

        with bc1:
            fig_pie = go.Figure(go.Pie(
                labels=["Positive", "Negative"],
                values=[pos_b, neg_b],
                marker_colors=["#35c58b", "#f06a7a"],
                hole=0.45,
                textfont=dict(color="#ffffff"),
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            ))
            fig_pie.update_layout(
                **PLOTLY_LAYOUT,
                height=300,
                title=dict(text="Sentiment Distribution", font=dict(color="#f5c518")),
                legend=dict(font=dict(color="#d1d6e4")),
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="batch_pie")

        with bc2:
            confs_b = [r["confidence"] for r in results]
            col_b   = ["#35c58b" if r["label"] == "positive" else "#f06a7a" for r in results]
            fig_hist = go.Figure(go.Bar(
                x=[f"#{i+1}" for i in range(len(results))],
                y=confs_b,
                marker_color=col_b,
                hovertemplate="Review %{x}: %{y}%<extra></extra>",
            ))
            fig_hist.update_layout(
                **PLOTLY_LAYOUT,
                height=300,
                title=dict(text="Confidence per Review", font=dict(color="#f5c518")),
                yaxis=dict(range=[48, 102], gridcolor="#2e3347", ticksuffix="%"),
                xaxis=dict(gridcolor="#2e3347"),
            )
            st.plotly_chart(fig_hist, use_container_width=True, key="batch_hist")

        # ── Intensity breakdown chart ────────────────────────────────────
        intensity_counts = Counter(r["intensity"] for r in results)
        if intensity_counts:
            fig_int = go.Figure(go.Bar(
                x=list(intensity_counts.values())[::-1],
                y=list(intensity_counts.keys())[::-1],
                orientation="h",
                marker_color="#f5c518",
                hovertemplate="%{y}: %{x} review(s)<extra></extra>",
            ))
            fig_int.update_layout(
                **PLOTLY_LAYOUT,
                height=280,
                title=dict(text="Intensity Label Breakdown", font=dict(color="#f5c518")),
                xaxis=dict(gridcolor="#2e3347"),
                yaxis=dict(gridcolor="#2e3347"),
            )
            st.plotly_chart(fig_int, use_container_width=True, key="batch_intensity")

        # ── Results table ────────────────────────────────────────────────
        st.markdown("### 📋 Detailed Results")
        for i, r in enumerate(results, 1):
            label_badge = (
                '<span style="color:#7af0bc;font-weight:700;">😊 Positive</span>'
                if r["label"] == "positive"
                else '<span style="color:#ffabb3;font-weight:700;">😞 Negative</span>'
            )
            lang_flag = f' <span style="color:#c97a10;font-size:12px;">[{html.escape(r["lang"].upper())}]</span>' if r.get("lang") and r["lang"] != "en" else ""
            raw_text  = r["text"]
            preview   = html.escape(raw_text[:80] + ("…" if len(raw_text) > 80 else ""))
            st.markdown(f"""
            <div class="hist-row">
                <span style="color:#8b93a8;min-width:28px;">#{i}</span>
                <span style="flex:1;margin:0 14px;">{preview}{lang_flag}</span>
                <span style="min-width:160px;text-align:right;">{label_badge}
                    <span style="color:#8b93a8;font-size:12px;"> · {r["intensity"]}</span>
                </span>
                <span style="min-width:70px;text-align:right;color:#f5c518;">{r["confidence"]}%</span>
            </div>""", unsafe_allow_html=True)

        # Download CSV
        st.markdown("### 💾 Download Results")
        csv_out = io.StringIO()
        writer  = csv.writer(csv_out)
        writer.writerow(["#", "Review", "Sentiment", "Intensity", "Confidence (%)", "Language"])
        for i, r in enumerate(results, 1):
            writer.writerow([i, r["text"], r["label"].capitalize(),
                             r["intensity"], r["confidence"], r.get("lang", "en")])
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_out.getvalue(),
            file_name="sentiment_results.csv",
            mime="text/csv",
            key="download_batch_csv"
        )


# ══════════════════════════════════════════════════════
# TAB 4 — HISTORY & ANALYTICS
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📜 Analysis History & Analytics</div>', unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.info("No analysis history yet. Analyze some reviews first!")
    else:
        total_h    = len(history)
        pos_h      = sum(1 for r in history if r["label"] == "positive")
        neg_h      = total_h - pos_h
        avg_conf_h = round(float(np.mean([r["confidence"] for r in history])), 1)

        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric("📊 Total Analyzed", total_h)
        hm2.metric("😊 Positive", pos_h)
        hm3.metric("😞 Negative", neg_h)
        hm4.metric("🎯 Avg Confidence", f"{avg_conf_h}%")

        st.markdown("---")

        # ── CONFIDENCE TREND CHART ─────────────────────────────────────────
        st.markdown("#### 📉 Confidence Trend Over Time")
        confs_h   = [r["confidence"] for r in history]
        labels_h  = [r["label"] for r in history]
        intens_h  = [r["intensity"] for r in history]
        times_h   = [r["time"] for r in history]
        colors_h  = ["#35c58b" if l == "positive" else "#f06a7a" for l in labels_h]
        x_labels  = [f"#{i+1}" for i in range(len(history))]

        fig_trend = go.Figure()
        # Shaded regions
        fig_trend.add_hrect(y0=80, y1=100, fillcolor="#35c58b", opacity=0.05, line_width=0)
        fig_trend.add_hrect(y0=50, y1=70,  fillcolor="#f06a7a", opacity=0.05, line_width=0)
        # Trend line
        fig_trend.add_trace(go.Scatter(
            x=x_labels, y=confs_h,
            mode="lines+markers",
            line=dict(color="#f5c518", width=2.5, dash="solid"),
            marker=dict(color=colors_h, size=10, line=dict(color="#f5c518", width=1.5)),
            customdata=list(zip(labels_h, intens_h, times_h)),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Confidence: %{y}%<br>"
                "Sentiment: %{customdata[0]}<br>"
                "Intensity: %{customdata[1]}<br>"
                "Time: %{customdata[2]}<extra></extra>"
            ),
            name="Confidence",
        ))
        # 80% threshold line
        fig_trend.add_hline(
            y=80, line_dash="dot", line_color="#8b93a8",
            annotation_text="80% threshold", annotation_font_color="#8b93a8",
        )
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            height=350,
            xaxis=dict(gridcolor="#2e3347", color="#8b93a8", title="Review #"),
            yaxis=dict(range=[48, 102], gridcolor="#2e3347", ticksuffix="%", title="Confidence"),
            title=dict(text="Confidence Score per Review (🟢 Positive  🔴 Negative)", font=dict(color="#f5c518")),
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")

        st.markdown("---")

        # ── CHARTS ROW ─────────────────────────────────────────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            fig_donut = go.Figure(go.Pie(
                labels=["Positive", "Negative"], values=[pos_h, neg_h],
                marker_colors=["#35c58b", "#f06a7a"], hole=0.5,
                textfont=dict(color="#ffffff"),
                hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
            ))
            fig_donut.update_layout(
                **PLOTLY_LAYOUT, height=280,
                title=dict(text="Sentiment Split", font=dict(color="#f5c518")),
                legend=dict(font=dict(color="#d1d6e4")),
            )
            st.plotly_chart(fig_donut, use_container_width=True, key="hist_donut")

        with ch2:
            intensity_hist = Counter(r["intensity"] for r in history)
            i_labels = list(intensity_hist.keys())
            i_vals   = list(intensity_hist.values())
            i_colors = ["#35c58b" if "Positive" in l else "#f06a7a" for l in i_labels]
            fig_intens = go.Figure(go.Bar(
                x=i_vals[::-1], y=i_labels[::-1], orientation="h",
                marker_color=i_colors[::-1],
                hovertemplate="%{y}: %{x} review(s)<extra></extra>",
            ))
            fig_intens.update_layout(
                **PLOTLY_LAYOUT, height=280,
                title=dict(text="Intensity Label Distribution", font=dict(color="#f5c518")),
                xaxis=dict(gridcolor="#2e3347"),
                yaxis=dict(gridcolor="#2e3347"),
            )
            st.plotly_chart(fig_intens, use_container_width=True, key="hist_intens")

        st.markdown("---")

        # ── CONFIDENCE DISTRIBUTION BUCKETS ────────────────────────────────
        st.markdown("#### 📈 Confidence Distribution")
        bins = {"< 60%": 0, "60–75%": 0, "75–90%": 0, "> 90%": 0}
        for r in history:
            c = r["confidence"]
            if c < 60:   bins["< 60%"]  += 1
            elif c < 75: bins["60–75%"] += 1
            elif c < 90: bins["75–90%"] += 1
            else:        bins["> 90%"]  += 1

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        for col, (rng, cnt) in zip([dcol1, dcol2, dcol3, dcol4], bins.items()):
            pct = round(cnt / total_h * 100) if total_h else 0
            with col:
                st.markdown(f"""<div class="stat-card">
                    <div class="stat-num">{cnt}</div>
                    <div class="stat-lbl">{rng} confidence</div>
                    <div style="color:#8b93a8;font-size:12px;">{pct}% of total</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── TOP KEYWORDS ───────────────────────────────────────────────────
        st.markdown("#### 🏆 Most Influential Words (this session)")
        all_pos_kws = Counter()
        all_neg_kws = Counter()
        for r in history:
            for word, score, direction in r.get("keywords", []):
                if direction == "positive": all_pos_kws[word] += 1
                else:                       all_neg_kws[word] += 1

        kw1, kw2 = st.columns(2)
        with kw1:
            st.markdown("**😊 Top Positive Words**")
            top_pos = all_pos_kws.most_common(8)
            if top_pos:
                fig_pos = go.Figure(go.Bar(
                    x=[v for _, v in top_pos[::-1]], y=[w for w, _ in top_pos[::-1]],
                    orientation="h", marker_color="#35c58b",
                    hovertemplate="%{y}: %{x} occurrence(s)<extra></extra>",
                ))
                fig_pos.update_layout(**PLOTLY_LAYOUT, height=260,
                    xaxis=dict(gridcolor="#2e3347"), yaxis=dict(gridcolor="#2e3347"))
                st.plotly_chart(fig_pos, use_container_width=True, key="kw_pos")
            else:
                st.caption("No data yet.")

        with kw2:
            st.markdown("**😞 Top Negative Words**")
            top_neg = all_neg_kws.most_common(8)
            if top_neg:
                fig_neg = go.Figure(go.Bar(
                    x=[v for _, v in top_neg[::-1]], y=[w for w, _ in top_neg[::-1]],
                    orientation="h", marker_color="#f06a7a",
                    hovertemplate="%{y}: %{x} occurrence(s)<extra></extra>",
                ))
                fig_neg.update_layout(**PLOTLY_LAYOUT, height=260,
                    xaxis=dict(gridcolor="#2e3347"), yaxis=dict(gridcolor="#2e3347"))
                st.plotly_chart(fig_neg, use_container_width=True, key="kw_neg")
            else:
                st.caption("No data yet.")

        st.markdown("---")

        # ── FULL HISTORY LIST ──────────────────────────────────────────────
        st.markdown("#### 🕒 All Analyzed Reviews")
        for i, r in enumerate(reversed(history)):
            orig_idx = len(history) - i
            label_badge = (
                '<span class="badge-pos">😊 Positive</span>'
                if r["label"] == "positive"
                else '<span class="badge-neg">😞 Negative</span>'
            )
            lang_flag = (
                f' <span style="color:#c97a10;font-size:12px;">[{html.escape(r["lang"].upper())}]</span>'
                if r.get("lang") and r["lang"] != "en" else ""
            )
            raw_text = r["text"]
            preview  = html.escape(raw_text[:90] + ("…" if len(raw_text) > 90 else ""))
            st.markdown(f"""
            <div class="hist-row">
                <span style="color:#8b93a8;min-width:24px;">#{orig_idx}</span>
                <span style="flex:1;margin:0 12px;">{preview}{lang_flag}</span>
                <span style="min-width:200px;text-align:right;">{label_badge}
                    <span style="color:#8b93a8;font-size:12px;"> · {r["intensity"]}</span>
                </span>
                <span style="min-width:70px;text-align:right;color:#f5c518;">{r["confidence"]}%</span>
                <span style="min-width:70px;text-align:right;color:#8b93a8;font-size:12px;">{r["time"]}</span>
            </div>""", unsafe_allow_html=True)


# =====================================================
# MODEL INFO EXPANDER
# =====================================================
st.write("")
with st.expander("ℹ️ About This Project & Model", expanded=False):
    about_c1, about_c2 = st.columns(2)

    with about_c1:
        st.markdown("""
        ### 📚 Project Overview
        **CineSense AI** is a Final Year Data Science project that performs
        sentiment analysis on IMDb movie reviews using classical NLP and ML.

        **Goal:** Classify a movie review as *Positive* or *Negative* with
        confidence scoring and intensity labelling.

        ---
        ### 🗂️ Dataset
        | Property | Value |
        |----------|-------|
        | Source   | IMDb Large Movie Review Dataset |
        | Size     | ~50,000 reviews |
        | Balance  | 50% Positive, 50% Negative |

        ---
        ### ⚙️ Model Pipeline
        | Step | Component | Details |
        |------|-----------|---------|
        | 1    | TF-IDF Vectorizer | Unigram tokenization (~95k vocabulary features) |
        | 2    | LinearSVC Classifier | Hinge loss, L2 regularization |
        """)

    with about_c2:
        st.markdown("""
        ### 📊 Performance Metrics
        | Metric    | Score   |
        |-----------|---------|
        | Accuracy  | ~89–91% |
        | Precision | ~90%    |
        | Recall    | ~90%    |
        | F1 Score  | ~90%    |

        ---
        ### 🛠️ Technology Stack
        | Layer         | Technology     |
        |---------------|----------------|
        | Language      | Python 3.x     |
        | ML Library    | Scikit-learn   |
        | Text Features | TF-IDF         |
        | Classifier    | LinearSVC      |
        | Charts        | Plotly         |
        | Lang Detect   | langdetect     |
        | UI Framework  | Streamlit      |

        ---
        ### 🔬 Key Concepts Used
        - **TF-IDF**: Term Frequency-Inverse Document Frequency
        - **LinearSVC**: Support Vector Classification (linear kernel)
        - **Decision Function**: Sigmoid-mapped confidence scores
        - **Intensity Labels**: Granular sentiment scale
        - **Live Preview**: Real-time inference as the user types
        - **Language Detection**: Non-English review flagging
        """)


# =====================================================
# FOOTER
# =====================================================
st.markdown("""
<div class="footer">
    <strong>🎬 CineSense AI — IMDb Movie Review Sentiment Analysis</strong><br>
    Final Year Data Science Mini Project<br>
    Built with Python &bull; Scikit-learn &bull; TF-IDF &bull; LinearSVC &bull; Plotly &bull; Streamlit
</div>
""", unsafe_allow_html=True)
