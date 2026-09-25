
import os
import html

from modules import fusion_inference
import streamlit as st
import pandas as pd
import plotly.graph_objects as go



# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL_DIR = "./bert_fake_reviews_model"
DEFAULT_FUSION_DIR = "./fusion_model"

MODEL_CHOICES = [
    "DistilBERT (Transformer)",
    "Fusion - Logistic Regression",
    "Fusion - Linear SVM",
]

INK = "#12181C"
PAPER = "#1B2328"
PAPER_RAISED = "#212B31"
HAIRLINE = "#2E3A41"
FOG = "#8DA0A8"
BONE = "#E9EDEE"
VERIFIED = "#5FD3A0"   # green -> "real" everywhere in this app
FLAGGED = "#E64980"    # pink  -> "fake" everywhere in this app
SIGNAL = "#F4C95D"

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
FAKE_CLASS_INDEX = 1
REAL_CLASS_INDEX = 0

st.set_page_config(
    page_title="Detecting Deception in Product Reviews",
    page_icon="🔎",
    layout="wide",
)


st.markdown(
    f"""
<style>

    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Manrope', sans-serif;
    }}

    .stApp {{
        background-color: {INK};
        color: {BONE};
    }}

    .mono {{
        font-family: 'IBM Plex Mono', monospace;
    }}

    h1, h2, h3 {{
        color: {BONE} !important;
        font-weight: 800 !important;
    }}

    .case-tag {{
        display:inline-flex;
        align-items:center;
        gap:8px;
        font-family:'IBM Plex Mono', monospace;
        font-size:12px;
        letter-spacing:0.1em;
        color:{FOG};
        border:1px solid {HAIRLINE};
        padding:6px 12px;
        border-radius:4px;
        margin-bottom:16px;
    }}

    .dot {{
        width:7px;
        height:7px;
        border-radius:50%;
        display:inline-block;
        background:{VERIFIED};
    }}

    .metric-card {{
        background:{PAPER};
        border:1px solid {HAIRLINE};
        border-radius:8px;
        padding:18px 20px;
        text-align:left;
    }}

    .metric-card .val {{
        font-family:'IBM Plex Mono', monospace;
        font-size:26px;
        font-weight:700;
        color:{BONE};
    }}

    .metric-card .lbl {{
        font-size:12px;
        color:{FOG};
        margin-top:4px;
    }}

    .verdict-real {{
        background: rgba(95,211,160,0.12);
        border:1px solid rgba(95,211,160,0.4);
        border-radius:8px;
        padding:20px 24px;
    }}

    .verdict-fake {{
        background: rgba(230,73,128,0.12);
        border:1px solid rgba(230,73,128,0.4);
        border-radius:8px;
        padding:20px 24px;
    }}

    .verdict-title {{
        font-family:'IBM Plex Mono', monospace;
        font-weight:700;
        font-size:20px;
    }}

    .verdict-sub {{
        font-size:13px;
        color:{FOG};
        margin-top:4px;
    }}

    .exhibit {{
        background:{PAPER};
        border:1px solid {HAIRLINE};
        border-radius:8px;
        padding:18px 20px;
        margin-bottom:12px;
    }}

    .stamp-fp {{
        font-family:'IBM Plex Mono', monospace;
        font-size:11px;
        font-weight:700;
        letter-spacing:0.06em;
        color:{FLAGGED};
        border:1px solid {FLAGGED};
        padding:3px 9px;
        border-radius:3px;
    }}

    .stamp-fn {{
        font-family:'IBM Plex Mono', monospace;
        font-size:11px;
        font-weight:700;
        letter-spacing:0.06em;
        color:{SIGNAL};
        border:1px solid {SIGNAL};
        padding:3px 9px;
        border-radius:3px;
    }}

    .exhibit-text {{
        font-size:14px;
        color:{BONE};
        border-left:2px solid {HAIRLINE};
        padding-left:12px;
        margin:10px 0;
    }}

    .exhibit-note {{
        font-size:12.5px;
        color:{FOG};
    }}

    .stTextArea textarea {{
        background:{PAPER_RAISED} !important;
        color:{BONE} !important;
        border:1px solid {HAIRLINE} !important;
        font-family:'Manrope', sans-serif;
    }}

    .stButton button {{
        background:{BONE};
        color:{INK};
        font-family:'IBM Plex Mono', monospace;
        font-weight:600;
        border:none;
        border-radius:8px;
        padding:10px 20px;
    }}

    .stButton button:hover {{
        opacity:0.85;
        color:{INK};
    }}

    section[data-testid="stSidebar"] {{
        background-color:{PAPER};
    }}

    .shap-explanation {{
        background:{PAPER};
        border:1px solid {HAIRLINE};
        border-radius:8px;
        padding:20px;
        margin-top:20px;
    }}

    .shap-title {{
        font-family:'IBM Plex Mono', monospace;
        font-size:18px;
        font-weight:700;
        color:{BONE};
    }}

    .shap-description {{
        font-size:13px;
        color:{FOG};
        margin-top:5px;
        margin-bottom:15px;
        line-height:1.6;
    }}

    .shap-positive {{
        color:{FLAGGED};
        font-weight:700;
    }}

    .shap-negative {{
        color:{VERIFIED};
        font-weight:700;
    }}

</style>
""",
    unsafe_allow_html=True,
)


def render(html_str: str):
    """Render a raw HTML string. Never build multi-line HTML with
    blank lines in it and pass it here - see the module docstring."""
    st.markdown(html_str, unsafe_allow_html=True)


def case_tag(label: str):
    render(
        f'<div class="case-tag"><span class="dot"></span>{html.escape(label)}</div>'
    )


def metric_card(value, label: str, color: str = None, container=st):
    color_style = f'color:{color};' if color else ''
    container.markdown(
        f'<div class="metric-card">'
        f'<div class="val" style="{color_style}">{html.escape(str(value))}</div>'
        f'<div class="lbl">{html.escape(label)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def verdict_box(is_real: bool, confidence: float):
    css_class = "verdict-real" if is_real else "verdict-fake"
    color = VERIFIED if is_real else FLAGGED
    title = "✓ LIKELY GENUINE" if is_real else "⚑ LIKELY AI-GENERATED"
    render(
        f'<div class="{css_class}">'
        f'<div class="verdict-title" style="color:{color}">{title}</div>'
        f'<div class="verdict-sub">Model confidence: {confidence * 100:.1f}%</div>'
        f'</div>'
    )


def exhibit_box(stamp_text: str, stamp_class: str, quote: str, note_label: str, note: str):
    safe_quote = html.escape(quote)
    safe_note = html.escape(note)
    render(
        f'<div class="exhibit">'
        f'<span class="{stamp_class}">{html.escape(stamp_text)}</span>'
        f'<div class="exhibit-text">&ldquo;{safe_quote}&rdquo;</div>'
        f'<div class="exhibit-note"><b>{html.escape(note_label)}</b> {safe_note}</div>'
        f'</div>'
    )


def shap_legend_box():
    render(
        '<div class="shap-explanation">'
        '<div class="shap-title">🔬 How to read this chart</div>'
        '<div class="shap-description">'
        f'<span class="shap-positive">Pink bars</span> push the prediction toward <b>FAKE</b>. '
        f'<span class="shap-negative">Green bars</span> push it toward <b>REAL</b>.'
        '<br><br>'
        'This coloring is always the same regardless of what the model actually predicted for '
        'this review, so pink always means "fake signal" and green always means "real signal" - '
        'the same as everywhere else in this app.'
        '<br><br>'
        'Longer bars had a bigger effect on this specific prediction. This shows what the model '
        'reacted to, not proof that the review really is real or fake.'
        '</div>'
        '</div>'
    )


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_dir: str):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        st.error(
            "Missing dependency: `torch` and `transformers` are required to load the "
            f"model. Install them with `pip install torch transformers`.\n\n({e})"
        )
        return None, None, None

    if not os.path.isdir(model_dir):
        return None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()
    except Exception as e:
        st.error(f"Found the folder, but couldn't load a model from it: {e}")
        return None, None, None

    return model, tokenizer, device


@st.cache_resource(show_spinner=False)
def load_fusion_artifacts_cached(model_dir: str):
    """Thin cache_resource wrapper - fusion_inference.py itself has no
    Streamlit dependency, same reasoning as create_shap_explainer()."""
    try:
        import joblib  # noqa: F401 - surfaced here so the error is clear if missing
        import sklearn  # noqa: F401
    except ImportError as e:
        st.error(
            "Missing dependency: `scikit-learn` and `joblib` are required for the "
            f"fusion model. Install them with `pip install scikit-learn joblib`.\n\n({e})"
        )
        return None

    return fusion_inference.load_fusion_artifacts(model_dir)


# ============================================================
# PREDICTION
# ============================================================

def predict(text: str, model, tokenizer, device, max_length: int = 256):
    import torch

    id2label = {REAL_CLASS_INDEX: "real", FAKE_CLASS_INDEX: "fake"}

    with torch.no_grad():
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())

    return {
        "label": id2label[pred_id],
        "confidence": float(probs[pred_id].item()),
        "prob_real": float(probs[REAL_CLASS_INDEX].item()),
        "prob_fake": float(probs[FAKE_CLASS_INDEX].item()),
    }


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource(show_spinner=False)
def create_shap_explainer(model_dir: str):
    """
    Creates the SHAP explainer.

    IMPORTANT:
    We pass model_dir instead of model/tokenizer as arguments because
    Streamlit cannot hash HuggingFace model/tokenizer objects.
    """
    try:
        import torch
        import shap
    except ImportError as e:
        st.error(
            "Missing dependency: `shap` is required for explanations. "
            f"Install it with `pip install shap`.\n\n({e})"
        )
        return None

    model, tokenizer, device = load_model(model_dir)
    if model is None:
        return None

    def model_predict(texts):
        texts = list(texts)  # SHAP may pass numpy arrays or tuples
        encoded = tokenizer(
            texts, truncation=True, padding=True, max_length=256, return_tensors="pt"
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities.cpu().numpy()

    masker = shap.maskers.Text(tokenizer)
    return shap.Explainer(model_predict, masker)


# ============================================================
# SHAP TOKEN EXTRACTION
# ============================================================

def get_shap_token_importance(explainer, text: str) -> pd.DataFrame:
    """
    Runs SHAP on `text` and returns a dataframe of merged, human-readable
    word-level contributions.

    Values are always relative to the FAKE class (index 1), regardless of
    what the model actually predicted for this review:
        positive shap_value -> pushed the model toward FAKE
        negative shap_value -> pushed the model toward REAL
    Keeping this fixed (instead of relative to "whichever class won")
    is what stops the bar colors from flipping meaning between reviews.
    """
    shap_values = explainer([text])

    # shape: (1 sample, n_tokens, n_classes)
    values = shap_values.values[0, :, FAKE_CLASS_INDEX]
    tokens = shap_values.data[0]

    raw_rows = []
    for token, value in zip(tokens, values):
        token = str(token)
        if not token.strip():
            continue
        if token in SPECIAL_TOKENS:
            continue
        raw_rows.append({"token": token, "shap_value": float(value)})

    if not raw_rows:
        return pd.DataFrame(columns=["display_token", "shap_value"])

    # Merge WordPiece subword continuations ("comfort", "##able" -> "comfortable"),
    # summing their contributions so each bar in the chart is a whole word.
    merged = []
    for row in raw_rows:
        token, value = row["token"], row["shap_value"]
        if token.startswith("##") and merged:
            merged[-1]["display_token"] += token[2:]
            merged[-1]["shap_value"] += value
        else:
            merged.append({"display_token": token.strip(), "shap_value": value})

    return pd.DataFrame(merged)


# ============================================================
# SHAP VISUALIZATION
# ============================================================

def show_shap_visualization(
    shap_df: pd.DataFrame,
    predicted_label: str,
    confidence: float,
    method_note: str = None,
):
    """Renders a word/feature contribution chart. Works for either the
    sampled SHAP values from DistilBERT or the exact linear contributions
    from a fusion model - both are passed in as a DataFrame with columns
    ["display_token", "shap_value"], positive = pushes toward FAKE."""

    if shap_df.empty:
        st.warning(
            "No word/feature contributions could be extracted (the review may be too short)."
        )
        return

    positive = shap_df[shap_df["shap_value"] > 0].sort_values("shap_value", ascending=False).head(10)
    negative = shap_df[shap_df["shap_value"] < 0].sort_values("shap_value", ascending=True).head(10)
    explanation_df = pd.concat([negative, positive])

    if explanation_df.empty:
        st.info("No strong word/feature contributions were found for this review.")
        return

    explanation_df = explanation_df.sort_values("shap_value")

    colors = [FLAGGED if v > 0 else VERIFIED for v in explanation_df["shap_value"]]

    fig = go.Figure(
        go.Bar(
            x=explanation_df["shap_value"],
            y=explanation_df["display_token"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in explanation_df["shap_value"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Which words/features pushed this review toward FAKE vs. REAL",
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(color=FOG, family="Manrope"),
        height=460,
        margin=dict(l=20, r=80, t=60, b=50),
        xaxis=dict(
            title="← pushes toward REAL          pushes toward FAKE →",
            zeroline=True,
            zerolinecolor=HAIRLINE,
            gridcolor=HAIRLINE,
        ),
        yaxis=dict(title=""),
        showlegend=False,
    )

    st.caption(f"Model predicted **{predicted_label.upper()}** with {confidence * 100:.1f}% confidence.")
    if method_note:
        st.caption(method_note)
    st.plotly_chart(fig, use_container_width=True)

    shap_legend_box()

    st.markdown("### 🔍 Most influential words / features")

    top_tokens = (
        shap_df.assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(10)[["display_token", "shap_value"]]
        .copy()
    )
    top_tokens["direction"] = top_tokens["shap_value"].apply(lambda v: "→ FAKE" if v > 0 else "→ REAL")
    top_tokens["shap_value"] = top_tokens["shap_value"].round(4)
    top_tokens.columns = ["Word / Feature", "Contribution", "Pushes toward"]

    st.dataframe(top_tokens, use_container_width=True, hide_index=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Model Settings")

    model_dir = st.text_input("DistilBERT model folder path", value=DEFAULT_MODEL_DIR)
    st.caption("Should contain config.json, model.safetensors, tokenizer files.")

    fusion_dir = st.text_input("Fusion model folder path", value=DEFAULT_FUSION_DIR)
    st.caption(
        "Should contain tfidf_vectorizer.joblib, scaler.joblib, "
        "logistic_regression.joblib, linear_svm.joblib, metadata.json "
        "- produced by running train_fusion_model.py."
    )

    st.divider()

    st.markdown("### 📁 Navigate")

    page = st.radio(
        "Section",
        [
            "Overview",
            "The Dataset",
            "The Investigation",
            "The Verdict",
            "Case Notes",
            "🔎 Live Interrogation",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.caption("Capstone Project in Data Science II (DS4105) - Sabaragamuwa University of Sri Lanka")


# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "Overview":

    case_tag("CASE FILE - DS4105 CAPSTONE PROJECT")

    st.title("Detecting deception in product reviews")

    st.markdown(
        "A full investigation trail - from raw review data to a fine-tuned transformer - "
        "built to tell genuine customer reviews apart from AI-generated ones."
    )

    cols = st.columns(4)
    stats = [
        ("40,405", "reviews analyzed"),
        ("10", "product categories"),
        ("4", "modeling approaches tested"),
        ("98.3%", "best model accuracy"),
    ]
    for col, (val, lbl) in zip(cols, stats):
        metric_card(val, lbl, container=col)

    st.markdown("")

    st.info(
        "Use the sidebar to navigate between the EDA report, model comparison, "
        "results, and the live classifier demo."
    )


# ============================================================
# PAGE: THE DATASET
# ============================================================

elif page == "The Dataset":

    case_tag("01 - THE DATASET")

    st.header("40,405 reviews, near-perfectly balanced")
    st.caption("10 product categories, real Amazon reviews vs. GPT-2-generated fakes.")

    col1, col2 = st.columns([1, 2])

    with col1:
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Real", "Fake"],
                    values=[20215, 20190],
                    hole=0.65,
                    marker=dict(colors=[VERIFIED, FLAGGED]),
                    textfont=dict(color=BONE, family="IBM Plex Mono"),
                )
            ]
        )
        fig.update_layout(
            title="Class Balance",
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=FOG),
            height=300,
            margin=dict(t=40, b=10, l=10, r=10),
            showlegend=True,
            legend=dict(font=dict(color=FOG)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(
            data=[
                go.Bar(name="Real", x=["avg. words per review"], y=[73.6], marker_color=VERIFIED),
                go.Bar(name="Fake", x=["avg. words per review"], y=[61.3], marker_color=FLAGGED),
            ]
        )
        fig.update_layout(
            title="Review Length by Label",
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=FOG),
            height=300,
            margin=dict(t=40, b=10, l=10, r=10),
            barmode="group",
            legend=dict(font=dict(color=FOG)),
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        categories = [
            "Kindle Store", "Books", "Pet Supplies", "Home & Kitchen", "Electronics",
            "Sports & Outdoors", "Tools & Home Impr.", "Clothing/Shoes/Jewelry",
            "Toys & Games", "Movies & TV",
        ]
        counts = [4700, 4400, 4250, 4050, 4000, 3950, 3900, 3850, 3800, 3600]

        fig = go.Figure(data=[go.Bar(x=counts, y=categories, orientation="h", marker_color="#7BB6C9")])
        fig.update_layout(
            title="Reviews per Category (approx.)",
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=FOG),
            height=380,
            margin=dict(t=40, b=10, l=10, r=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = go.Figure(
            data=[
                go.Bar(name="Fake", x=["1", "2", "3", "4", "5"], y=[1050, 950, 1950, 3950, 12250], marker_color=FLAGGED),
                go.Bar(name="Real", x=["1", "2", "3", "4", "5"], y=[1050, 950, 1850, 4050, 12250], marker_color=VERIFIED),
            ]
        )
        fig.update_layout(
            title="Rating Distribution by Label",
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=FOG),
            height=380,
            margin=dict(t=40, b=10, l=10, r=10),
            barmode="group",
            legend=dict(font=dict(color=FOG)),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Rating is near-identical across both classes - not a useful standalone fake-detection signal.")


# ============================================================
# PAGE: THE INVESTIGATION
# ============================================================

elif page == "The Investigation":

    case_tag("02 - THE INVESTIGATION")

    st.header("Four leads pursued")
    st.caption(
        "From simple hand-built features to a fine-tuned transformer, "
        "each tested fairly on the same held-out data."
    )

    stages = [
        "Engineered Features\n(Random Forest)",
        "TF-IDF\n(Linear SVM)",
        "Fusion Model\n(TF-IDF + Engineered)",
        "DistilBERT\n(fine-tuned)",
    ]
    scores = [85.27, 87.82, 90.89, 98.05]
    colors = [FOG, "#7BB6C9", SIGNAL, VERIFIED]

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=stages,
                orientation="h",
                marker_color=colors,
                text=[f"{s}%" for s in scores],
                textposition="outside",
                textfont=dict(family="IBM Plex Mono", color=BONE),
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(color=FOG),
        height=380,
        margin=dict(t=20, b=20, l=10, r=60),
        xaxis=dict(range=[0, 110], title="Accuracy (%)"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# PAGE: THE VERDICT
# ============================================================

elif page == "The Verdict":

    case_tag("03 - THE VERDICT")

    st.header("DistilBERT, evaluated on 6,061 held-out reviews")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Confusion Matrix")

        cm = [[2963, 70], [33, 2995]]
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=cm,
                    x=["Predicted Real", "Predicted Fake"],
                    y=["Actual Real", "Actual Fake"],
                    colorscale=[[0, PAPER_RAISED], [1, VERIFIED]],
                    text=cm,
                    texttemplate="%{text}",
                    textfont=dict(size=20, family="IBM Plex Mono", color=BONE),
                    showscale=False,
                )
            ]
        )
        fig.update_layout(
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=FOG),
            height=320,
            margin=dict(t=20, b=20, l=10, r=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### Test Set Metrics")

        m1, m2 = st.columns(2)
        metric_card("98.05%", "Accuracy", color=VERIFIED, container=m1)
        metric_card("98.07%", "F1 (fake class)", color=VERIFIED, container=m2)

        st.markdown("")

        m3, m4 = st.columns(2)
        metric_card("97.03%", "Precision (fake)", color=VERIFIED, container=m3)
        metric_card("99.14%", "Recall (fake)", color=VERIFIED, container=m4)

        st.markdown("")

        st.caption(
            "A ~7-point accuracy gain over the best classical model (fusion, 90.89%) - "
            "but likely inflated by GPT-2's comparatively easy-to-detect generation artifacts. "
            "See Case Notes."
        )


# ============================================================
# PAGE: CASE NOTES
# ============================================================

elif page == "Case Notes":

    case_tag("04 - CASE NOTES")

    st.header("103 of 6,061 test reviews were misclassified")
    st.caption("Reading the actual mistakes reveals what the model learned - and what fools it.")

    tab1, tab2 = st.tabs(["False Positives (70)", "False Negatives (33)"])

    false_positives = [
        (
            "I enjoyed the two stories contained in this book. "
            "These are the only books by these authors that I have read.",
            "Short, blunt, factual - no personal elaboration. The model appears to associate "
            "brevity and genericness with AI generation, penalizing naturally terse human writers.",
        ),
        (
            "They fit perfect, they look expensive, they are the most comfortable shoes "
            "that i had ever. I love the design.",
            "Enthusiastic but generic praise, close to the kind of language GPT-2 tends to generate.",
        ),
        (
            "I have two of these, one for the kitchen and one for the dining room.",
            "Minimal, matter-of-fact - the shortest kind of real review, easily confused "
            "with generic filler text.",
        ),
        (
            "Not at all what I was expecting. This was a large box of plastic pieces that "
            "don't fit together. I'm sure the only way this toy would be usable is if you "
            "glued the pieces together.",
            "A genuine complaint, but structured plainly enough to read as generic negative filler.",
        ),
        (
            "Very comfortable, just a little short for my taste. I also have long legs, "
            "so that might be the issue.",
            "Specific personal detail (long legs) still wasn't enough to overcome the "
            "short-review heuristic.",
        ),
    ]

    false_negatives = [
        (
            "We used these with 3 of my 3 year old grandkids. They really enjoy playing with them.",
            "Specific, human-sounding family detail - exactly the kind of anecdotal texture "
            "real reviews have.",
        ),
        (
            "Pros: Wash well. Laundry bag included, thicker than expected, great quality "
            "materials. Cons: Too small for my petite frame, I had to return it.",
            "Mimics a genuine human review convention - structured pros/cons lists are common "
            "in real reviews, so the model reads structure as authenticity.",
        ),
        (
            "Received product at discount for honest review. Waking Up by Kirsten Clare. "
            "I am a huge fan of both Haus of Tars and The Blind Side.",
            "Borrows a real, common Amazon disclosure phrase - a genuine trust signal the "
            "fake text co-opts convincingly.",
        ),
        (
            "Contains raw embedded HTML markup (a product link tag) rather than natural review prose.",
            "Likely not a genuine model failure - this looks like a data leakage or scraping "
            "artifact mislabeled as 'fake' in the source dataset.",
        ),
        (
            "SHEET COLOR IS NICE BUY FOR MY SIZE FOR MY TOWN.",
            "Ungrammatical, broken English - the model likely treats imperfect fluency as a "
            "human tell, which a fake review can exploit.",
        ),
    ]

    with tab1:
        for text, note in false_positives:
            exhibit_box("REAL FLAGGED AS FAKE", "stamp-fp", text, "Why it fooled the model:", note)

    with tab2:
        for text, note in false_negatives:
            exhibit_box("FAKE MISSED AS REAL", "stamp-fn", text, "Why it slipped through:", note)


# ============================================================
# PAGE: LIVE INTERROGATION
# ============================================================

elif page == "🔎 Live Interrogation":

    case_tag("LIVE MODEL")

    st.header("Try it yourself")
    st.caption("Paste a product review below and pick which model classifies it.")

    model_choice = st.selectbox("Model", MODEL_CHOICES)
    is_fusion = model_choice.startswith("Fusion")
    fusion_which = "logreg" if "Logistic" in model_choice else "svm"

    text = st.text_area(
        "Review text",
        height=150,
        placeholder=(
            "e.g. I bought this for my kitchen and it works great, the build quality "
            "is solid and it has lasted me over a year without any issues at all."
        ),
    )

    word_count = len(text.split()) if text.strip() else 0
    st.caption(f"{word_count} words (minimum 3 required)")

    if st.button("Analyze Review", disabled=(word_count < 3)):

        # ------------------------------------------------------
        # FUSION MODELS (TF-IDF + engineered features, linear classifier)
        # ------------------------------------------------------
        if is_fusion:

            with st.spinner("Loading fusion model..."):
                artifacts = load_fusion_artifacts_cached(fusion_dir)

            if artifacts is None:
                st.error(
                    f"Could not find a fusion model at: `{fusion_dir}`\n\n"
                    "Run `train_fusion_model.py` first to produce tfidf_vectorizer.joblib, "
                    "scaler.joblib, logistic_regression.joblib, linear_svm.joblib and "
                    "metadata.json, then point the sidebar path at that folder."
                )

            else:
                st.success("Fusion model loaded successfully.")

                with st.spinner("Running model prediction..."):
                    result = fusion_inference.predict_fusion(text, artifacts, which=fusion_which)

                verdict_box(is_real=(result["label"] == "real"), confidence=result["confidence"])

                st.markdown("")

                colr, colf = st.columns(2)
                metric_card(f"{result['prob_real'] * 100:.1f}%", "Real probability", color=VERIFIED, container=colr)
                metric_card(f"{result['prob_fake'] * 100:.1f}%", "Fake probability", color=FLAGGED, container=colf)

                st.divider()

                st.markdown("## 🔬 Why did the model make this prediction?")
                st.caption(
                    "Exact contribution of each TF-IDF term and engineered feature "
                    "(coefficient × feature value) toward this prediction."
                )

                try:
                    explain_df = fusion_inference.explain_fusion(text, artifacts, which=fusion_which)
                    note = (
                        "Linear SVM values are averaged across the model's calibration "
                        "folds - a close approximation, not an exact decomposition."
                        if fusion_which == "svm"
                        else None
                    )
                    show_shap_visualization(explain_df, result["label"], result["confidence"], method_note=note)
                except Exception as e:
                    st.error("Explanation could not be generated.")
                    st.exception(e)
                    st.info(
                        "The model prediction itself is still valid. The error is only "
                        "related to the explanation layer."
                    )

                st.divider()

                st.info(
                    "Important: this explains the model's reasoning patterns. It does not "
                    "prove that a review is actually genuine or AI-generated. A high model "
                    "confidence can still correspond to an incorrect prediction."
                )

        # ------------------------------------------------------
        # DISTILBERT (fine-tuned transformer)
        # ------------------------------------------------------
        else:

            with st.spinner("Loading model..."):
                model, tokenizer, device = load_model(model_dir)

            if model is None:
                st.error(
                    f"Could not find a model at: `{model_dir}`\n\n"
                    "Check the path in the sidebar - it should point to the folder containing "
                    "config.json, model.safetensors, tokenizer_config.json, and tokenizer.json."
                )

            else:
                st.success(f"Model loaded successfully on **{device.upper()}**.")

                with st.spinner("Running model prediction..."):
                    result = predict(text, model, tokenizer, device)

                verdict_box(is_real=(result["label"] == "real"), confidence=result["confidence"])

                st.markdown("")

                colr, colf = st.columns(2)
                metric_card(f"{result['prob_real'] * 100:.1f}%", "Real probability", color=VERIFIED, container=colr)
                metric_card(f"{result['prob_fake'] * 100:.1f}%", "Fake probability", color=FLAGGED, container=colf)

                st.divider()

                st.markdown("## 🔬 Why did the model make this prediction?")
                st.caption(
                    "SHAP highlights the words that influenced the DistilBERT prediction "
                    "for this individual review."
                )

                try:
                    with st.spinner("Generating SHAP explanation... this can take a little while."):
                        explainer = create_shap_explainer(model_dir)

                        if explainer is None:
                            st.error("Could not create the SHAP explainer.")
                        else:
                            shap_df = get_shap_token_importance(explainer, text)
                            show_shap_visualization(shap_df, result["label"], result["confidence"])

                except Exception as e:
                    st.error("SHAP explanation could not be generated.")
                    st.exception(e)
                    st.info(
                        "The model prediction itself is still valid. The error is only "
                        "related to the explanation layer."
                    )

                st.divider()

                st.info(
                    "Important: SHAP explains the model's reasoning patterns. It does not "
                    "prove that a review is actually genuine or AI-generated. A high model "
                    "confidence can still correspond to an incorrect prediction."
                )
