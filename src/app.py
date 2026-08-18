import os
import html

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL_DIR = "./bert_fake_reviews_model"

INK = "#12181C"
PAPER = "#1B2328"
PAPER_RAISED = "#212B31"
HAIRLINE = "#2E3A41"
FOG = "#8DA0A8"
BONE = "#E9EDEE"
VERIFIED = "#5FD3A0"
FLAGGED = "#E64980"
SIGNAL = "#F4C95D"


st.set_page_config(
    page_title="Detecting Deception in Product Reviews",
    page_icon="🔎",
    layout="wide",
)


# ============================================================
# THEME / CSS
# ============================================================

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


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_dir: str):

    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification
    )

    if not os.path.isdir(model_dir):
        return None, None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir
    ).to(device)

    model.eval()

    return model, tokenizer, device


# ============================================================
# NORMAL PREDICTION
# ============================================================

def predict(
    text: str,
    model,
    tokenizer,
    device,
    max_length: int = 256
):

    import torch

    id2label = {
        0: "real",
        1: "fake"
    }

    with torch.no_grad():

        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        logits = model(**enc).logits

        probs = torch.softmax(
            logits,
            dim=-1
        )[0]

        pred_id = int(
            torch.argmax(probs).item()
        )

    return {
        "label": id2label[pred_id],
        "confidence": float(
            probs[pred_id].item()
        ),
        "prob_real": float(
            probs[0].item()
        ),
        "prob_fake": float(
            probs[1].item()
        ),
    }


# ============================================================
# SHAP EXPLAINER
# ============================================================

@st.cache_resource(show_spinner=False)
def create_shap_explainer(model_dir: str):

    """
    Creates the SHAP explainer.

    IMPORTANT:
    We pass model_dir instead of model/tokenizer as arguments
    because Streamlit cannot hash HuggingFace model/tokenizer
    objects.
    """

    import torch
    import shap

    model, tokenizer, device = load_model(model_dir)

    if model is None:
        return None

    def model_predict(texts):

        # SHAP may pass numpy arrays or tuples.
        texts = list(texts)

        encoded = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            logits = model(
                **encoded
            ).logits

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

        return probabilities.cpu().numpy()

    # Text masker
    masker = shap.maskers.Text(tokenizer)

    # Create SHAP explainer
    explainer = shap.Explainer(
        model_predict,
        masker
    )

    return explainer


# ============================================================
# SHAP TOKEN EXTRACTION
# ============================================================

def get_shap_token_importance(
    explainer,
    text,
    predicted_class
):

    """
    Calculates SHAP values and converts them into a
    simple token/contribution dataframe.
    """

    shap_values = explainer(
        [text]
    )

    # SHAP output usually has:
    #
    # samples x tokens x classes
    #
    # Example:
    # (1, 23, 2)

    values = shap_values.values

    tokens = shap_values.data[0]

    # Select SHAP values for the predicted class
    class_values = values[0, :, predicted_class]

    rows = []

    for token, value in zip(
        tokens,
        class_values
    ):

        token = str(token)

        # Remove empty tokens
        if not token.strip():
            continue

        # Remove common special tokens
        if token in [
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "<s>",
            "</s>",
            "<pad>"
        ]:
            continue

        rows.append(
            {
                "token": token,
                "shap_value": float(value)
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    # Combine subword tokens where possible
    result["display_token"] = (
        result["token"]
        .str.replace("##", "", regex=False)
        .str.strip()
    )

    return result


# ============================================================
# CREATE SHAP VISUALIZATION
# ============================================================

def show_shap_visualization(
    shap_df,
    predicted_label
):

    if shap_df.empty:

        st.warning(
            "No SHAP token contributions could be extracted."
        )

        return

    # --------------------------------------------------------
    # Top positive and negative contributions
    # --------------------------------------------------------

    positive = (
        shap_df[
            shap_df["shap_value"] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False
        )
        .head(10)
    )

    negative = (
        shap_df[
            shap_df["shap_value"] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True
        )
        .head(10)
    )

    # Combine
    explanation_df = pd.concat(
        [
            negative,
            positive
        ]
    )

    if explanation_df.empty:

        st.info(
            "SHAP did not find strong token-level contributions."
        )

        return

    # --------------------------------------------------------
    # Sort for chart
    # --------------------------------------------------------

    explanation_df = explanation_df.copy()

    explanation_df = explanation_df.sort_values(
        "shap_value"
    )

    # --------------------------------------------------------
    # Plotly horizontal bar chart
    # --------------------------------------------------------

    fig = go.Figure()

    for _, row in explanation_df.iterrows():

        value = row["shap_value"]

        if value > 0:

            bar_color = FLAGGED

        else:

            bar_color = VERIFIED

        fig.add_trace(
            go.Bar(
                x=[value],
                y=[row["display_token"]],
                orientation="h",
                marker_color=bar_color,
                text=[f"{value:+.3f}"],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "SHAP contribution: %{x:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(

        title={
            "text": (
                f"Token-level explanation for "
                f"<b>{predicted_label.upper()}</b> prediction"
            ),
            "font": {
                "size": 18
            }
        },

        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,

        font=dict(
            color=FOG,
            family="Manrope"
        ),

        height=500,

        margin=dict(
            l=20,
            r=80,
            t=70,
            b=40
        ),

        xaxis=dict(
            title="SHAP contribution",
            zeroline=True,
            zerolinecolor=HAIRLINE,
            gridcolor=HAIRLINE,
        ),

        yaxis=dict(
            title="Token / word"
        ),

        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="shap-explanation">

        <div class="shap-title">
        🔬 How to read this explanation
        </div>

        <div class="shap-description">

        <span class="shap-positive">
        Pink / positive values
        </span>
        push the model toward the predicted
        <b>{predicted_label.upper()}</b> class.

        <br><br>

        <span class="shap-negative">
        Green / negative values
        </span>
        push the model away from the predicted
        <b>{predicted_label.upper()}</b> class.

        <br><br>

        Larger absolute SHAP values indicate stronger
        influence on this particular prediction.

        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Top contributing tokens
    # --------------------------------------------------------

    st.markdown("### 🔍 Most influential tokens")

    top_tokens = (
        shap_df
        .assign(
            absolute_shap=lambda x:
            x["shap_value"].abs()
        )
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .head(10)
        [
            [
                "display_token",
                "shap_value"
            ]
        ]
        .copy()
    )

    top_tokens.columns = [
        "Token",
        "SHAP contribution"
    ]

    top_tokens["SHAP contribution"] = (
        top_tokens["SHAP contribution"]
        .round(4)
    )

    st.dataframe(
        top_tokens,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### ⚙️ Model Settings")

    model_dir = st.text_input(
        "Model folder path",
        value=DEFAULT_MODEL_DIR
    )

    st.caption(
        "Should contain config.json, model.safetensors, "
        "tokenizer files."
    )

    st.markdown("---")

    st.markdown("### 📁 Navigate")

    page = st.radio(
        "Section",
        [
            "Overview",
            "The Dataset",
            "The Investigation",
            "The Verdict",
            "Case Notes",
            "🔎 Live Interrogation"
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.caption(
        "Capstone Project in Data Science II "
        "(DS3206) - Sabaragamuwa University of Sri Lanka"
    )


# ============================================================
# PAGE: OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="case-tag">'
        '<span class="dot"></span>'
        'CASE FILE - DS3206 CAPSTONE PROJECT'
        '</div>',
        unsafe_allow_html=True
    )

    st.title(
        "Detecting deception in product reviews"
    )

    st.markdown(
        "A full investigation trail - from raw review data "
        "to a fine-tuned transformer - built to tell genuine "
        "customer reviews apart from AI-generated ones."
    )

    cols = st.columns(4)

    stats = [
        ("40,405", "reviews analyzed"),
        ("10", "product categories"),
        ("4", "modeling approaches tested"),
        ("98.3%", "best model accuracy")
    ]

    for col, (val, lbl) in zip(
        cols,
        stats
    ):

        col.markdown(
            f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("")

    st.info(
        "Use the sidebar to navigate between the EDA report, "
        "model comparison, results, and the live classifier demo."
    )


# ============================================================
# PAGE: THE DATASET
# ============================================================

elif page == "The Dataset":

    st.markdown(
        '<div class="case-tag">'
        '<span class="dot"></span>'
        '01 - THE DATASET'
        '</div>',
        unsafe_allow_html=True
    )

    st.header(
        "40,405 reviews, near-perfectly balanced"
    )

    st.caption(
        "10 product categories, real Amazon reviews "
        "vs. GPT-2-generated fakes."
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Real", "Fake"],
                    values=[20215, 20190],
                    hole=0.65,
                    marker=dict(
                        colors=[
                            VERIFIED,
                            FLAGGED
                        ]
                    ),
                    textfont=dict(
                        color=BONE,
                        family="IBM Plex Mono"
                    ),
                )
            ]
        )

        fig.update_layout(
            title="Class Balance",
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=FOG),
            height=300,
            margin=dict(
                t=40,
                b=10,
                l=10,
                r=10
            ),
            showlegend=True,
            legend=dict(
                font=dict(color=FOG)
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = go.Figure(
            data=[
                go.Bar(
                    name="Real",
                    x=["avg. words per review"],
                    y=[73.6],
                    marker_color=VERIFIED
                ),

                go.Bar(
                    name="Fake",
                    x=["avg. words per review"],
                    y=[61.3],
                    marker_color=FLAGGED
                )
            ]
        )

        fig.update_layout(
            title="Review Length by Label",
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=FOG),
            height=300,
            margin=dict(
                t=40,
                b=10,
                l=10,
                r=10
            ),
            barmode="group",
            legend=dict(
                font=dict(color=FOG)
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    col3, col4 = st.columns(2)

    with col3:

        categories = [
            "Kindle Store",
            "Books",
            "Pet Supplies",
            "Home & Kitchen",
            "Electronics",
            "Sports & Outdoors",
            "Tools & Home Impr.",
            "Clothing/Shoes/Jewelry",
            "Toys & Games",
            "Movies & TV"
        ]

        counts = [
            4700,
            4400,
            4250,
            4050,
            4000,
            3950,
            3900,
            3850,
            3800,
            3600
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=counts,
                    y=categories,
                    orientation="h",
                    marker_color="#7BB6C9"
                )
            ]
        )

        fig.update_layout(
            title="Reviews per Category (approx.)",
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=FOG),
            height=380,
            margin=dict(
                t=40,
                b=10,
                l=10,
                r=10
            ),
            yaxis=dict(
                autorange="reversed"
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col4:

        fig = go.Figure(
            data=[
                go.Bar(
                    name="Fake",
                    x=["1", "2", "3", "4", "5"],
                    y=[
                        1050,
                        950,
                        1950,
                        3950,
                        12250
                    ],
                    marker_color=FLAGGED
                ),

                go.Bar(
                    name="Real",
                    x=["1", "2", "3", "4", "5"],
                    y=[
                        1050,
                        950,
                        1850,
                        4050,
                        12250
                    ],
                    marker_color=VERIFIED
                )
            ]
        )

        fig.update_layout(
            title="Rating Distribution by Label",
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=FOG),
            height=380,
            margin=dict(
                t=40,
                b=10,
                l=10,
                r=10
            ),
            barmode="group",
            legend=dict(
                font=dict(color=FOG)
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.caption(
        "Rating is near-identical across both classes - "
        "not a useful standalone fake-detection signal."
    )


# ============================================================
# PAGE: THE INVESTIGATION
# ============================================================

elif page == "The Investigation":

    st.markdown(
        '<div class="case-tag">'
        '<span class="dot"></span>'
        '02 - THE INVESTIGATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.header(
        "Four leads pursued"
    )

    st.caption(
        "From simple hand-built features to a fine-tuned "
        "transformer, each tested fairly on the same held-out data."
    )

    stages = [
        "Engineered Features\n(Random Forest)",
        "TF-IDF\n(Linear SVM)",
        "Fusion Model\n(TF-IDF + Engineered)",
        "DistilBERT\n(fine-tuned)"
    ]

    scores = [
        82.75,
        88.39,
        90.36,
        98.30
    ]

    colors = [
        FOG,
        "#7BB6C9",
        SIGNAL,
        VERIFIED
    ]

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=stages,
                orientation="h",
                marker_color=colors,
                text=[
                    f"{s}%"
                    for s in scores
                ],
                textposition="outside",
                textfont=dict(
                    family="IBM Plex Mono",
                    color=BONE
                ),
            )
        ]
    )

    fig.update_layout(
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(color=FOG),
        height=380,
        margin=dict(
            t=20,
            b=20,
            l=10,
            r=60
        ),
        xaxis=dict(
            range=[0, 110],
            title="Accuracy (%)"
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# PAGE: THE VERDICT
# ============================================================

elif page == "The Verdict":

    st.markdown(
        '<div class="case-tag">'
        '<span class="dot"></span>'
        '03 - THE VERDICT'
        '</div>',
        unsafe_allow_html=True
    )

    st.header(
        "DistilBERT, evaluated on 6,061 held-out reviews"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("##### Confusion Matrix")

        cm = [
            [2963, 70],
            [33, 2995]
        ]

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=cm,
                    x=[
                        "Predicted Real",
                        "Predicted Fake"
                    ],
                    y=[
                        "Actual Real",
                        "Actual Fake"
                    ],
                    colorscale=[
                        [0, PAPER_RAISED],
                        [1, VERIFIED]
                    ],
                    text=cm,
                    texttemplate="%{text}",
                    textfont=dict(
                        size=20,
                        family="IBM Plex Mono",
                        color=BONE
                    ),
                    showscale=False,
                )
            ]
        )

        fig.update_layout(
            paper_bgcolor=PAPER,
            plot_bgcolor=PAPER,
            font=dict(color=FOG),
            height=320,
            margin=dict(
                t=20,
                b=20,
                l=10,
                r=10
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.markdown("##### Test Set Metrics")

        m1, m2 = st.columns(2)

        m1.markdown(
            f"""
            <div class="metric-card">
                <div class="val"
                     style="color:{VERIFIED}">
                    98.30%
                </div>
                <div class="lbl">
                    Accuracy
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        m2.markdown(
            f"""
            <div class="metric-card">
                <div class="val"
                     style="color:{VERIFIED}">
                    98.31%
                </div>
                <div class="lbl">
                    F1 (fake class)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        m3, m4 = st.columns(2)

        m3.markdown(
            f"""
            <div class="metric-card">
                <div class="val"
                     style="color:{VERIFIED}">
                    97.72%
                </div>
                <div class="lbl">
                    Precision (fake)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        m4.markdown(
            f"""
            <div class="metric-card">
                <div class="val"
                     style="color:{VERIFIED}">
                    98.91%
                </div>
                <div class="lbl">
                    Recall (fake)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        st.caption(
            "A ~8-point accuracy gain over the best classical "
            "model (fusion, 90.36%) - but likely inflated by "
            "GPT-2's comparatively easy-to-detect generation artifacts. "
            "See Case Notes."
        )


# ============================================================
# PAGE: CASE NOTES
# ============================================================

elif page == "Case Notes":

    st.markdown(
        '<div class="case-tag">'
        '<span class="dot"></span>'
        '04 - CASE NOTES'
        '</div>',
        unsafe_allow_html=True
    )

    st.header(
        "103 of 6,061 test reviews were misclassified"
    )

    st.caption(
        "Reading the actual mistakes reveals what the model learned - "
        "and what fools it."
    )

    tab1, tab2 = st.tabs(
        [
            "False Positives (70)",
            "False Negatives (33)"
        ]
    )

    false_positives = [

        (
            "I enjoyed the two stories contained in this book. "
            "These are the only books by these authors that I have read.",

            "Short, blunt, factual - no personal elaboration. "
            "The model appears to associate brevity and genericness "
            "with AI generation, penalizing naturally terse human writers."
        ),

        (
            "They fit perfect, they look expensive, they are the "
            "most comfortable shoes that i had ever. I love the design.",

            "Enthusiastic but generic praise, close to the kind of "
            "language GPT-2 tends to generate."
        ),

        (
            "I have two of these, one for the kitchen and one "
            "for the dining room.",

            "Minimal, matter-of-fact - the shortest kind of real review, "
            "easily confused with generic filler text."
        ),

        (
            "Not at all what I was expecting. This was a large box "
            "of plastic pieces that don't fit together. I'm sure the "
            "only way this toy would be usable is if you glued the pieces together.",

            "A genuine complaint, but structured plainly enough to read "
            "as generic negative filler."
        ),

        (
            "Very comfortable, just a little short for my taste. "
            "I also have long legs, so that might be the issue.",

            "Specific personal detail (long legs) still wasn't enough "
            "to overcome the short-review heuristic."
        )
    ]

    false_negatives = [

        (
            "We used these with 3 of my 3 year old grandkids. "
            "They really enjoy playing with them.",

            "Specific, human-sounding family detail - exactly the kind "
            "of anecdotal texture real reviews have."
        ),

        (
            "Pros: Wash well. Laundry bag included, thicker than expected, "
            "great quality materials. Cons: Too small for my petite frame, "
            "I had to return it.",

            "Mimics a genuine human review convention - structured pros/cons "
            "lists are common in real reviews, so the model reads structure as authenticity."
        ),

        (
            "Received product at discount for honest review. "
            "Waking Up by Kirsten Clare. I am a huge fan of both "
            "Haus of Tars and The Blind Side.",

            "Borrows a real, common Amazon disclosure phrase - "
            "a genuine trust signal the fake text co-opts convincingly."
        ),

        (
            "Contains raw embedded HTML markup (a product link tag) "
            "rather than natural review prose.",

            "Likely not a genuine model failure - this looks like a data "
            "leakage or scraping artifact mislabeled as 'fake' in the source dataset."
        ),

        (
            "SHEET COLOR IS NICE BUY FOR MY SIZE FOR MY TOWN.",

            "Ungrammatical, broken English - the model likely treats "
            "imperfect fluency as a human tell, which a fake review can exploit."
        )
    ]

    with tab1:

        for text, note in false_positives:

            safe_text = html.escape(text)
            safe_note = html.escape(note)

            st.markdown(
                f"""
                <div class="exhibit">

                    <span class="stamp-fp">
                        REAL FLAGGED AS FAKE
                    </span>

                    <div class="exhibit-text">
                        "{safe_text}"
                    </div>

                    <div class="exhibit-note">
                        <b>Why it fooled the model:</b>
                        {safe_note}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    with tab2:

        for text, note in false_negatives:

            safe_text = html.escape(text)
            safe_note = html.escape(note)

            st.markdown(
                f"""
                <div class="exhibit">

                    <span class="stamp-fn">
                        FAKE MISSED AS REAL
                    </span>

                    <div class="exhibit-text">
                        "{safe_text}"
                    </div>

                    <div class="exhibit-note">
                        <b>Why it slipped through:</b>
                        {safe_note}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# PAGE: LIVE INTERROGATION
# ============================================================

elif page == "🔎 Live Interrogation":

    st.markdown(
        '<div class="case-tag">'
        '<span class="dot"></span>'
        'LIVE MODEL - DISTILBERT'
        '</div>',
        unsafe_allow_html=True
    )

    st.header(
        "Try it yourself"
    )

    st.caption(
        "Paste a product review below. The fine-tuned model "
        "will classify it live."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    with st.spinner("Loading model..."):

        model, tokenizer, device = load_model(
            model_dir
        )

    if model is None:

        st.error(
            f"Could not find a model at: `{model_dir}`\n\n"
            "Check the path in the sidebar - it should point "
            "to the folder containing config.json, "
            "model.safetensors, tokenizer_config.json, "
            "and tokenizer.json."
        )

    else:

        st.success(
            f"Model loaded successfully on **{device.upper()}**."
        )

        # ----------------------------------------------------
        # Review input
        # ----------------------------------------------------

        text = st.text_area(
            "Review text",
            height=150,
            placeholder=(
                "e.g. I bought this for my kitchen and it works great, "
                "the build quality is solid and it has lasted me over "
                "a year without any issues at all."
            ),
        )

        word_count = (
            len(text.split())
            if text.strip()
            else 0
        )

        st.caption(
            f"{word_count} words (minimum 3 required)"
        )

        # ----------------------------------------------------
        # Analyze button
        # ----------------------------------------------------

        if st.button(
            "Analyze Review",
            disabled=(word_count < 3)
        ):

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            with st.spinner(
                "Running model prediction..."
            ):

                result = predict(
                    text,
                    model,
                    tokenizer,
                    device
                )

            # ------------------------------------------------
            # VERDICT
            # ------------------------------------------------

            if result["label"] == "real":

                st.markdown(
                    f"""
                    <div class="verdict-real">

                        <div class="verdict-title"
                             style="color:{VERIFIED}">
                            ✓ LIKELY GENUINE
                        </div>

                        <div class="verdict-sub">
                            Model confidence:
                            {result['confidence'] * 100:.1f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="verdict-fake">

                        <div class="verdict-title"
                             style="color:{FLAGGED}">
                            ⚑ LIKELY AI-GENERATED
                        </div>

                        <div class="verdict-sub">
                            Model confidence:
                            {result['confidence'] * 100:.1f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # ------------------------------------------------
            # PROBABILITIES
            # ------------------------------------------------

            st.markdown("")

            colr, colf = st.columns(2)

            colr.metric(
                "Real probability",
                f"{result['prob_real'] * 100:.1f}%"
            )

            colf.metric(
                "Fake probability",
                f"{result['prob_fake'] * 100:.1f}%"
            )

            # =================================================
            # SHAP EXPLANATION
            # =================================================

            st.markdown("---")

            st.markdown(
                "## 🔬 Why did the model make this prediction?"
            )

            st.caption(
                "SHAP highlights the words and tokens that influenced "
                "the DistilBERT prediction for this individual review."
            )

            # ------------------------------------------------
            # Generate SHAP explanation
            # ------------------------------------------------

            try:

                with st.spinner(
                    "Generating SHAP explanation... "
                    "This can take a little while."
                ):

                    explainer = create_shap_explainer(
                        model_dir
                    )

                    if explainer is None:

                        st.error(
                            "Could not create the SHAP explainer."
                        )

                    else:

                        predicted_class = (
                            1
                            if result["label"] == "fake"
                            else 0
                        )

                        shap_df = get_shap_token_importance(
                            explainer,
                            text,
                            predicted_class
                        )

                        # ------------------------------------------------
                        # Visual explanation
                        # ------------------------------------------------

                        show_shap_visualization(
                            shap_df,
                            result["label"]
                        )

            except Exception as e:

                st.error(
                    "SHAP explanation could not be generated."
                )

                st.exception(e)

                st.info(
                    "The model prediction itself is still valid. "
                    "The error is only related to the explanation layer."
                )

            # ------------------------------------------------
            # Interpretation note
            # ------------------------------------------------

            st.markdown("---")

            st.info(
                "Important: SHAP explains the model's reasoning "
                "patterns. It does not prove that a review is "
                "actually genuine or AI-generated. A high model "
                "confidence can still correspond to an incorrect prediction."
            )