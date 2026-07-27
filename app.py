

# app.py
import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import torch
from deep_translator import GoogleTranslator
from langdetect import detect

# -----------------------------
# Config / model path
# -----------------------------
MODEL_PATH = "fine_tuned_model_v3"
if not os.path.exists(MODEL_PATH):
    # fallback absolute example (adjust if needed)
    MODEL_PATH = r"C:\Users\A\Downloads\EDI Project\fine_tuned_model_v3"

# -----------------------------
# Load model (cached resource)
# -----------------------------
@st.cache_resource
def load_model(path):
    return SentenceTransformer(path)

model = load_model(MODEL_PATH)

# -----------------------------
# Load codes dataset (local combined_codes.csv)
# -----------------------------
@st.cache_data
def load_codes(path="combined_codes.csv"):
    df = pd.read_csv(path)
    # ensure columns exist and clean
    for c in ["Code", "Title", "Description"]:
        if c not in df.columns:
            df[c] = ""
    df["clean_desc"] = df["Description"].fillna("").astype(str).str.replace("T","").str.strip()
    df["text"] = df["Title"].fillna("").astype(str) + " " + df["clean_desc"]
    df["Code"] = df["Code"].astype(str)
    return df.reset_index(drop=True)

codes_df = load_codes()

# -----------------------------
# Compute embeddings (avoid hashing model)
# -----------------------------
@st.cache_resource
def compute_embeddings(_model_obj, texts):
    return _model_obj.encode(texts, convert_to_tensor=True)

corpus_embeddings = compute_embeddings(model, codes_df["text"].tolist())

# -----------------------------
# Utilities
# -----------------------------
def extract_keywords(text, top_n=10):
    if not isinstance(text, str):
        return []
    text = text.lower()
    # simple tokenization
    tokens = re.findall(r"[a-zA-Z0-9\-]{3,}", text)
    # remove purely numeric tokens
    tokens = [t for t in tokens if not t.isdigit()]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    # sort by freq then length
    sorted_tokens = sorted(freq.keys(), key=lambda x: (-freq[x], -len(x)))
    return sorted_tokens[:top_n]

def translate_text(text, target_lang):
    if not text or target_lang in ["en","unknown"]:
        return text
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text

def build_rationale(translated_query, top_idx, top_score):
    """
    Build an English textual rationale explaining why the top code is selected.
    We use overlapping tokens and top-3 neighbor codes as evidence.
    """
    q_kw = extract_keywords(translated_query, top_n=20)
    title = codes_df.iloc[top_idx]["Title"]
    desc = codes_df.iloc[top_idx]["clean_desc"]
    desc_kw = extract_keywords(desc, top_n=40)
    overlap = [w for w in q_kw if w in desc_kw]
    # top3 neighbor codes for context
    sims = util.cos_sim(corpus_embeddings[top_idx], corpus_embeddings).cpu().numpy()[0]
    neighbor_idxs = sims.argsort()[::-1][1:4]  # exclude self
    neighbors = []
    for j in neighbor_idxs:
        neighbors.append({
            "code": codes_df.iloc[int(j)]["Code"],
            "title": codes_df.iloc[int(j)]["Title"],
            "sim": float(sims[j])
        })

    rationale = f"The model matched your description to **{title}** (Code {codes_df.iloc[top_idx]['Code']}) with similarity {top_score:.3f}.\n"
    if overlap:
        rationale += f"Overlapping tokens found between your text and the code description: {', '.join(overlap[:8])}.\n"
    else:
        # show some top keywords from both
        rationale += f"No exact token overlap found — the model relies on semantic similarity. Top tokens from your query: {', '.join(q_kw[:8])}; top tokens in code description: {', '.join(desc_kw[:8])}.\n"

    rationale += "Related industry codes (context):\n"
    for n in neighbors:
        rationale += f"- {n['code']} — {n['title']} (sim={n['sim']:.3f})\n"

    return rationale, overlap, neighbors

def confidence_to_percentage(sim_score):
    # sim_score in [-1,1] - convert to [0,100] with sensible scaling
    # embeddings similarity rarely negative here, clamp
    s = max(-1.0, min(1.0, sim_score))
    # map 0.2 -> ~0, 0.5->50, 0.8->90 etc using piecewise transform
    pct = int(((s + 1) / 2) * 100)  # simple linear
    # calibrate a bit: very low sims should be pushed lower
    if s < 0.25:
        pct = int(pct * 0.6)
    return max(0, min(100, pct))

def refinement_suggestions(query, pct, translated_query):
    suggestions = []
    # too short
    token_count = len(re.findall(r"[a-zA-Z0-9]{2,}", query))
    if token_count < 6:
        suggestions.append("Your description is short — add details: products/services, target customers, main activities, tools/machines used, and scale (local/national/export).")
    # low confidence
    if pct < 40:
        suggestions.append("Model confidence is low — try adding concrete keywords like 'manufacturing', 'retail', 'software development', 'export', 'wholesale', 'dairy farm', 'logistics', etc.")
    # ambiguous words
    if "service" in translated_query.lower() or "business" in translated_query.lower():
        suggestions.append("Your text is generic (uses words like 'service'/'business'). Specify what the service does (e.g., 'IT consulting for hospitals').")
    # recommend examples
    if not suggestions:
        suggestions.append("Query looks good. If you want a more precise code, mention specific products, customers, or processes.")
    return suggestions

def classify_text(query: str, forced_lang: str | None = None):
    """
    Core classification routine reused by UI (and optional API).
    Returns a dict with all relevant fields or raises on error.
    """
    if not query or not query.strip():
        raise ValueError("Empty query.")

    # language detection and translation for embedding
    if forced_lang and forced_lang != "auto":
        lang = forced_lang
    else:
        try:
            lang = detect(query)
        except Exception:
            lang = "unknown"

    if lang not in ["en", "hi", "es", "fr", "de", "zh-cn", "ja"]:
        try:
            translated = GoogleTranslator(source="auto", target="en").translate(query)
        except Exception:
            translated = query
    else:
        translated = query

    # embedding & similarity
    q_emb = model.encode(translated, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, corpus_embeddings)[0]
    topk = torch.topk(sims, k=min(6, len(sims)))
    indices = topk.indices.tolist()
    scores = [float(s) for s in topk.values.tolist()]

    top_idx = int(indices[0])
    top_score = scores[0]
    top_code = codes_df.iloc[top_idx]["Code"]
    top_title = codes_df.iloc[top_idx]["Title"]
    top_desc = codes_df.iloc[top_idx]["clean_desc"]

    pct = confidence_to_percentage(top_score)
    rationale_en, overlap, neighbors = build_rationale(translated, top_idx, top_score)
    rationale_translated = translate_text(rationale_en, lang)
    recommendations = refinement_suggestions(query, pct, translated)

    return {
        "lang": lang,
        "translated_query": translated,
        "indices": indices,
        "scores": scores,
        "top_idx": top_idx,
        "top_score": top_score,
        "top_code": top_code,
        "top_title": top_title,
        "top_desc": top_desc,
        "confidence_pct": pct,
        "rationale_en": rationale_en,
        "rationale_translated": rationale_translated,
        "overlap": overlap,
        "neighbors": neighbors,
        "recommendations": recommendations,
    }


def init_session_state():
    if "page" not in st.session_state:
        st.session_state.page = "Landing"
    if "history" not in st.session_state:
        st.session_state.history = []
    if "feedback_log" not in st.session_state:
        st.session_state.feedback_log = []
    if "preset_example" not in st.session_state:
        st.session_state.preset_example = ""


def log_history(query, result):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "lang": result["lang"],
        "top_code": result["top_code"],
        "top_title": result["top_title"],
        "confidence_pct": result["confidence_pct"],
    }
    st.session_state.history.append(entry)


def log_feedback(query, result, feedback_label, correct_code_text):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "lang": result["lang"],
        "top_code": result["top_code"],
        "top_title": result["top_title"],
        "confidence_pct": result["confidence_pct"],
        "feedback": feedback_label,
        "correct_code": correct_code_text.strip() if correct_code_text else "",
    }
    st.session_state.feedback_log.append(entry)
    # Optionally also append to a local CSV for offline analysis
    try:
        fb_df = pd.DataFrame([entry])
        file_exists = os.path.exists("feedback_log.csv")
        fb_df.to_csv("feedback_log.csv", mode="a", index=False, header=not file_exists)
    except Exception:
        # Silent fail for environments where writing is not possible
        pass


def render_landing_page():
    st.title("🌍 Industrial Code Finder")
    st.subheader("Type your business description and get the best-matching industrial code.")
    st.write(
        "This demo uses a fine-tuned multilingual transformer model to map free-text "
        "business descriptions to standard industrial activity codes."
    )

    st.markdown("### How it works")
    st.write("- **Step 1**: Describe your business in your own words, in any language.")
    st.write("- **Step 2**: The model finds the closest industry codes using semantic similarity.")
    st.write("- **Step 3**: You see the **top match**, **confidence score**, and **top-3 suggestions** with explanations.")

    st.markdown("### Example descriptions you can try")
    examples = [
        "We manufacture lithium-ion battery packs for electric scooters and small electric vehicles.",
        "Small neighborhood grocery shop selling fresh vegetables, packaged foods, and everyday household items.",
        "Online platform that connects freelance software developers with international clients.",
        "Dairy farm producing raw milk and cheese for local supermarkets and restaurants.",
    ]
    for ex in examples:
        st.code(ex)

    if st.button("🚀 Try demo"):
        # Pre-fill a good example and jump to classifier page
        st.session_state.preset_example = examples[0]
        st.session_state.page = "Classifier"


def render_classifier_page():
    st.title("🔍 Industrial Code Classifier")

    col_main, col_side = st.columns([3, 1])

    with col_main:
        default_text = st.session_state.get("preset_example", "")
        language_options = ["auto", "en", "hi", "es", "fr", "de", "zh-cn", "ja"]
        lang_choice = st.selectbox(
            "Language (auto-detect or choose):",
            options=language_options,
            format_func=lambda x: "Auto-detect" if x == "auto" else x,
        )

        user_input = st.text_area(
            "Describe your business or activity:",
            height=160,
            placeholder="e.g. 'We manufacture lithium-ion battery packs for electric scooters'",
            value=default_text,
        )

        if st.button("🔍 Classify"):
            if not user_input.strip():
                st.warning("Please enter a business description.")
            else:
                try:
                    result = classify_text(user_input, forced_lang=lang_choice)
                except Exception as e:
                    st.error("The classifier is currently unavailable or took too long. Please try again.")
                    return

                log_history(user_input, result)

                lang = result["lang"]
                st.info(f"Detected language: {lang}")

                pct = result["confidence_pct"]
                top_code = result["top_code"]
                top_title = result["top_title"]
                top_score = result["top_score"]

                st.metric(label="Primary match", value=f"{top_code} — {top_title}")
                st.write(f"**Similarity score:** {top_score:.4f}")
                st.progress(pct / 100)
                st.markdown(f"**Confidence:** {pct}%")

                if pct < 35:
                    st.warning("We are not very sure about this prediction. Please refine your description or add more details.")

                st.subheader("Why this code? (semantic explanation)")
                st.markdown(result["rationale_en"])

                if lang != "en" and result["rationale_translated"] and result["rationale_translated"] != result["rationale_en"]:
                    with st.expander(f"Explanation in your language ({lang})"):
                        st.write(result["rationale_translated"])

                st.subheader("Top suggestions (ranked)")
                for rank, (i, s) in enumerate(zip(result["indices"][:5], result["scores"][:5]), start=1):
                    i = int(i)
                    code_i = codes_df.iloc[i]["Code"]
                    title_i = codes_df.iloc[i]["Title"]
                    st.write(f"{rank}. {code_i} — {title_i} (sim={s:.4f})")

                st.subheader("Query Refinement Assistant")
                for rec in result["recommendations"]:
                    st.write("- " + rec)

                st.markdown("---")
                st.markdown("#### Was this prediction useful?")
                fb_col1, fb_col2, fb_col3 = st.columns(3)
                correct_code_text = st.text_input("Correct code (if different):", value="")

                if fb_col1.button("✅ Correct"):
                    log_feedback(user_input, result, "correct", correct_code_text)
                    st.success("Thanks for your feedback!")
                if fb_col2.button("➗ Partly correct"):
                    log_feedback(user_input, result, "partly_correct", correct_code_text)
                    st.success("Thanks for your feedback!")
                if fb_col3.button("❌ Wrong"):
                    log_feedback(user_input, result, "wrong", correct_code_text)
                    st.success("Thanks for your feedback!")

    with col_side:
        st.markdown("### Quick tips")
        st.write("- Be specific: mention products, processes, tools, and customers.")
        st.write("- Add scale (e.g., 'retail store', 'export', 'domestic').")
        st.write("- You can write in many languages; we translate internally when needed.")
        st.write("---")
        st.markdown("### Recent queries (this session)")
        if st.session_state.history:
            # show last 5
            last_items = st.session_state.history[-5:][::-1]
            hist_df = pd.DataFrame(last_items)[["query", "top_code", "confidence_pct"]]
            hist_df.rename(
                columns={"query": "Description", "top_code": "Code", "confidence_pct": "Conf.%"},
                inplace=True,
            )
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.write("No queries yet in this session.")

        st.write("---")
        st.markdown("### Model info")
        st.write(f"Model: {MODEL_PATH}")
        st.write(f"Codes loaded: {len(codes_df)}")


def render_analytics_page():
    st.title("📊 Simple Analytics (Session)")
    if not st.session_state.history:
        st.info("No session data yet. Run a few classifications first.")
        return

    df = pd.DataFrame(st.session_state.history)

    st.subheader("Most common predicted codes")
    code_counts = df["top_code"].value_counts().head(10)
    st.bar_chart(code_counts)

    st.subheader("Distribution by detected language")
    lang_counts = df["lang"].value_counts()
    st.bar_chart(lang_counts)

    st.subheader("Raw history")
    st.dataframe(df, use_container_width=True)


def render_help_page():
    st.title("ℹ️ Help, Examples & FAQ")

    st.markdown("### How to write a good description")
    st.write("- **Be concrete**: mention your main products or services.")
    st.write("- **Mention your customers**: e.g., households, hospitals, factories, export clients.")
    st.write("- **Describe processes**: manufacturing, retailing, software development, logistics, farming, etc.")
    st.write("- **Add scale and channel**: online marketplace, local store, nationwide wholesaler, etc.")

    st.markdown("### Example inputs and indicative outputs")
    st.write("These are illustrative examples; actual codes will depend on your code list.")
    st.code("We operate an online marketplace where small retailers list clothes and shoes for consumers.")
    st.write("→ Likely retail / e-commerce related trade code.")

    st.code("Company assembling solar panels and installing rooftop solar systems for factories and homes.")
    st.write("→ Likely manufacturing + installation of electrical equipment / renewable energy code.")

    st.code("Clinic providing general medical consultations and basic diagnostic tests to outpatients.")
    st.write("→ Likely human health activities / outpatient care code.")

    st.markdown("### FAQ")
    st.write("**What is the confidence score?**")
    st.write(
        "It is a scaled similarity between your description and the closest code description in the embedding space. "
        "Higher values mean the model believes the match is stronger."
    )
    st.write("**What if the code is wrong?**")
    st.write(
        "Use the feedback buttons and optionally provide the correct code. "
        "This helps us understand typical errors and can be used for future fine-tuning."
    )
    st.write("**Do you store my data?**")
    st.write(
        "For this demo, descriptions may be temporarily kept in session memory and optional feedback logs. "
        "We do not intentionally collect personal data; please avoid entering names, addresses, or sensitive information."
    )


# -----------------------------
# Streamlit UI entrypoint
# -----------------------------
st.set_page_config(page_title="Industrial Code Classifier", layout="wide")
init_session_state()

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Landing", "Classifier", "Analytics", "Help / FAQ"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Privacy note**: This demo does not intentionally store personal data. "
                    "Descriptions are used only for research/demo purposes. "
                    "Avoid entering names, addresses, or sensitive information.")

if page == "Landing":
    render_landing_page()
elif page == "Classifier":
    render_classifier_page()
elif page == "Analytics":
    render_analytics_page()
else:
    render_help_page()

st.caption(
    "Features: Confidence meter, semantic explanations, multilingual rationale, "
    "query refinement assistant, feedback capture, history and simple analytics."
)
