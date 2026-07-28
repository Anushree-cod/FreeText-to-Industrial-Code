import os
import re
from datetime import datetime, timedelta
import sqlite3

import pandas as pd
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from langdetect import detect

# SerpAPI for news search
try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    try:
        # Alternative import path
        from google_search_results import GoogleSearch
        SERPAPI_AVAILABLE = True
    except ImportError:
        SERPAPI_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
local_model = os.path.join(BASE_DIR, "fine_tuned_model_v3")

if os.path.exists(local_model):
    MODEL_PATH = local_model
else:
    MODEL_PATH = "hermoine1234/fine_tuned_model_v3"

DB_PATH = os.path.join(BASE_DIR, "app.db")

# SerpAPI Key - You can set it here directly or use environment variable
# Option 1: Set it directly here (for development/testing)
SERPAPI_KEY_DIRECT = None
# Option 2: Use environment variable (recommended for production)
# Set with: $env:SERPAPI_KEY="your_key" (PowerShell) or export SERPAPI_KEY="your_key" (Linux/Mac)


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS query_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            text TEXT NOT NULL,
            detected_language TEXT,
            top_code TEXT,
            confidence INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            text TEXT NOT NULL,
            rating TEXT NOT NULL,
            model_top_code TEXT,
            user_code TEXT,
            comment TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def load_model(path):
    return SentenceTransformer(path)


print("Loading SentenceTransformer model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully.")


def load_codes(path="combined_codes.csv"):
    df = pd.read_csv(path)
    for c in ["Code", "Title", "Description"]:
        if c not in df.columns:
            df[c] = ""
    df["clean_desc"] = df["Description"].fillna("").astype(str).str.replace("T", "").str.strip()
    df["text"] = df["Title"].fillna("").astype(str) + " " + df["clean_desc"]
    df["Code"] = df["Code"].astype(str)
    return df.reset_index(drop=True)


codes_df = load_codes(os.path.join(BASE_DIR, "combined_codes.csv"))


def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

print("Computing corpus embeddings...")
corpus_embeddings = compute_embeddings(codes_df["text"].tolist())
print("Corpus embeddings computed successfully.")

def extract_keywords(text, top_n=10):
    if not isinstance(text, str):
        return []
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9\-]{3,}", text)
    tokens = [t for t in tokens if not t.isdigit()]
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    sorted_tokens = sorted(freq.keys(), key=lambda x: (-freq[x], -len(x)))
    return sorted_tokens[:top_n]


def build_rationale(translated_query, top_idx, top_score):
    q_kw = extract_keywords(translated_query, top_n=20)
    title = codes_df.iloc[top_idx]["Title"]
    desc = codes_df.iloc[top_idx]["clean_desc"]
    desc_kw = extract_keywords(desc, top_n=40)
    overlap = [w for w in q_kw if w in desc_kw]

    sims = util.cos_sim(corpus_embeddings[top_idx], corpus_embeddings).cpu().numpy()[0]
    neighbor_idxs = sims.argsort()[::-1][1:4]
    neighbors = []
    for j in neighbor_idxs:
        neighbors.append(
            {
                "code": codes_df.iloc[int(j)]["Code"],
                "title": codes_df.iloc[int(j)]["Title"],
                "sim": float(sims[j]),
            }
        )

    rationale = (
        f"The model matched your description to {title} (Code {codes_df.iloc[top_idx]['Code']}) "
        f"with similarity {top_score:.3f}."
    )
    if overlap:
        rationale += f" Overlapping tokens: {', '.join(overlap[:8])}."
    else:
        rationale += " No exact token overlap found — the model relies on semantic similarity."

    return rationale, overlap, neighbors


def confidence_to_percentage(sim_score):
    s = max(-1.0, min(1.0, sim_score))
    pct = int(((s + 1) / 2) * 100)
    if s < 0.25:
        pct = int(pct * 0.6)
    return max(0, min(100, pct))


def _code_source(idx: int) -> str:
    return str(codes_df.iloc[idx].get("Source", "")).strip().upper()


def select_balanced_top_codes(
    indices,
    scores,
    total: int = 5,
    min_nic: int = 2,
    min_naics: int = 2,
):
    """
    Pick up to `total` codes from a ranked list, requiring at least
    `min_nic` NIC and `min_naics` NAICS when enough of each exist.
    Final list is sorted by similarity (highest first).
    """
    pairs = [(int(i), float(s)) for i, s in zip(indices, scores)]

    nic_pool = [(i, s) for i, s in pairs if _code_source(i) == "NIC"]
    naics_pool = [(i, s) for i, s in pairs if _code_source(i) == "NAICS"]

    selected = []
    selected_ids: set[int] = set()

    def take_from(pool, count):
        added = 0
        for i, s in pool:
            if added >= count:
                break
            if i in selected_ids:
                continue
            selected.append((i, s))
            selected_ids.add(i)
            added += 1

    # Guarantee quotas first (as many as available, up to the minima)
    take_from(nic_pool, min_nic)
    take_from(naics_pool, min_naics)

    # Fill remaining slots from the overall ranking
    for i, s in pairs:
        if len(selected) >= total:
            break
        if i in selected_ids:
            continue
        selected.append((i, s))
        selected_ids.add(i)

    selected.sort(key=lambda item: item[1], reverse=True)
    return selected[:total]


def filter_indices_by_system(indices, scores, system: str):
    """
    Filter ranked indices/scores by requested code system.
    system: 'nic', 'naics', or 'both'
    Returns filtered (indices, scores). If filter would be empty, returns originals.
    """
    system = (system or "both").lower()
    if system not in {"nic", "naics"}:
        return [int(i) for i in indices], [float(s) for s in scores]

    target_source = "NIC" if system == "nic" else "NAICS"
    filtered_indices = []
    filtered_scores = []
    for i, s in zip(indices, scores):
        i_int = int(i)
        source = str(codes_df.iloc[i_int].get("Source", ""))
        if source.upper() == target_source:
            filtered_indices.append(i_int)
            filtered_scores.append(float(s))

    # If nothing matches, fall back to original lists
    if not filtered_indices:
        return [int(i) for i in indices], [float(s) for s in scores]

    return filtered_indices, filtered_scores


class ClassifyRequest(BaseModel):
    text: str
    language: str | None = None
    code_system: str | None = "both"


class ClassifyResponse(BaseModel):
    top_code: str
    top_title: str
    top_description: str
    confidence: int
    similarity: float
    detected_language: str
    translated_query: str
    top_suggestions: list
    rationale: str
    neighbors: list
    timestamp: str
    sector_news: list | None = None


class FeedbackRequest(BaseModel):
    text: str
    rating: str
    model_top_code: str | None = None
    user_code: str | None = None
    comment: str | None = None


api = FastAPI(title="Industrial Code Classifier API")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
api.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static",
)
api.mount(
    "/image",
    StaticFiles(directory=os.path.join(BASE_DIR, "image")),
    name="image",
)


@api.on_event("startup")
def on_startup():
    init_db()


def fetch_sector_news(sector_title: str, sector_code: str, max_results: int = 5):
    """
    Fetch recent news articles about the industrial sector using SerpAPI.
    Returns a list of news items with title, snippet, link, source, and date.
    """
    if not SERPAPI_AVAILABLE:
        print("SerpAPI: Package not installed. Install with: pip install google-search-results")
        return None

    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        print("SerpAPI: API key not found. Set SERPAPI_KEY environment variable")
        return None

    try:
        # Build search query - focus on recent news (last 24 hours)
        search_query = f"{sector_title} industrial sector news"
        print(f"SerpAPI: Searching for: {search_query}")

        # Use Google News search via SerpAPI
        params = {
            "engine": "google",
            "q": search_query,
            "tbm": "nws",  # News search
            "api_key": serpapi_key,
            "num": max_results,
            "tbs": "qdr:d",  # Past 24 hours
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        print(f"SerpAPI: Response keys: {list(results.keys())}")
        if "error" in results:
            print(f"SerpAPI Error: {results.get('error')}")
            return None

        news_items = []
        if "news_results" in results and results["news_results"]:
            for item in results["news_results"][:max_results]:
                news_items.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "link": item.get("link", ""),
                    "source": item.get("source", ""),
                    "date": item.get("date", ""),
                })
            print(f"SerpAPI: Found {len(news_items)} news items")
        else:
            print("SerpAPI: No news_results in response")

        return news_items if news_items else None

    except Exception as e:
        # Fail gracefully - return None if SerpAPI fails
        print(f"SerpAPI error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


@api.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@api.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    query = req.text
    if not query or not query.strip():
        raise ValueError("Empty query.")

    if req.language and req.language != "auto":
        lang = req.language
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

    # --- Similarity computation ---
    q_emb = model.encode(translated, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, corpus_embeddings)[0]

    system = (req.code_system or "both").lower()

    # --- Candidate selection depending on requested code system ---
    if system in {"nic", "naics"}:
        # Restrict search space strictly to the chosen code system
        target_source = "NIC" if system == "nic" else "NAICS"
        source_series = codes_df.get("Source", "").astype(str).str.upper()
        candidate_indices = [i for i, src in enumerate(source_series) if src == target_source]

        if not candidate_indices:
            # If there are no codes of this type, fall back to global search
            topk = torch.topk(sims, k=min(40, len(sims)))
            indices = topk.indices.tolist()
            scores = [float(s) for s in topk.values.tolist()]
        else:
            # Take top-k within the filtered set
            candidate_sims = sims[candidate_indices]
            k = min(40, len(candidate_indices))
            topk = torch.topk(candidate_sims, k=k)
            local_indices = topk.indices.tolist()
            scores = [float(s) for s in topk.values.tolist()]
            indices = [int(candidate_indices[i]) for i in local_indices]
    else:
        # 'both' or anything else → global candidate set
        topk = torch.topk(sims, k=min(40, len(sims)))
        indices = topk.indices.tolist()
        scores = [float(s) for s in topk.values.tolist()]

    # build top-5 list depending on requested system
    if system == "both":
        # Always include at least 2 NIC and 2 NAICS when available
        selected_pairs = select_balanced_top_codes(
            indices, scores, total=5, min_nic=2, min_naics=2
        )
    else:
        # NIC-only or NAICS-only: just take top 5 from filtered list
        selected_pairs = [(int(i), float(s)) for i, s in zip(indices[:5], scores[:5])]

    if not selected_pairs:
        selected_pairs = [(int(indices[0]), float(scores[0]))]

    # Primary match = highest-scoring code in the final balanced set
    top_idx, top_score = selected_pairs[0]
    top_code = codes_df.iloc[top_idx]["Code"]
    top_title = codes_df.iloc[top_idx]["Title"]
    top_desc = codes_df.iloc[top_idx]["clean_desc"]
    pct = confidence_to_percentage(top_score)

    rationale, overlap, neighbors = build_rationale(translated, top_idx, top_score)

    suggestions = []
    for i, s in selected_pairs:
        source = str(codes_df.iloc[i].get("Source", ""))
        desc = str(codes_df.iloc[i].get("Description", ""))
        short_desc = desc.strip()
        if len(short_desc) > 110:
            short_desc = short_desc[:107].rstrip() + "..."
        suggestions.append(
            {
                "code": codes_df.iloc[i]["Code"],
                "title": codes_df.iloc[i]["Title"],
                "similarity": float(s),
                "source": source,
                "short_description": short_desc,
            }
        )

    # Fetch sector news (non-blocking, fails gracefully)
    sector_news = None
    try:
        sector_news = fetch_sector_news(top_title, top_code, max_results=5)
        print(f"Classify: sector_news result: {sector_news}")
    except Exception as e:
        # Log error but don't fail the request
        print(f"Classify: Error fetching sector news: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sector_news = None

    # log query to database
    try:
        conn = get_db_connection()
        conn.execute(
            "INSERT INTO query_log (timestamp, text, detected_language, top_code, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                query,
                lang,
                top_code,
                pct,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        # fail silently for logging
        pass

    return ClassifyResponse(
        top_code=top_code,
        top_title=top_title,
        top_description=top_desc,
        confidence=pct,
        similarity=float(top_score),
        detected_language=lang,
        translated_query=translated,
        top_suggestions=suggestions,
        rationale=rationale,
        neighbors=neighbors,
        timestamp=datetime.utcnow().isoformat(),
        sector_news=sector_news,
    )


@api.get("/api/stats")
def get_database_stats():
    """
    Return database statistics: total codes, NIC codes count, NAICS codes count.
    """
    try:
        total_codes = len(codes_df)
        nic_codes = len(codes_df[codes_df.get("Source", "") == "NIC"])
        naics_codes = len(codes_df[codes_df.get("Source", "") == "NAICS"])
        
        return {
            "total_codes": total_codes,
            "nic_codes": nic_codes,
            "naics_codes": naics_codes,
        }
    except Exception as e:
        return {
            "total_codes": 0,
            "nic_codes": 0,
            "naics_codes": 0,
        }


@api.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """
    Store simple feedback in SQLite for later analysis.
    """
    try:
        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO feedback (timestamp, text, rating, model_top_code, user_code, comment)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                req.text,
                req.rating,
                req.model_top_code,
                req.user_code,
                req.comment,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        # If logging fails, still return success to keep UX smooth
        return {"status": "ok", "logged": False}

    return {"status": "ok", "logged": True}
