# precompute_embeddings.py
import pandas as pd
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
import os
from tqdm import tqdm

# --------- CONFIG ----------
INPUT_CSV = "combined_codes.csv"
OUT_CSV = "processed_codes.csv"
EMB_NPY = "corpus_embeddings.npy"
FAISS_IDX = "faiss_index.bin"
MODEL_NAME = "fine_tuned_model_v3"  # your finetuned model
BATCH_SIZE = 64
# --------------------------

def load_df(path):
    df = pd.read_csv(path)
    df['text'] = df['Title'].fillna('') + " " + df['Description'].fillna('')
    df = df.reset_index(drop=True)
    return df

def compute_embeddings(model, texts, batch_size=64, device="cpu"):
    all_embs = []
    model.to(device)
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch,
                           convert_to_numpy=True,
                           show_progress_bar=False,
                           device=device)
        all_embs.append(emb)
    all_embs = np.vstack(all_embs).astype('float32')
    return all_embs

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device for encoding:", device)

    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f"Input file {INPUT_CSV} not found.")

    df = load_df(INPUT_CSV)
    print(f"Loaded {len(df)} rows")

    model = SentenceTransformer(MODEL_NAME)

    embs = compute_embeddings(model, df['text'].tolist(), batch_size=BATCH_SIZE, device=device)

    # Normalize vectors for cosine-sim via inner product
    faiss.normalize_L2(embs)

    # Save numpy embeddings (optional but handy)
    np.save(EMB_NPY, embs)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved embeddings to {EMB_NPY} and processed CSV to {OUT_CSV}")

    # Build FAISS index (IndexFlatIP for inner product on normalized vectors)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)  # add all vectors
    faiss.write_index(index, FAISS_IDX)
    print(f"FAISS index written to {FAISS_IDX}. Total vectors: {index.ntotal}")

if __name__ == "__main__":
    main()
