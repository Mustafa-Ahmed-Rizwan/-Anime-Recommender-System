# nlu/embedder.py
# Dense query retrieval: embed query via HuggingFace API,
# project into content space, search FAISS directly.
# synonym_map.py kept only as fallback.

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ── Genre label embeddings cache ─────────────────────────────────────────────
_genre_embeddings_cache = None


def _get_genre_embeddings(genre_classes: list) -> np.ndarray | None:
    """
    Get sentence embeddings for all genre labels via HF API.
    Cached in memory after first call so only runs once per session.
    """
    global _genre_embeddings_cache
    if _genre_embeddings_cache is not None:
        return _genre_embeddings_cache

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

        embeddings = []
        for genre in genre_classes:
            result = client.feature_extraction(
                f"anime genre: {genre}",
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            emb = np.array(result, dtype=np.float32)
            if emb.ndim == 2:
                emb = emb.mean(axis=0)
            norm = np.linalg.norm(emb)
            embeddings.append(emb / (norm + 1e-8))

        _genre_embeddings_cache = np.array(embeddings, dtype=np.float32)
        print(f"Genre embeddings cached: {_genre_embeddings_cache.shape}")
        return _genre_embeddings_cache

    except Exception as e:
        print(f"Genre embedding API error: {e}")
        return None


def get_query_embedding(query: str) -> np.ndarray | None:
    """
    Embed a user query string via HF API.
    Returns L2-normalized embedding or None on failure.
    """
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)

        result = client.feature_extraction(
            query,
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        emb  = np.array(result, dtype=np.float32)
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-8)

    except Exception as e:
        print(f"Query embedding API error: {e}")
        return None


def parse_query(
    query: str,
    faiss_artifacts: dict,
    content_matrix_norm: np.ndarray,
    anime_meta: pd.DataFrame = None
) -> tuple[np.ndarray, dict, list[int], str]:
    """
    Main entry point for NLU.
    Returns (query_vector_in_content_space, genre_weights_dict, seed_indices, method_used).

    Path 1 - Semantic (HF API available):
      Embed query → compare with genre label embeddings →
      softmax genre weights → build content-space query vector

    Path 2 - Keyword fallback (HF API down):
      Keyword match → genre weights → content-space query vector

    Path 3 - Last resort:
      Return mean content vector (popularity-based)
    """
    genre_classes = faiss_artifacts['genre_classes']
    n_genres      = len(genre_classes)
    content_dim   = content_matrix_norm.shape[1]
    q_lower       = query.lower()

    # ── TITLE DETECTION (Entity Recognition) ──────────────────────────────────
    seed_indices = []
    if anime_meta is not None:
        # Check for exact or near-exact matches of titles in query
        # We look for titles with length > 3 to avoid false positives
        for idx, row in anime_meta.iterrows():
            title = str(row['display_name']).lower()
            if len(title) > 3 and title in q_lower:
                original_id = row['anime_id']
                internal_idx = faiss_artifacts['anime2idx'].get(original_id)
                if internal_idx is not None:
                    seed_indices.append(internal_idx)
                    if len(seed_indices) >= 5: # Limit seeds
                        break

    # ── PATH 1: Semantic via HF API ───────────────────────────────────────────
    query_emb     = get_query_embedding(query)
    genre_embeds  = _get_genre_embeddings(genre_classes) if query_emb is not None else None

    if query_emb is not None and genre_embeds is not None:
        # Cosine similarity: query vs each genre label
        sims         = np.dot(genre_embeds, query_emb)       # shape: (n_genres,)

        # ── KEYWORD BOOSTING ──────────────────────────────────────────────────
        q_lower = query.lower()
        for i, g in enumerate(genre_classes):
            if g.lower() in q_lower:
                sims[i] += 0.5  # Large boost for exact keyword match

        # ── SEED-BASED GENRE BOOSTING ─────────────────────────────────────────
        if seed_indices and anime_meta is not None:
            for s_idx in seed_indices:
                orig_id = list(faiss_artifacts['anime2idx'].keys())[list(faiss_artifacts['anime2idx'].values()).index(s_idx)]
                anime_row = anime_meta[anime_meta['anime_id'] == orig_id]
                if not anime_row.empty:
                    genres_str = anime_row.iloc[0]['Genres']
                    if isinstance(genres_str, str):
                        for g in genres_str.split(','):
                            g_clean = g.strip().lower()
                            for i, gc in enumerate(genre_classes):
                                if gc.lower() == g_clean:
                                    sims[i] += 0.5 # Add to similarity score

        # Sharpen with temperature then softmax
        sims_shifted = sims - sims.max()
        weights      = np.exp(sims_shifted * 15)  # Significant increase (15)
        weights      = weights / (weights.sum() + 1e-8)

        # Build query vector: weighted sum using genre columns of content matrix
        # Genre features occupy first n_genres columns
        genre_cols   = content_matrix_norm[:, :n_genres]     # (n_anime, n_genres)
        anime_scores = genre_cols @ weights                   # (n_anime,)

        # Take top-30 scoring anime, average their content vectors as query
        top_idx      = np.argsort(anime_scores)[::-1][:30]
        query_vector = content_matrix_norm[top_idx].mean(axis=0)
        norm         = np.linalg.norm(query_vector)
        query_vector = query_vector / (norm + 1e-8)

        genre_weights = {
            genre_classes[i]: float(weights[i])
            for i in range(n_genres)
            if weights[i] > 0.05  # Higher threshold for cleaner UI
        }
        return query_vector, genre_weights, seed_indices, "semantic"

    # ── PATH 2: Keyword fallback ──────────────────────────────────────────────
    print("HF API unavailable, using keyword fallback")
    try:
        from nlu.synonym_map import SYNONYM_MAP
        q          = query.lower()
        genre_hits = {}
        for keyword, genres in SYNONYM_MAP.items():
            if keyword in q:
                for g in genres:
                    genre_hits[g] = genre_hits.get(g, 0) + 1

        if genre_hits:
            genre_classes_lower = [g.lower() for g in genre_classes]
            weights             = np.zeros(n_genres, dtype=np.float32)
            for genre, count in genre_hits.items():
                if genre.lower() in genre_classes_lower:
                    idx           = genre_classes_lower.index(genre.lower())
                    weights[idx]  = float(count)

            if weights.sum() > 0:
                weights      = weights / weights.sum()
                genre_cols   = content_matrix_norm[:, :n_genres]
                anime_scores = genre_cols @ weights
                top_idx      = np.argsort(anime_scores)[::-1][:30]
                query_vector = content_matrix_norm[top_idx].mean(axis=0)
                norm         = np.linalg.norm(query_vector)
                genre_weights = {
                    genre_classes[i]: float(weights[i])
                    for i in range(n_genres)
                    if weights[i] > 0
                }
                return query_vector / (norm + 1e-8), genre_weights, seed_indices, "keyword"
    except Exception as e:
        print(f"Keyword fallback error: {e}")

    # ── PATH 3: Last resort ───────────────────────────────────────────────────
    mean_vec = content_matrix_norm.mean(axis=0)
    norm     = np.linalg.norm(mean_vec)
    return mean_vec / (norm + 1e-8), {}, seed_indices, "fallback"