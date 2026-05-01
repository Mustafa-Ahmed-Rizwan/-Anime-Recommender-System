# engine/recommender.py
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle
import sqlite3
import faiss


def load_artifacts(model_dir: str, processed_dir: str) -> dict:
    """Load all model artifacts once at app startup."""
    user_factors = np.load(f"{model_dir}/user_factors.npy")
    item_factors = np.load(f"{model_dir}/item_factors.npy")
    user_means = np.load(f"{model_dir}/user_means.npy")
    anime_similarity = np.load(f"{model_dir}/anime_similarity_filtered.npy")
    content_matrix_norm = np.load(f"{model_dir}/content_matrix_norm.npy")
    train_matrix = sp.load_npz(f"{processed_dir}/train_matrix.npz")
    anime_meta = pd.read_parquet(f"{processed_dir}/anime_meta.parquet")

    faiss_index = faiss.read_index(f"{model_dir}/faiss_index.bin")

    with open(f"{processed_dir}/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    with open(f"{processed_dir}/feature_meta.pkl", "rb") as f:
        feature_meta = pickle.load(f)

    with open(f"{model_dir}/model_config.pkl", "rb") as f:
        config = pickle.load(f)

    return {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_means": user_means,
        "anime_similarity": anime_similarity,
        "content_matrix_norm": content_matrix_norm,
        "train_matrix": train_matrix,
        "anime_meta": anime_meta,
        "faiss_index": faiss_index,
        "user2idx": mappings["user2idx"],
        "anime2idx": mappings["anime2idx"],
        "idx2anime": mappings["idx2anime"],
        "genre_classes": feature_meta["genre_classes"],
        "shift": config["shift"],
        "n_anime": item_factors.shape[0],
    }


# ── FAISS search helper ───────────────────────────────────────────────────────
def _faiss_search(
    query_vector: np.ndarray, artifacts: dict, top_k: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Search FAISS index, returns (scores, indices)."""
    index = artifacts["faiss_index"]
    q = query_vector.astype(np.float32).reshape(1, -1)
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    D, I = index.search(q, top_k)
    return D[0], I[0]


# ── Build explanation line ────────────────────────────────────────────────────
def _make_explanation(
    anime_genres: str,
    genre_classes: list,
    query_vector: np.ndarray,
    content_matrix_norm: np.ndarray,
    a_idx: int,
    method: str,
) -> str:
    if not isinstance(anime_genres, str):
        return "Recommended based on content similarity"

    genres_in_anime = [g.strip() for g in anime_genres.split(",")]

    if method == "semantic":
        # Find top genres contributing to match
        n_genres = len(genre_classes)
        genre_vals = content_matrix_norm[a_idx, :n_genres]
        top_genre_idxs = np.argsort(genre_vals)[::-1][:3]
        matched = [
            genre_classes[i]
            for i in top_genre_idxs
            if genre_classes[i] in genres_in_anime
        ]
        if matched:
            return f"Semantic match — strong in: {', '.join(matched)}"
        return "Semantic content match to your query"

    matched = [
        g
        for g in genres_in_anime
        if any(
            g.lower() in gc.lower() or gc.lower() in g.lower() for gc in genre_classes
        )
    ][:3]
    return f"Matches: {', '.join(matched)}" if matched else "Content similarity match"


# ── Content-based: new user, query vector from NLU ───────────────────────────
def recommend_by_query_vector(
    query_vector: np.ndarray,
    method: str,
    artifacts: dict,
    top_n: int = 10,
    seed_indices: list[int] = None,
) -> pd.DataFrame:
    """
    Content-based recommendation from NLU query vector.
    Uses FAISS for fast retrieval. For new (not logged-in) users.
    """
    anime_meta = artifacts["anime_meta"]
    content_matrix_norm = artifacts["content_matrix_norm"]
    n_anime = artifacts["n_anime"]
    idx2anime = artifacts["idx2anime"]
    genre_classes = artifacts["genre_classes"]

    # Blend seed items into query_vector if available
    if seed_indices:
        seed_vec = content_matrix_norm[seed_indices].mean(axis=0)
        query_vector = 0.8 * seed_vec + 0.2 * query_vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

    # FAISS search
    scores, indices = _faiss_search(query_vector, artifacts, top_k=min(150, n_anime))

    results = []
    for score, a_idx in zip(scores, indices):
        if a_idx < 0 or a_idx >= n_anime:
            continue
        original_id = idx2anime.get(int(a_idx))
        if original_id is None:
            continue
        row = anime_meta[anime_meta["anime_id"] == original_id]
        if row.empty:
            continue
        info = row.iloc[0]
        why = _make_explanation(
            info["Genres"],
            genre_classes,
            query_vector,
            content_matrix_norm,
            a_idx,
            method,
        )
        results.append(
            {
                "anime_id": original_id,
                "name": info["display_name"],
                "genres": info["Genres"],
                "type": info["Type"],
                "score": info["Score"],
                "match_score": round(float(score), 4),
                "why": why,
                "source": "Content (FAISS)",
            }
        )
        if len(results) >= top_n:
            break

    return pd.DataFrame(results)


# ── Hybrid: logged-in user ────────────────────────────────────────────────────
def recommend_hybrid(
    dataset_uid: int,
    query_vector: np.ndarray,
    method: str,
    artifacts: dict,
    db_path: str,
    top_n: int = 10,
    seed_indices: list[int] = None,
) -> tuple[pd.DataFrame, float, int]:
    """
    Full hybrid recommendation for a logged-in user.
    Combines:
      - ALS collaborative filtering scores
      - FAISS content-based scores from NLU query vector
      - History-based content scores from user's rated anime
    Returns (recommendations_df, alpha_used, n_rated).
    """
    user_factors = artifacts["user_factors"]
    item_factors = artifacts["item_factors"]
    user_means = artifacts["user_means"]
    anime_similarity = artifacts["anime_similarity"]
    content_matrix_norm = artifacts["content_matrix_norm"]
    train_matrix = artifacts["train_matrix"]
    anime_meta = artifacts["anime_meta"]
    shift = artifacts["shift"]
    n_anime = artifacts["n_anime"]
    user2idx = artifacts["user2idx"]
    idx2anime = artifacts["idx2anime"]
    genre_classes = artifacts["genre_classes"]

    # Blend seed items into query_vector if available
    if seed_indices:
        seed_vec = content_matrix_norm[seed_indices].mean(axis=0)
        query_vector = 0.8 * seed_vec + 0.2 * query_vector
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

    # Fallback to content-only if user not in model
    if dataset_uid not in user2idx:
        recs = recommend_by_query_vector(query_vector, method, artifacts, top_n)
        return recs, 0.3, 0

    user_idx = user2idx[dataset_uid]

    # ── CF scores (ALS) ───────────────────────────────────────────────────────
    cf_raw = np.dot(user_factors[user_idx], item_factors.T)  # (n_anime,)
    cf_scores = np.clip((cf_raw - shift) + user_means[user_idx], 1.0, 10.0)
    cf_norm = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min() + 1e-8)

    # ── Content scores: query vector via FAISS ────────────────────────────────
    faiss_scores, faiss_indices = _faiss_search(
        query_vector, artifacts, top_k=min(500, n_anime)  # Increased from 200 to 500
    )
    cb_query = np.zeros(n_anime, dtype=np.float32)
    for sc, idx in zip(faiss_scores, faiss_indices):
        if 0 <= idx < n_anime:
            cb_query[idx] = float(sc)
    cb_query_norm = (cb_query - cb_query.min()) / (
        cb_query.max() - cb_query.min() + 1e-8
    )

    # ── Content scores: history-based ─────────────────────────────────────────
    user_row = train_matrix[user_idx]
    rated_indices = user_row.indices
    rated_indices = rated_indices[rated_indices < n_anime]
    rated_values = np.array(user_row.data, dtype=np.float32)[: len(rated_indices)]

    if len(rated_indices) > 0:
        weights = rated_values - user_means[user_idx]
        weights = np.clip(weights, 0, None)
        if weights.sum() > 0:
            weights = weights / weights.sum()
            cb_hist = np.zeros(n_anime, dtype=np.float32)
            for sim_idx, w in zip(rated_indices, weights):
                cb_hist += w * anime_similarity[sim_idx]
        else:
            cb_hist = anime_similarity[rated_indices].mean(axis=0)
    else:
        cb_hist = np.zeros(n_anime, dtype=np.float32)

    cb_hist_norm = (cb_hist - cb_hist.min()) / (cb_hist.max() - cb_hist.min() + 1e-8)

    # Combined content: Query-First logic (80% query if specific, else 40%)
    query_weight = 0.8 if method != "fallback" else 0.4
    cb_combined = query_weight * cb_query_norm + (1 - query_weight) * cb_hist_norm

    # ── Adaptive alpha ─────────────────────────────────────────────────────────
    n_rated = len(rated_indices)
    if n_rated < 20:
        alpha = 0.3
    elif n_rated < 100:
        alpha = 0.5
    else:
        alpha = 0.7

    # ── Hybrid score ───────────────────────────────────────────────────────────
    # If we have a specific query, REDUCE alpha to let the query dominate history
    if seed_indices:
        alpha = alpha * 0.3  # Very aggressive override if specific title matched
    elif method != "fallback":
        alpha = alpha * 0.6  # Standard override for semantic/keyword search
    
    hybrid = alpha * cf_norm + (1 - alpha) * cb_combined

    # Exclude already watched
    if len(rated_indices) > 0:
        hybrid[rated_indices] = -1.0

    top_indices = np.argsort(hybrid)[::-1][: top_n + 30]

    results = []
    for a_idx in top_indices:
        if hybrid[a_idx] < 0:
            continue
        original_id = idx2anime.get(int(a_idx))
        if original_id is None:
            continue
        row = anime_meta[anime_meta["anime_id"] == original_id]
        if row.empty:
            continue
        info = row.iloc[0]

        # Source label for explainability
        cf_contribution = alpha * cf_norm[a_idx]
        cb_contribution = (1 - alpha) * cb_combined[a_idx]
        dominant_source = (
            "Collaborative filtering"
            if cf_contribution > cb_contribution
            else "Content match"
        )

        why = _make_explanation(
            info["Genres"],
            genre_classes,
            query_vector,
            content_matrix_norm,
            a_idx,
            method,
        )

        results.append(
            {
                "anime_id": original_id,
                "name": info["display_name"],
                "genres": info["Genres"],
                "type": info["Type"],
                "score": info["Score"],
                "hybrid_score": round(float(hybrid[a_idx]), 4),
                "cf_score": round(float(cf_norm[a_idx]), 4),
                "cb_score": round(float(cb_combined[a_idx]), 4),
                "why": why,
                "source": dominant_source,
            }
        )
        if len(results) >= top_n:
            break

    return pd.DataFrame(results), alpha, n_rated


# ── DB helpers ────────────────────────────────────────────────────────────────
def verify_login(username: str, password: str, db_path: str) -> dict | None:
    import hashlib

    pw_hash = hashlib.sha256(password.encode()).hexdigest()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username, dataset_uid FROM users WHERE username=? AND password_hash=?",
        (username, pw_hash),
    )
    row = cur.fetchone()
    conn.close()
    return {"id": row[0], "username": row[1], "dataset_uid": row[2]} if row else None


def get_user_top_rated(
    dataset_uid: int, artifacts: dict, top_n: int = 10
) -> pd.DataFrame:
    user2idx = artifacts["user2idx"]
    idx2anime = artifacts["idx2anime"]
    train_matrix = artifacts["train_matrix"]
    anime_meta = artifacts["anime_meta"]

    if dataset_uid not in user2idx:
        return pd.DataFrame()

    user_idx = user2idx[dataset_uid]
    user_row = train_matrix[user_idx]
    indices = user_row.indices
    values = user_row.data

    if len(indices) == 0:
        return pd.DataFrame()

    sorted_idx = np.argsort(values)[::-1][:top_n]
    results = []
    for i in sorted_idx:
        a_idx = indices[i]
        original_id = idx2anime.get(int(a_idx))
        if original_id is None:
            continue
        row = anime_meta[anime_meta["anime_id"] == original_id]
        if row.empty:
            continue
        info = row.iloc[0]
        results.append(
            {
                "name": info["display_name"],
                "genres": info["Genres"],
                "rating": int(values[i]),
            }
        )

    return pd.DataFrame(results)
