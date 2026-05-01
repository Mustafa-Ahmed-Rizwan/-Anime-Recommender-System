# engine/faiss_index.py
import faiss
import numpy as np
import pickle


def load_faiss(model_dir: str, processed_dir: str) -> dict:
    """Load FAISS index and related artifacts."""
    index = faiss.read_index(f"{model_dir}/faiss_index.bin")
    content_matrix_norm = np.load(f"{model_dir}/content_matrix_norm.npy")

    with open(f"{processed_dir}/feature_meta.pkl", "rb") as f:
        feature_meta = pickle.load(f)

    return {
        "index": index,
        "content_matrix_norm": content_matrix_norm,
        "genre_classes": feature_meta["genre_classes"],
        "top_studios": feature_meta["top_studios"],
    }


def search_faiss(
    query_vector: np.ndarray, faiss_artifacts: dict, top_k: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index with a query vector.
    Returns (scores, indices) of top_k results.
    query_vector must be L2-normalized, shape (dim,).
    """
    index = faiss_artifacts["index"]
    q = query_vector.astype(np.float32).reshape(1, -1)

    # Normalize query
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm

    D, I = index.search(q, top_k)
    return D[0], I[0]  # scores and indices


def build_genre_query_vector(
    genre_weights: dict, faiss_artifacts: dict, content_matrix_norm: np.ndarray
) -> np.ndarray:
    """
    Build a query vector from genre weights by averaging
    the content vectors of anime that match those genres.
    Used as fallback when HF API is unavailable.
    """
    # This is not used in the main path but kept as fallback
    n = content_matrix_norm.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    genre_classes = faiss_artifacts["genre_classes"]

    for genre, weight in genre_weights.items():
        for i, gc in enumerate(genre_classes):
            if genre.lower() == gc.lower():
                scores += weight * content_matrix_norm[:, i]

    if scores.sum() == 0:
        return content_matrix_norm.mean(axis=0)

    # Return the weighted average as a pseudo query vector
    top_idx = np.argsort(scores)[::-1][:20]
    query_vec = content_matrix_norm[top_idx].mean(axis=0)
    norm = np.linalg.norm(query_vec)
    return query_vec / (norm + 1e-8)
