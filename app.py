# app.py
import streamlit as st
import pandas as pd
import sys
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AniRec — Anime Recommender", page_icon="🎌", layout="wide"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR = "model"
PROCESSED_DIR = "processed"
DB_PATH = "anime_recommender.db"


# ── Load artifacts once ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading recommendation engine...")
def load_engine():
    from engine.recommender import load_artifacts

    return load_artifacts(MODEL_DIR, PROCESSED_DIR)


artifacts = load_engine()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.card {
    background: #1a1a2e;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    border-left: 4px solid #e94560;
}
.card h4 { color: #e94560; margin: 0 0 6px 0; font-size: 1rem; }
.card .genres { color: #a0a0b0; font-size: 0.8rem; margin: 4px 0; }
.card .score  { color: #ffd700; font-size: 0.85rem; }
.card .why    { color: #80c0ff; font-size: 0.8rem; font-style: italic; margin-top: 6px; }
.card .badge  { background: #e94560; color: white; border-radius: 8px;
                padding: 2px 8px; font-size: 0.75rem; margin-right: 4px; }
.history-card {
    background: #0f3460;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.history-card .title { color: #ffffff; font-size: 0.9rem; font-weight: bold; }
.history-card .info  { color: #a0c4ff; font-size: 0.75rem; }
.metric-box {
    background: #16213e;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    margin-bottom: 10px;
}
.metric-box .label { color: #a0a0b0; font-size: 0.75rem; }
.metric-box .value { color: #e94560; font-size: 1.4rem; font-weight: bold; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state defaults ────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "recs" not in st.session_state:
    st.session_state.recs = None
if "alpha_used" not in st.session_state:
    st.session_state.alpha_used = None
if "nlu_genres" not in st.session_state:
    st.session_state.nlu_genres = {}
if "nlu_method" not in st.session_state:
    st.session_state.nlu_method = ""
if "n_rated" not in st.session_state:
    st.session_state.n_rated = 0
if "nlu_seeds" not in st.session_state:
    st.session_state.nlu_seeds = []


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎌 AniRec")
    st.markdown("---")

    if not st.session_state.logged_in:
        st.markdown("### Login")
        st.markdown(
            "*Login to get personalized recommendations based on your taste history.*"
        )
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login", use_container_width=True):
            from engine.recommender import verify_login

            user = verify_login(username, password, DB_PATH)
            if user:
                st.session_state.logged_in = True
                st.session_state.user = user
                st.session_state.recs = None
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("---")
        st.markdown("**Test accounts** (password: `test1234`)")
        for acc in [
            "action_fan",
            "romance_fan",
            "psychological_fan",
            "scifi_fan",
            "mixed_fan",
        ]:
            st.markdown(f"- `{acc}`")
    else:
        user = st.session_state.user
        st.markdown(f"### 👤 {user['username']}")
        st.markdown(f"*Persona: {user['username'].replace('_',' ').title()}*")

        if st.session_state.n_rated > 0:
            st.markdown(f"**Ratings in model:** {st.session_state.n_rated:,}")

        if st.session_state.alpha_used is not None:
            alpha = st.session_state.alpha_used
            st.markdown(f"**Alpha (CF weight):** {alpha}")
            if alpha == 0.3:
                st.info("Mode: Content-dominant (cold start)")
            elif alpha == 0.5:
                st.info("Mode: Balanced hybrid")
            else:
                st.success("Mode: Full hybrid (CF dominant)")

        st.markdown("---")
        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.recs = None
            st.session_state.alpha_used = None
            st.session_state.n_rated = 0
            st.rerun()

    st.markdown("---")
    st.markdown("### System Metrics")
    st.markdown(
        """
    <div class='metric-box'>
        <div class='label'>Precision@10</div>
        <div class='value'>0.36</div>
    </div>
    <div class='metric-box'>
        <div class='label'>Recall@10</div>
        <div class='value'>0.9986</div>
    </div>
    <div class='metric-box'>
        <div class='label'>Training Ratings</div>
        <div class='value'>44.5M</div>
    </div>
    <div class='metric-box'>
        <div class='label'>Anime in Model</div>
        <div class='value'>10,502</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Engine:** ALS + Content-Based Hybrid")
    st.markdown("**NLU:** Keyword + Sentence Transformers")


# ── MAIN PAGE ─────────────────────────────────────────────────────────────────
st.markdown("# 🎌 Anime Recommender System")

if st.session_state.logged_in:
    st.markdown(
        f"*Logged in as **{st.session_state.user['username']}** — recommendations are personalized using your taste history.*"
    )
else:
    st.markdown(
        "*Describe what you want to watch and get instant recommendations. Login for personalized results.*"
    )

st.markdown("---")

# ── NLU INPUT ─────────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])

with col_input:
    query = st.text_input(
        label="What do you want to watch?",
        placeholder='e.g. "dark psychological thriller with plot twists" or "funny romantic school anime"',
        key="nlu_query",
        label_visibility="collapsed",
    )

with col_btn:
    search_clicked = st.button("🔍 Recommend", use_container_width=True)

# ── RUN ENGINE ────────────────────────────────────────────────────────────────
if search_clicked and query.strip():
    from nlu.embedder import parse_query

    with st.spinner("Understanding your query..."):
        query_vector, genre_weights, seed_indices, method = parse_query(
            query.strip(),
            artifacts,
            artifacts["content_matrix_norm"],
            artifacts["anime_meta"],
        )
        st.session_state.nlu_genres = genre_weights
        st.session_state.nlu_method = method
        st.session_state.nlu_seeds = seed_indices

    if st.session_state.logged_in:
        dataset_uid = st.session_state.user["dataset_uid"]
        with st.spinner("Running hybrid recommendation engine..."):
            from engine.recommender import recommend_hybrid

            recs, alpha, n_rated = recommend_hybrid(
                dataset_uid,
                query_vector,
                method,
                artifacts,
                DB_PATH,
                top_n=10,
                seed_indices=seed_indices,
            )
            st.session_state.recs = recs
            st.session_state.alpha_used = alpha
            st.session_state.n_rated = n_rated
    else:
        with st.spinner("Finding matching anime..."):
            from engine.recommender import recommend_by_query_vector

            recs = recommend_by_query_vector(
                query_vector, method, artifacts, top_n=10, seed_indices=seed_indices
            )
            st.session_state.recs = recs

elif search_clicked and not query.strip():
    st.warning("Please type something to search.")

# ── SHOW NLU EXTRACTION ───────────────────────────────────────────────────────
if st.session_state.nlu_genres and st.session_state.recs is not None:
    method = st.session_state.nlu_method
    genres = st.session_state.nlu_genres
    method_label = {
        "keyword": "keyword matching",
        "semantic": "semantic understanding",
        "fallback": "general fallback",
    }.get(method, method)

    st.markdown(f"**Query understood via {method_label}. Extracted genres:**")
    badges = " ".join(
        [
            f"`{g}` ({round(w,2)})"
            for g, w in sorted(genres.items(), key=lambda x: -x[1])
        ]
    )
    st.markdown(badges)
    
    if st.session_state.nlu_seeds:
        seeds = st.session_state.nlu_seeds
        idx2anime = artifacts["idx2anime"]
        anime_meta = artifacts["anime_meta"]
        seed_names = []
        for s_idx in seeds:
            original_id = idx2anime.get(int(s_idx))
            if original_id:
                name = anime_meta[anime_meta["anime_id"] == original_id]["display_name"].iloc[0]
                seed_names.append(f"**{name}**")
        if seed_names:
            st.markdown(f"**Detected Anime:** {' · '.join(seed_names)}")
            
    st.markdown("---")

# ── SHOW RESULTS ──────────────────────────────────────────────────────────────
if st.session_state.recs is not None and not st.session_state.recs.empty:
    recs = st.session_state.recs

    if st.session_state.logged_in:
        # Two columns: history on left, recommendations on right
        col_hist, col_recs = st.columns([1, 2])

        with col_hist:
            st.markdown("### 📚 Your Taste History")
            st.markdown("*Top rated anime from your profile:*")
            from engine.recommender import get_user_top_rated

            history = get_user_top_rated(
                st.session_state.user["dataset_uid"], artifacts, top_n=10
            )
            if not history.empty:
                for _, row in history.iterrows():
                    stars = "⭐" * int(row["rating"])
                    st.markdown(
                        f"""
                    <div class='history-card'>
                        <div class='title'>{row['name']}</div>
                        <div class='info'>{row['genres'] if isinstance(row['genres'], str) else 'N/A'}</div>
                        <div class='info'>{stars} ({row['rating']}/10)</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No history found.")

        with col_recs:
            st.markdown("### 🎯 Recommended For You")
            st.markdown(
                f"*Hybrid engine — CF weight (alpha): **{st.session_state.alpha_used}***"
            )
            for i, row in recs.iterrows():
                score_key = (
                    "hybrid_score" if "hybrid_score" in recs.columns else "match_score"
                )
                st.markdown(
                    f"""
                <div class='card'>
                    <h4>{i+1}. {row['name']}</h4>
                    <div class='genres'>🎭 {row['genres'] if isinstance(row['genres'], str) else 'N/A'}</div>
                    <div class='score'>⭐ Community Score: {row['score'] if pd.notna(row['score']) else 'N/A'}
                    &nbsp;|&nbsp; Match: {row[score_key]}</div>
                    <div class='why'>💡 {row['why']}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
    else:
        st.markdown("### 🎯 Top Recommendations")
        st.markdown(
            "*Based on your query — login for personalized hybrid recommendations.*"
        )

        cols = st.columns(2)
        for i, row in recs.iterrows():
            with cols[i % 2]:
                st.markdown(
                    f"""
                <div class='card'>
                    <h4>{i+1}. {row['name']}</h4>
                    <div class='genres'>🎭 {row['genres'] if isinstance(row['genres'], str) else 'N/A'}</div>
                    <div class='score'>⭐ Community Score: {row['score'] if pd.notna(row['score']) else 'N/A'}
                    &nbsp;|&nbsp; Match: {row['match_score']}</div>
                    <div class='why'>💡 {row['why']}</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

elif st.session_state.recs is not None and st.session_state.recs.empty:
    st.warning("No recommendations found. Try a different query.")
else:
    st.markdown(
        """
    <div style='text-align:center; padding: 60px 0; color: #666;'>
        <div style='font-size: 3rem;'>🎌</div>
        <div style='font-size: 1.1rem; margin-top: 12px;'>
            Type what you want to watch above and hit Recommend
        </div>
        <div style='font-size: 0.85rem; margin-top: 8px; color: #444;'>
            Try: "dark psychological thriller" · "funny school romance" · "epic fantasy adventure"
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
