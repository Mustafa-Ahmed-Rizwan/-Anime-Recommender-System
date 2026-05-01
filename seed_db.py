import sqlite3
import pandas as pd
import numpy as np
import hashlib
import os
import pickle
import json

# ── CONFIG ──────────────────────────────────────────────────────────────────
DB_PATH = "anime_recommender.db"
PROCESSED_DIR = "processed"
RATINGS_PATH = "data/rating_complete.csv"  # change to your actual path

TEST_ACCOUNTS = {
    "action_fan": {"dataset_uid": 127483, "password": "test1234"},
    "romance_fan": {"dataset_uid": 327150, "password": "test1234"},
    "psychological_fan": {"dataset_uid": 332300, "password": "test1234"},
    "scifi_fan": {"dataset_uid": 117521, "password": "test1234"},
    "mixed_fan": {"dataset_uid": 353398, "password": "test1234"},
}


# ── HELPERS ──────────────────────────────────────────────────────────────────
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ── STEP 1: CREATE TABLES ────────────────────────────────────────────────────
print("\n[1/6] Creating database and tables...")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.executescript("""
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    dataset_uid   INTEGER,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ratings (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  INTEGER NOT NULL,
    anime_id INTEGER NOT NULL,
    rating   INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS anime (
    anime_id     INTEGER PRIMARY KEY,
    name         TEXT,
    genres       TEXT,
    type         TEXT,
    score        REAL,
    episodes     TEXT,
    studios      TEXT,
    members      INTEGER,
    display_name TEXT
);

CREATE INDEX IF NOT EXISTS idx_ratings_user  ON ratings(user_id);
CREATE INDEX IF NOT EXISTS idx_ratings_anime ON ratings(anime_id);
""")
conn.commit()
print("    Tables created.")

# ── STEP 2: LOAD ANIME METADATA ───────────────────────────────────────────────
print("\n[2/6] Loading anime metadata into DB...")
anime_meta = pd.read_parquet(f"{PROCESSED_DIR}/anime_meta.parquet")

anime_meta[
    [
        "anime_id",
        "display_name",
        "Genres",
        "Type",
        "Score",
        "Episodes",
        "Studios",
        "Members",
    ]
].rename(
    columns={
        "Genres": "genres",
        "Type": "type",
        "Score": "score",
        "Episodes": "episodes",
        "Studios": "studios",
        "Members": "members",
    }
).to_sql(
    "anime", conn, if_exists="replace", index=False
)

print(f"    Loaded {len(anime_meta)} anime rows.")

# ── STEP 3: CREATE TEST USER ACCOUNTS ────────────────────────────────────────
print("\n[3/6] Creating test user accounts...")
for username, info in TEST_ACCOUNTS.items():
    pw_hash = hash_password(info["password"])
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, dataset_uid) VALUES (?,?,?)",
            (username, pw_hash, info["dataset_uid"]),
        )
        print(f"    Created : {username}  (dataset_uid={info['dataset_uid']})")
    except sqlite3.IntegrityError:
        print(f"    Skipped : {username} already exists")
conn.commit()

# ── STEP 4: LOAD RATINGS FROM CSV ────────────────────────────────────────────
print("\n[4/6] Reading ratings from CSV (chunk-based, may take a minute)...")
with open(f"{PROCESSED_DIR}/mappings.pkl", "rb") as f:
    mappings = pickle.load(f)
valid_anime_ids = set(mappings["anime2idx"].keys())

dataset_uids_needed = [v["dataset_uid"] for v in TEST_ACCOUNTS.values()]
collected = []
CHUNK = 1_000_000

for chunk in pd.read_csv(
    RATINGS_PATH,
    dtype={"user_id": "int32", "anime_id": "int32", "rating": "int8"},
    chunksize=CHUNK,
):
    filtered = chunk[chunk["user_id"].isin(dataset_uids_needed)]
    if len(filtered) > 0:
        collected.append(filtered)

ratings_seed = pd.concat(collected, ignore_index=True)
print(f"    Collected {len(ratings_seed):,} ratings total.")
print(
    ratings_seed.groupby("user_id")
    .size()
    .reset_index(name="n_ratings")
    .to_string(index=False)
)

# ── STEP 5: INSERT RATINGS INTO DB ───────────────────────────────────────────
print("\n[5/6] Inserting ratings into DB...")
user_rows = pd.read_sql("SELECT id, username, dataset_uid FROM users", conn)
dataset_uid_to_db_id = dict(zip(user_rows["dataset_uid"], user_rows["id"]))

inserted_total = 0
for username, info in TEST_ACCOUNTS.items():
    dataset_uid = info["dataset_uid"]
    db_user_id = dataset_uid_to_db_id[dataset_uid]

    user_ratings = ratings_seed[
        (ratings_seed["user_id"] == dataset_uid)
        & (ratings_seed["anime_id"].isin(valid_anime_ids))
    ][["anime_id", "rating"]].copy()

    user_ratings["user_id"] = db_user_id
    user_ratings[["user_id", "anime_id", "rating"]].to_sql(
        "ratings", conn, if_exists="append", index=False
    )

    inserted_total += len(user_ratings)
    print(
        f"    {username}: {len(user_ratings):,} ratings inserted (db_id={db_user_id})"
    )

conn.commit()
print(f"    Total inserted: {inserted_total:,}")

# ── STEP 6: VERIFY ───────────────────────────────────────────────────────────
print("\n[6/6] Verification...")

print("\n  Users:")
print(
    pd.read_sql("SELECT id, username, dataset_uid FROM users", conn).to_string(
        index=False
    )
)

print("\n  Ratings per user:")
print(
    pd.read_sql(
        """
    SELECT u.username,
           COUNT(r.id)          AS n_ratings,
           ROUND(AVG(r.rating), 2) AS avg_rating,
           MIN(r.rating)        AS min_r,
           MAX(r.rating)        AS max_r
    FROM ratings r
    JOIN users u ON u.id = r.user_id
    GROUP BY u.username
""",
        conn,
    ).to_string(index=False)
)

print("\n  Anime table row count:")
print(
    pd.read_sql("SELECT COUNT(*) AS total_anime FROM anime", conn).to_string(
        index=False
    )
)

print("\n  Top 10 rated anime for action_fan:")
print(
    pd.read_sql(
        """
    SELECT a.display_name, r.rating
    FROM ratings r
    JOIN users  u ON u.id       = r.user_id
    JOIN anime  a ON a.anime_id = r.anime_id
    WHERE u.username = 'action_fan'
    ORDER BY r.rating DESC
    LIMIT 10
""",
        conn,
    ).to_string(index=False)
)

conn.close()

# ── SAVE CREDENTIALS JSON ─────────────────────────────────────────────────────
credentials = {
    username: {
        "password": info["password"],
        "dataset_uid": info["dataset_uid"],
        "persona": username.replace("_", " ").title(),
    }
    for username, info in TEST_ACCOUNTS.items()
}
with open("test_credentials.json", "w") as f:
    json.dump(credentials, f, indent=2)

print(f"\nDB saved   : {os.path.abspath(DB_PATH)}")
print(f"DB size    : {os.path.getsize(DB_PATH)/1e6:.1f} MB")
print(f"Credentials: {os.path.abspath('test_credentials.json')}")
print("\nAll done. Run app.py next.")
