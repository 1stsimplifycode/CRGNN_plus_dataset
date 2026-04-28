"""
CRGNN+: Emotion-Aware Multi-Perspective Movie Summarization
Dataset Generation Script

Generates synthetic but structurally realistic multimodal movie scene data.
Fixed random seed ensures full reproducibility.

Usage:
    python generate_dataset.py

Outputs:
    metadata.csv
    audio_features.npy
    visual_features.npy
    subtitle_embeddings.npy
    emotion_labels.csv
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
N_SCENES          = 20
EMOTION_CLASSES   = ["happy", "sad", "tense", "calm", "surprised", "fearful"]
GENRES            = ["action", "thriller", "drama", "action", "thriller",
                     "drama", "action", "thriller", "drama", "action",
                     "thriller", "drama", "action", "thriller", "drama",
                     "action", "thriller", "drama", "action", "drama"]

AUDIO_DIM         = 40     # 40-coefficient MFCC vectors
VISUAL_DIM        = 512    # CNN frame embedding dimension
SUBTITLE_DIM      = 768    # BERT-style subtitle embedding dimension

PERSPECTIVES      = ["protagonist", "antagonist", "narrator"]

# ── 1. Scene Metadata ─────────────────────────────────────────────────────────
def generate_metadata():
    scene_ids  = [f"scene_{i:03d}" for i in range(1, N_SCENES + 1)]

    # Cumulative timestamps (each scene 30–120 s)
    durations  = rng.integers(30, 121, size=N_SCENES)
    start_times, end_times = [], []
    cursor = 0
    for d in durations:
        start_times.append(cursor)
        cursor += int(d)
        end_times.append(cursor)

    tension_scores = np.round(rng.uniform(0.1, 0.95, N_SCENES), 4)
    shot_counts    = rng.integers(3, 18, size=N_SCENES)
    has_dialogue   = rng.choice([True, False], size=N_SCENES, p=[0.75, 0.25])

    df = pd.DataFrame({
        "scene_id"      : scene_ids,
        "start_time_s"  : start_times,
        "end_time_s"    : end_times,
        "duration_s"    : durations,
        "genre"         : GENRES[:N_SCENES],
        "tension_score" : tension_scores,
        "shot_count"    : shot_counts,
        "has_dialogue"  : has_dialogue.astype(int),
    })
    df.to_csv("metadata.csv", index=False)
    print(f"[✓] metadata.csv          — {len(df)} scenes")
    return df


# ── 2. Emotion Labels ─────────────────────────────────────────────────────────
def generate_emotion_labels(metadata_df):
    """
    Each scene has:
      - A dominant emotion (argmax) consistent with genre/tension.
      - A full 6-class probability distribution (sums to 1.0).
      - Per-perspective dominant emotion labels.
    """
    rows = []
    for _, row in metadata_df.iterrows():
        # Bias distribution toward tense/fearful for high-tension scenes
        alpha = np.ones(6)
        if row["tension_score"] > 0.6:
            alpha[[2, 5]] += 4   # tense, fearful
        elif row["genre"] == "drama":
            alpha[[1, 3]] += 3   # sad, calm
        else:
            alpha[[0, 4]] += 2   # happy, surprised

        dist = rng.dirichlet(alpha)
        dist = np.round(dist, 4)
        dist[-1] = round(1.0 - dist[:-1].sum(), 4)   # enforce sum=1

        dominant = EMOTION_CLASSES[int(np.argmax(dist))]

        # Per-perspective labels (slight variation around dominant)
        persp_labels = {}
        for p in PERSPECTIVES:
            p_alpha = alpha.copy() + rng.uniform(0, 1.5, 6)
            p_dist  = rng.dirichlet(p_alpha)
            persp_labels[f"{p}_dominant"] = EMOTION_CLASSES[int(np.argmax(p_dist))]

        entry = {
            "scene_id"       : row["scene_id"],
            "dominant_emotion": dominant,
            **{f"p_{c}": v for c, v in zip(EMOTION_CLASSES, dist)},
            **persp_labels,
        }
        rows.append(entry)

    df = pd.DataFrame(rows)
    df.to_csv("emotion_labels.csv", index=False)
    print(f"[✓] emotion_labels.csv    — {len(df)} scenes × {len(EMOTION_CLASSES)} classes")
    return df


# ── 3. Audio Features (MFCC) ──────────────────────────────────────────────────
def generate_audio_features():
    """
    Shape: (N_SCENES, AUDIO_DIM)
    Each row = 40-dim cepstral mean-variance normalised MFCC vector.
    Lower coefficients have higher variance (mimics real MFCC structure).
    """
    scale = np.linspace(3.0, 0.3, AUDIO_DIM)   # c0 most variable
    features = rng.normal(loc=0.0, scale=scale, size=(N_SCENES, AUDIO_DIM))
    features = np.round(features.astype(np.float32), 6)
    np.save("audio_features.npy", features)
    print(f"[✓] audio_features.npy    — shape {features.shape}  dtype float32")
    return features


# ── 4. Visual Features ────────────────────────────────────────────────────────
def generate_visual_features():
    """
    Shape: (N_SCENES, VISUAL_DIM)
    L2-normalised 512-dim CNN keyframe embeddings (ResNet-style).
    """
    raw      = rng.standard_normal((N_SCENES, VISUAL_DIM)).astype(np.float32)
    norms    = np.linalg.norm(raw, axis=1, keepdims=True)
    features = np.round(raw / norms, 6)
    np.save("visual_features.npy", features)
    print(f"[✓] visual_features.npy   — shape {features.shape}  dtype float32")
    return features


# ── 5. Subtitle Embeddings ────────────────────────────────────────────────────
def generate_subtitle_embeddings():
    """
    Shape: (N_SCENES, 3, SUBTITLE_DIM)
    Axis 1 indexes perspective: [protagonist, antagonist, narrator].
    Each 768-dim vector is L2-normalised (BERT CLS-token style).
    Protagonist/antagonist are correlated; narrator is more orthogonal.
    """
    base   = rng.standard_normal((N_SCENES, SUBTITLE_DIM)).astype(np.float32)
    embs   = np.zeros((N_SCENES, 3, SUBTITLE_DIM), dtype=np.float32)

    for i, noise_scale in enumerate([0.25, 0.35, 0.80]):
        perturbed = base + rng.standard_normal((N_SCENES, SUBTITLE_DIM)).astype(np.float32) * noise_scale
        norms     = np.linalg.norm(perturbed, axis=1, keepdims=True)
        embs[:, i, :] = perturbed / norms

    embs = np.round(embs, 6)
    np.save("subtitle_embeddings.npy", embs)
    print(f"[✓] subtitle_embeddings.npy — shape {embs.shape}  dtype float32")
    return embs


# ── 6. Sanity Checks ──────────────────────────────────────────────────────────
def run_sanity_checks():
    meta   = pd.read_csv("metadata.csv")
    emot   = pd.read_csv("emotion_labels.csv")
    audio  = np.load("audio_features.npy")
    visual = np.load("visual_features.npy")
    subs   = np.load("subtitle_embeddings.npy")

    assert len(meta)   == N_SCENES, "metadata row count mismatch"
    assert len(emot)   == N_SCENES, "emotion_labels row count mismatch"
    assert audio.shape == (N_SCENES, AUDIO_DIM),    f"audio shape: {audio.shape}"
    assert visual.shape== (N_SCENES, VISUAL_DIM),   f"visual shape: {visual.shape}"
    assert subs.shape  == (N_SCENES, 3, SUBTITLE_DIM), f"subtitle shape: {subs.shape}"

    prob_cols = [f"p_{c}" for c in EMOTION_CLASSES]
    prob_sums = emot[prob_cols].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-3), "emotion probs do not sum to 1"

    print("\n[✓] All sanity checks passed.")
    print(f"    Scenes            : {N_SCENES}")
    print(f"    Emotion classes   : {EMOTION_CLASSES}")
    print(f"    Audio dim         : {AUDIO_DIM}")
    print(f"    Visual dim        : {VISUAL_DIM}")
    print(f"    Subtitle dim      : {SUBTITLE_DIM} × 3 perspectives")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  CRGNN+ Dataset Generator  |  seed =", SEED)
    print("=" * 60)

    meta_df = generate_metadata()
    generate_emotion_labels(meta_df)
    generate_audio_features()
    generate_visual_features()
    generate_subtitle_embeddings()
    run_sanity_checks()

    print("\nDataset generation complete.")
    print("Files written to:", os.path.abspath("."))
