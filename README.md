CRGNN+ Synthetic Multimodal Movie Scene Dataset
Associated Paper:
> *Emotion-Aware Multi-Perspective Movie Summarization via Adaptive Multimodal Fusion and Causal Narrative Graphs*
> Dr. Ashwini M Joshi, Adishree Gupta, Sai Roshini Kolla, Gaurav M
> Department of CSE, PES University, Bangalore, India
> Submitted to IEEE
---
Overview
This dataset package provides a synthetic but structurally realistic multimodal representation of movie scenes, designed to support reproducible experimentation with the CRGNN+ architecture. All data is programmatically generated under a fixed random seed (`42`) — no real copyrighted movie content is included.
The dataset covers 20 scenes across three narrative genres (action, thriller, drama) and exposes three parallel modality streams — audio, visual, and subtitle — alongside six-class emotion annotations from three narrative perspectives (protagonist, antagonist, narrator), directly mirroring the evaluation protocol described in the paper.
> **Full-scale dataset and extended version will be released upon publication.**
---
File Descriptions
`metadata.csv`
Scene-level descriptive attributes. One row per scene.
Column	Type	Description
`scene_id`	string	Unique scene identifier (`scene_001` … `scene_020`)
`start_time_s`	int	Scene start time in seconds (cumulative)
`end_time_s`	int	Scene end time in seconds (cumulative)
`duration_s`	int	Scene duration in seconds (range: 30–120 s)
`genre`	string	Narrative genre: `action`, `thriller`, or `drama`
`tension_score`	float	Scalar tension in [0, 1]; mirrors τ in the paper's Eq. (7)
`shot_count`	int	Number of shots detected within the scene
`has_dialogue`	int	Binary flag: 1 = dialogue present, 0 = silent/action
---
`audio_features.npy`
Shape: `(20, 40)` — `float32`
Each row is a 40-dimensional cepstral mean-variance normalised MFCC feature vector for one scene, consistent with the 40-coefficient MFCC extraction described in §IV-B of the paper (25 ms Hamming window, 10 ms hop). Lower-index coefficients carry higher variance, matching real-world MFCC spectral structure.
Load with:
```python
import numpy as np
audio = np.load("audio_features.npy")   # shape (20, 40)
```
---
`visual_features.npy`
Shape: `(20, 512)` — `float32`
L2-normalised 512-dimensional CNN keyframe embeddings, one per scene. Mimics ResNet-style spatial-entropy-maximising keyframe representations produced by the P1 visual module (§IV-A).
Load with:
```python
visual = np.load("visual_features.npy")   # shape (20, 512)
```
---
`subtitle_embeddings.npy`
Shape: `(20, 3, 768)` — `float32`
L2-normalised 768-dimensional subtitle embeddings (BERT CLS-token style) for each scene × perspective triplet.
Axis 0: scene index (0–19)
Axis 1: perspective index — `0 = protagonist`, `1 = antagonist`, `2 = narrator`
Axis 2: embedding dimension (768)
Protagonist and antagonist embeddings are mutually correlated (low perturbation); narrator embeddings are more orthogonal (high perturbation), consistent with the inter-perspective Jensen–Shannon divergence reported in Table III of the paper (JSD_prot-nar = 0.76 > JSD_prot-ant = 0.44).
Load with:
```python
subs = np.load("subtitle_embeddings.npy")   # shape (20, 3, 768)
protagonist_embs = subs[:, 0, :]
antagonist_embs  = subs[:, 1, :]
narrator_embs    = subs[:, 2, :]
```
---
`emotion_labels.csv`
Six-class emotion probability distributions and perspective-conditioned dominant labels. One row per scene.
Column	Type	Description
`scene_id`	string	Scene identifier
`dominant_emotion`	string	Argmax class across all 6 probabilities
`p_happy` … `p_fearful`	float	Probability for each of the 6 emotion classes (sums to 1.0)
`protagonist_dominant`	string	Dominant emotion from protagonist subspace
`antagonist_dominant`	string	Dominant emotion from antagonist subspace
`narrator_dominant`	string	Dominant emotion from narrator subspace
Emotion classes (6): `happy`, `sad`, `tense`, `calm`, `surprised`, `fearful`
These correspond directly to the six-class taxonomy used by the MSP-Podcast-trained audio classifier in §IV-B.
---
`generate_dataset.py`
Self-contained Python script that regenerates all five dataset files from scratch. Requires only `numpy` and `pandas`.
```bash
python generate_dataset.py
```
All outputs are written to the current working directory. The fixed seed (`SEED = 42`) guarantees byte-identical reproduction across platforms.
---
Alignment with CRGNN+ Architecture
Dataset Element	Architecture Component	Paper Section
`audio_features.npy`	P2: MFCC Audio Classifier → entropy-calibrated fusion	§III-B, §IV-B
`visual_features.npy`	P1: Keyframe CNN Encoder → CausalNode.salience	§III-D, §IV-A
`subtitle_embeddings.npy`	P3: GNN Subtitle Encoder → perspective subspaces	§III-C, §IV-C
`emotion_labels.csv` (dist.)	Shannon entropy computation (Eq. 1–2), arc modeling	§III-B, §III-E
`emotion_labels.csv` (persp.)	Protagonist/antagonist/narrator subspace validation	§III-C, §V-B
`metadata.csv` (tension)	Causal DAG edge weight τ (Eq. 7), counterfactual propagation	§III-D, §V-I
`metadata.csv` (has_dialogue)	Modality reliability signal for adaptive fusion	§III-B
---
Requirements
```
python >= 3.8
numpy  >= 1.21
pandas >= 1.3
```
Install dependencies:
```bash
pip install numpy pandas
```
---
Citation
If you use this dataset or the CRGNN+ architecture, please cite:
```
@article{joshi2025crgnn,
  title     = {Emotion-Aware Multi-Perspective Movie Summarization via
               Adaptive Multimodal Fusion and Causal Narrative Graphs},
  author    = {Joshi, Ashwini M and Gupta, Adishree and Kolla, Sai Roshini and M, Gaurav},
  journal   = {IEEE},
  year      = {2026},
  note      = {Under review}
}
```
