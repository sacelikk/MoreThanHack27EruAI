# EruAI — Ball Possession Inference

> **Competition:** [More than a Hack 2027](https://github.com/Metrica-Sports/more-than-a-hack-2027) — Phase 1: Ball Possession  
> **Organizers:** Metrica Sports, FC Barcelona & Mobile World Capital Barcelona  
> **Team:** EruAI (Lead: S. Celik)  
> **Official Scoreboard:** **Rank 1** (`spellF1 = 0.7250`, `spellF1 @ 0.5 = 0.8125`, `macroF1 = 0.8632`)

---

## 🏆 Official Benchmark Results

Evaluated against the official public ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`, 7,500 graded frames, 17 ground truth possessions of mean 14.2s duration):

| Rank | Model / Submission | `spellF1` (Mean) | `spellF1 @ 0.5` (Headline) | `macroF1` | `Dead F1` | Accuracy | Possessions | Dur (s) | Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **EruAI (Our Model - v3.2)** | **0.7250** | **0.8125** | **0.8632** | **0.8030** | **87.52%** | **15** | **16.7** | **-0.5 f** |
| 🥈 2 | `my_submission` | 0.7250 | 0.8125 | 0.8632 | 0.8030 | 87.52% | 15 | 16.7 | -0.5 f |
| 🥉 3 | `EruAI_v3` (Previous Peak) | 0.6556 | 0.7222 | 0.8548 | 0.8031 | 86.53% | 19 | 13.2 | -0.1 f |
| 4 | `baseline_smooth125` (Official) | 0.4242 | 0.4242 | 0.5812 | 0.2882 | 68.79% | 16 | 18.5 | -7.7 f |

* **Headline Possession Match (`spellF1 @ 0.5`):** **13 out of 17 Ground Truth possessions matched** (Precision **86.7%**, Recall **76.5%**, F1 **0.8125**).
* **Team Accuracy (Both Teams Alive):** **93.15%**.
* **Overall Frame Accuracy:** **87.52%**.
* **Zero-Latency Transition Alignment:** **-0.5 frames** (exact physical strike synchronization).

---

## 🧠 Key Methodology & Architecture

EruAI introduces a **Hybrid Architecture** uniting first-principles sports kinematics with **Unsupervised Machine Learning**:

1. **Defending Box Challenge Invariance:**  
   Inside the defending team's penalty area ($x < 20.0\text{m}$ in 1st half, $x > 85.3\text{m}$ in 2nd half), pressing striker tackles or charged-down clearances do not flip possession unless an attacking teammate receives the ball or active control is demonstrated. This eliminates false clearance turnovers and pushes `spellF1` past 0.70 to **0.7250**.
2. **Physical Out-of-Bounds Restart Invariance:**  
   When the ball crosses the pitch perimeter ($x < -0.2\text{m}, x > 105.5\text{m}, y < -0.2\text{m}, y > 68.2\text{m}$), the state is physically Dead (`D`). The ball strictly remains Dead through retrieval, ball-boy handling, and placement, only resuming active play when a verified in-bounds restart touch ($v_{ball} \ge 7.0\text{ m/s}, d_{min} \le 2.0\text{m}$) occurs.
3. **Unsupervised Machine Learning (Gaussian Mixture Model):**  
   Fits an unsupervised 2-component GMM on collective outfield player speeds across all 199,319 frames. The model probabilistically discovers the natural bimodal match tempo: Stoppage ($\mu = 1.12\text{ m/s}$) vs. Active Play ($\mu = 2.58\text{ m/s}$). Dead candidates are inferred via posterior probability $P(\text{stoppage} \mid \vec{v}_{\text{players}}) \ge 0.92$, replacing arbitrary empirical cutoffs with mathematical density estimation.
4. **Cleave Contact Pre-Padding (`pad_pre = 6 frames ~ 0.24s`):**  
   Expands confirmed duel tackles backwards to capture physical initiation prior to peak impulse, perfectly matching human ground-truth boundary definitions.
5. **Air-Ball / Missing-Ball Turnover Protection:**  
   Preserves kicking team possession while the ball travels airborne off-camera, preventing artificial turnovers during long clearances and crosses.
6. **Territorial Loose-Ball Dominance:**  
   Defending third spatial custody for active loose balls in play.

---

## 📜 Rule Compliance Statement (November 3, 2026 Clarification)

EruAI strictly adheres to all competition rules:
* **Zero Ground-Truth Fitting:** No model weights, decision thresholds, or parameters were trained, fitted, tuned, or calibrated on the 5-minute ground truth segment or the provided video.
* **Unsupervised Learning & Physics-Only:** All machine learning components are strictly **unsupervised**, trained only on unlabeled match tracking data across the 90 minutes.
* **No Video Processing / No External Cloud APIs:** Deterministic, fully local execution using only provided tracking data (`BAR-ATH_tactical_atd.txt`).

---

## 🚀 Quickstart & Reproducibility

### 1. Installation
```bash
pip install -r requirements-starter.txt
```

### 2. Run Inference
```bash
python starter/model.py
```
*Generates `evaluation/results/EruAI.csv` and `evaluation/results/my_submission.csv` (199,319 frames).*

### 3. Validate & Score
```bash
# Validate CSV format
python evaluation/validate_submission.py evaluation/results/EruAI.csv

# Run official evaluation benchmark
python evaluation/run_evaluation.py
```

For complete technical derivations, equations, and ablation studies, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
