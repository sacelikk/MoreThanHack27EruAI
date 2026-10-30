# Technical Report: Ball Possession Inference via Spatio-Temporal State Machine
**Competition:** More than a Hack 2027 — Phase 1: Ball Possession  
**Team Name:** EruAI  
**GitHub Repository:** [https://github.com/sacelikk/MoreThanHack27EruAI](https://github.com/sacelikk/MoreThanHack27EruAI)  
**Date:** October 2026  

---

## 1. Executive Summary

In Phase 1 of the *More than a Hack 2027* competition (organized by Metrica Sports, FC Barcelona, and Mobile World Capital Barcelona), the goal is to predict frame-by-frame ball possession (`H` for FC Barcelona, `A` for Athletic Club, `D` for Dead ball/stoppages, and `X` for unannotated/out-of-play periods) across all 199,319 frames of tactical broadcast footage.

The evaluation metric prioritizes **`spellF1`** (mean possession-spell F1 score across temporal IoU thresholds 0.3 to 0.7), broken by **`macroF1`** at the frame level.

Our solution, **EruAI v2.1**, introduces a **Domain-Specific Spatio-Temporal State Machine** augmented with:
1. **Air-Ball / Missing-Ball Turnover Protection:** Prevents premature possession switches while the ball is unobserved or traveling in the air across long clearances/lobs.
2. **Territorial Loose-Ball Bias:** Allocating loose balls in defensive thirds to the defending team.
3. **Dynamic Duel & Impulse Cleaver:** Automatically cleaving long spells at true tackle and acceleration impulses ($\ge 7.0\text{ m/s}$) where the tackler retains possession.
4. **Unconditional Walking Stoppage Detection:** Marking dead balls whenever pitch-wide outfield speed drops below $1.20\text{ m/s}$ or ball exits boundary lines.
5. **Human Perception Latency Alignment:** Synchronizing transitions with physical ball strikes (+3 frames).

On the official competition scoreboard, **EruAI v2.1** sets a new all-time benchmark at **Rank 1**:
* **`spellF1` (Mean IoU 0.3..0.7):** **`0.6333`** (+49.3% relative improvement over the official `baseline_smooth125` score of `0.4242`).
* **`spellF1` (Headline IoU $\ge 0.5$):** **`0.6667`** (12 out of 17 Ground Truth spells matched with 63.2% Precision and 70.6% Recall).
* **`macroF1`:** **`0.8035`** (+38.3% relative improvement over baseline `0.5812`).
* **Per-Frame Accuracy:** **`83.23%`** (Team accuracy both-alive: **`92.22%`**).
* **Canlı Top / Dead Ball F1 (`alive/dead F1`):** **`0.9361`**.
* **Transition Latency:** Reduced to **`+1.7 frames`** (~0.06s).

---

## 2. Approach & Architecture

```mermaid
flowchart TD
    A["Raw Tactical Tracking (199,319 frames)<br>Player positions, speeds & Ball coordinates"] --> B["Stage 1: Spatial Attribution & Separate Team Proximities (min_H, min_A)"]
    A --> C["Stage 2: Unconditional Walking Stoppage Detection<br>• Pitch Boundary Violation (x,y out of bounds)<br>• Outfield Walking Speed (< 1.20 m/s)"]
    B --> D["Stage 3: Territorial Loose-Ball Adaptation<br>Loose balls (> 3.5m) in defensive third assigned to defender"]
    D --> E["Stage 4: Temporal Majority Window (W = ±123 frames ~ 10s)<br>O(1) cumulative prefix sum consensus"]
    C --> F["Stage 5: Sustained Stoppage Filter<br>Only retain dead candidates lasting >= 100 frames (~4.0s)"]
    E & F --> G["Stage 6: Perception Latency Shift (+3 frames) & 'X' Masking"]
    G --> H["Stage 7: Dynamic Duel & Impulse Cleaver<br>Intercept spells at confirmed high-speed turnovers (spd >= 7.0 m/s, d <= 1.8m)"]
    H --> I["Stage 8: Air-Ball / Missing-Ball Turnover Protection<br>Hold kicking team possession while ball is airborne/unobserved"]
    I --> J["Final Submission: EruAI.csv (199,319 rows)"]
```

### 2.1 Algorithmic Formulation

#### 1. Air-Ball / Missing-Ball Turnover Protection (`fix_missing_ball_transitions`)
A symmetric rolling temporal window ($W = \pm 123$) suffers from forward lookahead leak across missing-ball spans. When a team clears the ball or kicks a high lob, the ball disappears from the tactical camera. A rolling window prematurely credits possession to the receiving team frames before the ball actually lands. By enforcing that possession cannot change hands while `ball_vis == False`, we eliminate 49 premature turnover artifacts across the 90 minutes, jumping `spellF1` from 0.6111 to **0.6333** and `macroF1` past 80% to **0.8035**.

#### 2. Spatial Attribution & Territorial Control (`extract_features_and_labels`)
For each frame $t$, distances to the closest Barcelona player ($d_H$) and Athletic player ($d_A$) are tracked. In open play where the ball is loose ($d_H > 3.5\text{m}$ and $d_A > 3.5\text{m}$), spatial pitch dominance assigns custody to the defending side ($x < 40\text{m}$ in 1st half, $x > 65\text{m}$ in 2nd half). This prevents trailing attackers from artificially stealing possession during clearances.

#### 3. Dynamic Duel & Impulse Cleaver (`receiver_cleaver`)
Fixed temporal windows (e.g. $W = 123$) can over-smooth short tackles (e.g. 0.8s deflections/steals). When an opponent approaches within $1.8\text{m}$, ball speed spikes $\ge 7.0\text{ m/s}$, and the opponent team receives the ball within 25 frames, the cleaver splits the continuous spell into two distinct possessions.

#### 4. Physical Dead-Ball Detection
Two complementary physical rules identify candidate stoppages:
* **Pitch Boundary Constraint:** If normalized ball coordinates satisfy $x_{norm} < -0.01$, $x_{norm} > 1.01$, $y_{norm} < -0.01$, or $y_{norm} > 1.01$, the ball is physically out of bounds (`is_out = True`).
* **Unconditional Walking Stoppage:** When all 22 outfield players drop to walking speed ($\bar{v}_{players} < 1.20\text{ m/s}$), play has unconditionally halted for a set-piece or foul, even if a player is standing immediately over the ball ($d < 1\text{m}$).

---

## 3. Official Scoreboard Comparison

Evaluated on the official ground truth (`BAR-ATH_possession_gt_public.csv`, 7,500 graded frames, 21 segments, 17 ground truth possessions):

```
rank  submission         macroF1  spellF1   poss     dur  transF1  lat_f  shareE  frag
----  -----------------  -------  -------  -----  ------  -------  -----  ------  ----
   1  EruAI (v2.1)        0.8035   0.6333     19    13.7   0.3333   +1.7   0.066  1.10
   2  baseline_smooth125  0.5812   0.4242     16    18.5   0.1667   -7.7   0.052  0.81
   3  baseline_smooth50   0.6207   0.3256     26    10.8   0.0816   +4.0   0.048  1.43
   4  baseline_smooth25   0.6200   0.2868     36     7.6   0.2034   +8.2   0.056  1.90
   5  baseline_raw        0.5972   0.0378    131     2.0   0.1410   -3.1   0.046  6.52
```

### Detailed Class Breakdown:
* **`spellF1`:** **0.6333** vs 0.4242 (+49.3% over official baseline).
* **`spellF1 @ 0.5`:** **0.6667** (12 of 17 Ground Truth spells matched).
* **`macroF1`:** **0.8035** vs 0.5812 (+38.3% over official baseline).
* **Per-Class F1:** Barcelona (`H`): **0.8443**, Athletic Club (`A`): **0.8925**, Dead Ball (`D`): **0.6737**.
* **Per-Frame Accuracy:** **83.23%**.
* **IoU Breakdown:**
  * IoU $\ge 0.3$: **0.7778** (14 / 17 matched)
  * IoU $\ge 0.4$: **0.7778** (14 / 17 matched)
  * IoU $\ge 0.5$: **0.6667** (12 / 17 matched)
  * IoU $\ge 0.6$: **0.5556** (10 / 17 matched)
  * IoU $\ge 0.7$: **0.3889** (7 / 17 matched)

---

## 4. Reproduction Guide

```bash
# 1. Run model
python starter/model.py

# 2. Validate submission formatting
python evaluation/validate_submission.py evaluation/results/EruAI.csv

# 3. Score against ground truth
python evaluation/run_evaluation.py
```
Outputs `EruAI rank 1 (spellF1: 0.6333, macroF1: 0.8035)`.
