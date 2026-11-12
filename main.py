#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01: rank-transition forecast with entry and exit.

Model (fully offline, stdlib only, deterministic — no RNG is needed, but
MAKEFUN_TRIAL_SEED is honoured for interface compatibility and ignored):

1. Parse /input/history.ndjson (one chart read per line) and /input/task.json.
2. From every ordered pair of reads (i, j), estimate per-rank membership curves
   q(rank, h) = P(game at `rank` at read i is in the top 10 h hours later),
   on a fixed knot grid in h, with shrinkage toward default priors.
   Ranks 1-14 are exact; deeper ranks are pooled into bands.
3. Estimate place kernels: P(at place k | in top 10, was at place/rank r), from
   short-horizon pairs (h <= 78h), shrunk toward priors.  Bench games (rank 11+)
   enter lower places more often, matching the archive.
4. Per-game covariates: momentum (rank slope over the last NW reads, positive
   slope = sliding down = demotion, negative slope = climbing), re-entry for
   fresh leavers (games on the chart within 48h of the last read), and current
   player count.
5. The scored window starts H0 hours after the last archived read (stale
   anchor); each hour t in the window is scored at effective horizon
   h_eff = H0 + t.  p(game, place) is the arithmetic mean over t=1..72 of
   q * kernel, named mass per place is capped, and the remainder is left
   unnamed (shared by the read's unnamed chart games, per the rules).
6. top10 = the ten games with the highest place-weighted expected score.

Outside the grader:
  MAKEFUN_INPUT_DIR=... MAKEFUN_OUTPUT_DIR=... python3 main.py
"""

import json
import math
import os
from collections import defaultdict
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# ---- hyperparameters (tuned by rolling 72-hour backtests over the archive) ----
PRIOR_W = 480.0  # pseudo-count blending archive-estimated curves toward the priors
MOM = 2.0      # climb multiplier: q *= 1 + MOM * min(-slope, 3) for bench climbers
DEM = 1.2      # slide multiplier: promotion odds * (1 + DEM * min(-slope, 2))... see code
BETA = 0.20    # elasticity of promotion odds to current player count (bench only)
QCAP = 0.65    # ceiling on a single bench game's hourly membership
PCAP = 0.90    # maximum named probability mass per place (rest stays unnamed)
NW = 6         # number of trailing reads used for the momentum slope
SLOTS = 72     # hours in the scored window

KNOTS = [6, 18, 30, 42, 54, 66, 78]
BANDS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
         (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 17), (18, 20),
         (21, 25), (26, 30), (31, 38), (39, 45), (46, 60), (61, 97)]


def band_key(rank):
    for lo, hi in BANDS:
        if lo <= rank <= hi:
            return lo
    return 61


# ---- default (prior) membership curves, from the shipped archive offline ----
PRIOR_CURVE = {
    1:  [1, 1, 1, 1, 0.94, 0.97, 0.85],
    2:  [1, 1, 0.982, 0.851, 0.723, 0.575, 0.45],
    3:  [1, 0.968, 0.804, 0.681, 0.596, 0.55, 0.55],
    4:  [1, 0.857, 0.714, 0.553, 0.489, 0.325, 0.4],
    5:  [1, 0.889, 0.75, 0.681, 0.574, 0.525, 0.5],
    6:  [0.956, 0.746, 0.625, 0.553, 0.319, 0.275, 0.25],
    7:  [0.867, 0.556, 0.375, 0.234, 0.149, 0.1, 0.1],
    8:  [0.711, 0.571, 0.536, 0.426, 0.426, 0.275, 0.1],
    9:  [0.622, 0.397, 0.179, 0.128, 0.128, 0.1, 0.1],
    10:  [0.489, 0.333, 0.25, 0.255, 0.128, 0.175, 0.25],
    11:  [0.244, 0.317, 0.304, 0.255, 0.17, 0.05, 0],
    12:  [0.333, 0.302, 0.339, 0.34, 0.362, 0.175, 0.1],
    13:  [0.2, 0.24, 0.26, 0.2, 0.15, 0.09, 0.06],
    14:  [0.06, 0.15, 0.15, 0.15, 0.15, 0.17, 0.17],
    15:  [0.1, 0.13, 0.12, 0.1, 0.1, 0.09, 0.08],
    18:  [0.05, 0.08, 0.07, 0.07, 0.07, 0.08, 0.08],
    21:  [0.02, 0.06, 0.12, 0.11, 0.1, 0.06, 0.03],
    26:  [0.005, 0.015, 0.025, 0.035, 0.035, 0.035, 0.04],
    31:  [0.002, 0.01, 0.03, 0.04, 0.045, 0.05, 0.06],
    39:  [0.001, 0.007, 0.02, 0.03, 0.04, 0.05, 0.06],
    46:  [0.0005, 0.003, 0.01, 0.02, 0.025, 0.03, 0.035],
    61:  [0.0002, 0.002, 0.006, 0.012, 0.02, 0.03, 0.035],
}

# ---- default place kernels: P(place | in top10, from current place/rank) ----
PRIOR_PD_SHORT = {  # current place 1-10 (rows sum to ~0.9; ~0.1 leaves top10 at the far end)
    1:  [0.35, 0.16, 0.09, 0.09, 0.08, 0.12, 0.06, 0.04, 0.03, 0.01],
    2:  [0.04, 0.25, 0.17, 0.04, 0.17, 0.13, 0.05, 0.05, 0.07, 0.02],
    3:  [0.12, 0.15, 0.14, 0.11, 0.17, 0.11, 0.03, 0.04, 0.06, 0.07],
    4:  [0.10, 0.09, 0.14, 0.25, 0.13, 0.13, 0.02, 0.08, 0.01, 0.05],
    5:  [0.06, 0.06, 0.09, 0.27, 0.26, 0.15, 0.03, 0.02, 0.03, 0.02],
    6:  [0.02, 0.03, 0.08, 0.13, 0.11, 0.16, 0.26, 0.06, 0.06, 0.09],
    7:  [0.16, 0.08, 0.07, 0.03, 0.08, 0.06, 0.33, 0.08, 0.01, 0.10],
    8:  [0.02, 0.06, 0.09, 0.08, 0.12, 0.18, 0.03, 0.18, 0.12, 0.13],
    9:  [0.01, 0.02, 0.01, 0.00, 0.00, 0.05, 0.02, 0.29, 0.40, 0.19],
    10: [0.00, 0.02, 0.01, 0.04, 0.03, 0.13, 0.12, 0.22, 0.24, 0.17],
}
PRIOR_BED_SHORT = [0.079, 0.082, 0.054, 0.070, 0.044, 0.079, 0.131, 0.133, 0.159, 0.170]
PRIOR_PD_LONG = {   # from current place (h >= 96h): survives drift to lower places
    1:  [0.00, 0.00, 0.00, 0.00, 0.02, 0.06, 0.10, 0.28, 0.28, 0.26],
    2:  [0.00, 0.00, 0.00, 0.00, 0.02, 0.06, 0.10, 0.30, 0.34, 0.18],
    3:  [0.00, 0.04, 0.02, 0.30, 0.10, 0.10, 0.06, 0.19, 0.19, 0.00],
    4:  [0.00, 0.04, 0.02, 0.14, 0.10, 0.18, 0.06, 0.22, 0.24, 0.00],
    5:  [0.00, 0.12, 0.06, 0.08, 0.06, 0.12, 0.04, 0.22, 0.28, 0.02],
    6:  [0.00, 0.00, 0.00, 0.00, 0.08, 0.10, 0.06, 0.30, 0.36, 0.10],
    7:  [0.00, 0.00, 0.00, 0.00, 0.10, 0.34, 0.22, 0.10, 0.12, 0.12],
    8:  [0.00, 0.00, 0.00, 0.00, 0.02, 0.20, 0.08, 0.10, 0.30, 0.30],
    9:  [0.00, 0.00, 0.00, 0.00, 0.02, 0.06, 0.08, 0.14, 0.35, 0.35],
    10: [0.00, 0.00, 0.00, 0.00, 0.06, 0.08, 0.08, 0.16, 0.31, 0.31],
}
PRIOR_BED_LONG = [0.051, 0.133, 0.146, 0.078, 0.105, 0.195, 0.099, 0.070, 0.041, 0.082]


def parse_reads(path):
    reads = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("ok", True):
                continue
            rows = rec.get("up_and_coming")
            if not rows:
                continue
            try:
                when = datetime.fromisoformat(rec["at"].replace("Z", "+00:00"))
            except (KeyError, ValueError, AttributeError):
                continue
            chart = {}
            for g in rows:
                uid = g.get("universe_id")
                rank = g.get("rank")
                if uid is None or rank is None:
                    continue
                chart[int(uid)] = int(rank)
            if chart:
                reads.append((when, chart, {int(g["universe_id"]): int(g.get("playing") or 0)
                                            for g in rows if g.get("universe_id") is not None}))
    reads.sort(key=lambda x: x[0])
    return reads


def knot_weights(h):
    out = []
    for k in KNOTS:
        half = max(6.0, 0.35 * k)
        w = max(0.0, 1.0 - abs(h - k) / half)
        if w > 0:
            out.append((k, w))
    return out


def norm(vec):
    s = math.fsum(vec)
    if s <= 0:
        return [0.1] * len(vec)
    return [v / s for v in vec]


def estimate(reads):
    """Estimate membership curves and place kernels from the archive itself."""
    n = len(reads)
    wsum = defaultdict(lambda: defaultdict(float))   # band key -> knot -> weight
    hits = defaultdict(lambda: defaultdict(float))   # band key -> knot -> in-top10 weight
    pd_s = defaultdict(lambda: defaultdict(int))     # from place -> to place (h<=78)
    pd_l = defaultdict(lambda: defaultdict(int))     # from place -> to place (h>=96)
    bed_s = defaultdict(int)
    bed_l = defaultdict(int)
    for i in range(n):
        ti, chart_i, _ = reads[i]
        for j in range(i + 1, n):
            tj, chart_j, _ = reads[j]
            h = (tj - ti).total_seconds() / 3600.0
            if h <= 0.5 or h > 300:
                continue
            for uid, ra in chart_i.items():
                rb = chart_j.get(uid)
                key = band_key(ra)
                for k, w in knot_weights(h):
                    wsum[key][k] += w
                    hits[key][k] += w * (1.0 if (rb is not None and rb <= 10) else 0.0)
                if rb is None:
                    continue
                if ra <= 10:
                    if rb <= 10:
                        (pd_s if h <= 78 else pd_l)[ra][rb] += 1
                elif rb <= 10:
                    (bed_s if h <= 78 else bed_l)[rb] += 1
    # blend each band curve with its prior (pseudo-count PRIOR_W)
    prior_w = PRIOR_W
    curves = {}
    for lo, hi in BANDS:
        key = lo
        prior = PRIOR_CURVE[key]
        row = []
        for idx, k in enumerate(KNOTS):
            w = wsum[key].get(k, 0.0)
            hit = hits[key].get(k, 0.0)
            est = hit / w if w > 0 else None
            row.append((est * w + prior[idx] * prior_w) / (w + prior_w) if est is not None else prior[idx])
        curves[key] = [min(1.0, max(0.0, v)) for v in row]
    # kernels: shrink toward priors with pseudo-count 10
    def shrink(table, prior_table, pc=10.0):
        out = {}
        for p in range(1, 11):
            counts = [table[p].get(k, 0) for k in range(1, 11)]
            tot = math.fsum(counts)
            prior = prior_table[p]
            out[p] = norm([(c + pc * q) for c, q in zip(counts, prior)]) if tot + pc > 0 else prior
            s = math.fsum(out[p])
            if s > 1.0:  # row is conditional on staying; allow but cap spread
                out[p] = [v / s for v in out[p]]
        return out
    pd_short = shrink(pd_s, PRIOR_PD_SHORT)
    pd_long = shrink(pd_l, PRIOR_PD_LONG)
    def shrink_vec(counts, prior, pc=10.0):
        tot = math.fsum(counts.values())
        vec = [counts.get(k, 0) + pc * q for k, q in zip(range(1, 11), prior)]
        return norm(vec)
    bed_short = shrink_vec(bed_s, PRIOR_BED_SHORT)
    bed_long = shrink_vec(bed_l, PRIOR_BED_LONG)
    return curves, pd_short, pd_long, bed_short, bed_long


def interp_curve(curve, h):
    if h <= KNOTS[0]:
        return curve[0]
    if h >= KNOTS[-1]:
        r = curve[-1] / max(curve[-2], 1e-9)
        r = min(max(r, 0.5), 0.95)
        return curve[-1] * (r ** ((h - KNOTS[-1]) / 12.0))
    for idx in range(len(KNOTS) - 1):
        if KNOTS[idx] <= h <= KNOTS[idx + 1]:
            t = (h - KNOTS[idx]) / (KNOTS[idx + 1] - KNOTS[idx])
            return curve[idx] + t * (curve[idx + 1] - curve[idx])
    return curve[-1]


def blend_vecs(a, b, t):
    return [x * (1.0 - t) + y * t for x, y in zip(a, b)]


def forecast(reads, task):
    curves, pd_short, pd_long, bed_short, bed_long = estimate(reads)
    t_last = reads[-1][0]
    t0 = task.get("_t0", t_last)
    h0 = max(0.0, (t0 - t_last).total_seconds() / 3600.0)

    last_chart = reads[-1][1]
    last_playing = reads[-1][2]

    # fresh leavers: on the chart until shortly before the last read, then gone.
    # They re-enter at bottom-places at non-trivial rates, so keep them as
    # candidates with a small membership (band 61 curve scaled by recency).
    recent_leavers = {}
    t_fresh = 48.0
    for when, chart, _ in reads[:-1]:
        age = (t_last - when).total_seconds() / 3600.0
        if age > t_fresh:
            continue
        for uid in chart:
            if uid not in last_chart:
                recent_leavers[uid] = min(recent_leavers.get(uid, 0.0), age)
    candidates = dict(last_chart)
    for uid in recent_leavers:
        candidates[uid] = 98  # re-entry band key

    # --- covariates -------------------------------------------------------
    window = reads[-NW:]
    ranks_hist = defaultdict(list)
    for _, chart, _ in window:
        for uid, r in chart.items():
            ranks_hist[uid].append(r)
    t_span = max(1.0, (t_last - window[0][0]).total_seconds() / 3600.0)
    slope = {}
    for uid, rh in ranks_hist.items():
        if len(rh) >= 2:
            slope[uid] = (rh[-1] - rh[0]) / t_span
    play_med = sorted(last_playing.values())[len(last_playing) // 2] if last_playing else 1.0

    # --- accumulate p(game, place) over the scored hours -------------------
    acc = defaultdict(lambda: defaultdict(float))
    for t in range(0, SLOTS):
        h_eff = h0 + t
        t_long = min(1.0, max(0.0, (h_eff - 78.0) / 48.0))
        for uid, rank in candidates.items():
            key = band_key(rank)
            q = interp_curve(curves[key], h_eff)
            if uid in recent_leavers and h_eff >= 24:
                # re-entry: fresh leavers regain a bottom place ~5-10% of the time,
                # ramping in with horizon
                q = max(q, 0.05 * min(1.0, (h_eff - 24.0) / 72.0))
            if q <= 1e-4:
                continue
            sl = slope.get(uid, 0.0)
            if rank <= 10 and uid in last_chart:
                # slide demotion for games sliding down; promotion damping for risers
                if sl < 0:
                    q = min(0.97, q * (1.0 + DEM * min(-sl, 2.0)))
                elif sl > 0:
                    q = q / (1.0 + DEM * min(sl, 2.0))
            elif uid in last_chart:
                pl = last_playing.get(uid, play_med) or play_med
                q *= max(0.5, min(1.6, (max(pl, 1.0) / max(play_med, 1.0)) ** BETA))
                if sl < 0:
                    q = min(QCAP, q * (1.0 + MOM * min(-sl, 3.0)))
            if q <= 1e-4:
                continue
            if rank <= 10 and uid in last_chart:
                kern = pd_short[rank]
            else:
                kern = bed_short
            for k in range(1, 11):
                p = q * kern[k - 1]
                if p > 1e-6:
                    acc[uid][k] += p

    prob = {k: {} for k in range(1, 11)}
    for uid, d in acc.items():
        for k, p in d.items():
            prob[k][uid] = p / SLOTS
    for k in range(1, 11):
        d = prob[k]
        tot = math.fsum(d.values())
        if tot > PCAP:
            f = PCAP / tot
            for u in d:
                d[u] *= f

    # prune: naming a game below the unnamed share is strictly worse than not
    # naming it (it would otherwise receive that share for free), so keep a
    # row only while its p beats the per-game share of the unnamed remainder.
    n_chart = max(10, len(last_chart))
    for k in range(1, 11):
        d = prob[k]
        kept = {}
        for uid, p in sorted(d.items(), key=lambda x: (-x[1], x[0])):
            named = math.fsum(kept.values())
            share = (1.0 - min(named, 1.0)) / max(1, n_chart - len(kept))
            if p <= share or len(kept) >= 100:
                break
            kept[uid] = p
        prob[k] = kept

    score = {uid: math.fsum((11 - k) * prob[k].get(uid, 0.0) for k in range(1, 11))
             for uid in acc}
    top10 = sorted(score, key=lambda u: (-score[u], u))[:10]
    return prob, top10


def main():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8"))
    reads = parse_reads(os.path.join(INPUT_DIR, "history.ndjson"))
    if not reads:
        raise SystemExit("no usable reads in history.ndjson")
    window = task.get("window", {}) or {}
    t0 = window.get("from") or window.get("t0")
    if t0:
        task["_t0"] = datetime.fromisoformat(t0.replace("Z", "+00:00"))
    places = int(task.get("places", 10))
    prob, top10 = forecast(reads, task)

    rows = []
    for k in range(1, places + 1):
        for uid, p in sorted(prob[k].items(), key=lambda x: (-x[1], x[0]))[:100]:
            if p > 1e-6:
                rows.append({"universe_id": int(uid), "place": k, "p": round(p, 6)})
    payload = {"probabilities": rows, "top10": [int(u) for u in top10[:places]]}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
        fh.write("\n")
    print("wrote %s (%d probability rows, %d top10)" % (out_path, len(rows), len(top10)))


if __name__ == "__main__":
    main()
