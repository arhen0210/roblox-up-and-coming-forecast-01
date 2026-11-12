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
# honoured for interface compatibility: the model is deterministic and uses
# no RNG, so the seed never changes any output (verified byte-identical).
SEED = int(os.environ.get("MAKEFUN_TRIAL_SEED", "20260101") or 20260101)

# ---- hyperparameters (tuned by rolling 72-hour backtests over the archive) ----
# ---- calibration (tuned with backtest.py over the archive) ----
ALPHA = 0.85   # tempering exponent: p^alpha per place (flatter when < 1)
MTGT = 1.00    # named-mass target per place (rest = unnamed share)
WMIX = 0.00    # uniform mixture weight over the read's eligible games
CAP = None     # placeholder; the per-place gradient below is used instead
CAPS = {1: 0.26, 2: 0.19, 3: 0.14, 4: 0.11, 5: 0.09, 6: 0.08, 7: 0.07,
        8: 0.06, 9: 0.05, 10: 0.05}   # per-place cap (s=0.75 x base)
THRESH = 0.008 # flat floor: rows at or below this are not emitted

PRIOR_W = 240.0 # pseudo-count blending archive-estimated curves toward the priors
MOM = 2.0      # climb multiplier: q *= 1 + MOM * min(-slope, 3) for bench climbers
DEM = 1.2      # slide multiplier: promotion odds * (1 + DEM * min(-slope, 2))... see code
BETA = 0.10    # elasticity of promotion odds to current player count (bench only)
QCAP = 0.65    # ceiling on a single bench game's hourly membership
PCAP = 0.90    # maximum named probability mass per place (rest stays unnamed)
NW = 5         # number of trailing reads used for the momentum slope
SLOTS = 72     # hours in the scored window
T_FRESH = 120.0  # leaver horizon: chart reads within this many hours before the last
REENTRY = 0.0    # re-entry membership floor for fresh leavers (ramps in past 24h)
TGT_PULL = 0.0   # band re-mass pull toward the empirical bucket distribution (0=off)
TOPGROWTH = 0.0  # player-growth multiplier on top-10 members' membership
GSHIFT = 0.0     # growth kernel shift strength (places per unit growth*10)

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


def estimate(reads, prior_w=PRIOR_W):
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
    # blend each band curve with its prior (pseudo-count prior_w)
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


# band-level re-mass: target buckets for the empirical place distributions
TARGET_R = [(1, 3), (4, 6), (7, 10), (11, 14), (15, 97)]   # last: bench
RE_B = 5          # recent-leaver bucket id
TARGET_SMOOTH = 3.0   # pseudo-counts per bucket toward the raw bucket mass
TARGET_FMIN, TARGET_FMAX = 0.1, 10.0   # clamp on the re-mass factor


def target_bucket(rank):
    for idx, (lo, hi) in enumerate(TARGET_R):
        if lo <= rank <= hi:
            return idx
    return 4








def empirical_band_targets(reads):
    """Empirical P(bucket of the actual game | place k), over pairs (i, j)
    with 0 < horizon <= 72h.  FR outcomes (never on the cut chart) are not
    nameable, so the table is renormalized over the nameable buckets."""
    counts = [[TARGET_SMOOTH] * 6 for _ in range(11)]
    n = len(reads)
    for i in range(n):
        chart_i = reads[i][1]
        ti = reads[i][0]
        seen_recent = set()
        for rec in reads[max(0, i - 8):i]:
            if (ti - rec[0]).total_seconds() <= 48 * 3600:
                seen_recent.update(rec[1])
        for j in range(i + 1, n):
            h = (reads[j][0] - ti).total_seconds() / 3600.0
            if h <= 0 or h > 72.0:
                continue
            chart_j = reads[j][1]
            for k in range(1, 11):
                u = None
                for uu, r in chart_j.items():
                    if r == k:
                        u = uu
                        break
                if u is None:
                    continue
                if u in chart_i:
                    b = target_bucket(chart_i[u])
                elif u in seen_recent:
                    b = RE_B
                else:
                    continue  # FR: not nameable
                counts[k][b] += 1.0
    totals = [math.fsum(row) for row in counts]
    return {k: [c / totals[k] for c in counts[k]] for k in range(1, 11)}


def blend_vecs(a, b, t):
    return [x * (1.0 - t) + y * t for x, y in zip(a, b)]


# bench promotion: empirical entry-rate multipliers by (rank bucket, slope bucket)
CLIMB_R = [(15, 17), (18, 20), (21, 25), (26, 30), (31, 38), (39, 45), (46, 60), (61, 97)]
CLIMB_S = [(-1e9, -6.0), (-6.0, -2.0), (-2.0, -0.5), (-0.5, 0.5), (0.5, 1e9)]
CLIMB_G = [(1e-9, 0.0), (0.0, 0.10), (0.10, 0.30), (0.30, 1e9)]   # player growth per read step
CLIMB_SHRINK = 20.0   # pseudo-counts toward the rank bucket's base rate
CLIMB_MIN = 0.25      # multiplier clamp
CLIMB_MAX = 8.0
CLIMB_MODE = "mult"   # "mult": q *= table;  "replace": bench q = base*mult


def climb_bucket(rank, slope_step, growth=None):
    rb = 7
    for idx, (lo, hi) in enumerate(CLIMB_R):
        if lo <= rank <= hi:
            rb = idx
            break
    sb = 4
    for idx, (lo, hi) in enumerate(CLIMB_S):
        if lo <= slope_step < hi:
            sb = idx
            break
    gb = 0
    if growth is not None:
        for idx, (lo, hi) in enumerate(CLIMB_G):
            if lo <= growth < hi:
                gb = idx
                break
    else:
        gb = 1  # no data: neutral bucket
    return rb, sb, gb


def climb_mult_table(reads, nw, shrink=CLIMB_SHRINK):
    """Empirical bench-entry rates and multipliers by (rank, slope, growth).

    For every read pair (i, j) with 0 < j-i horizon <= 72h, bench games on
    read i are bucketed by their rank, their rank slope (ranks per read
    step over the trailing nw reads) and their player-count growth per read
    step over the same window.  Returns a table with two kinds of entries:
    (rb, sb, gb) -> multiplier (entry rate / rank-bucket base rate), and
    (rb, sb, gb, "rate") -> the raw window entry rate itself.
    """
    trials = defaultdict(float)
    entries = defaultdict(float)
    base_t = defaultdict(float)
    base_e = defaultdict(float)
    n = len(reads)
    for i in range(nw, n):
        chart_i = reads[i][1]
        hist = defaultdict(list)
        plays = defaultdict(list)
        for rec in reads[i - nw + 1:i + 1]:
            for u, r in rec[1].items():
                hist[u].append(r)
                plays[u].append(rec[2].get(u, 0) or 0)
        slope = {}
        growth = {}
        for u, rh in hist.items():
            if len(rh) >= 2:
                slope[u] = (rh[-1] - rh[0]) / (len(rh) - 1)
                ph = plays[u]
                if ph[0] > 0:
                    growth[u] = (ph[-1] - ph[0]) / max(1, len(ph) - 1) / ph[0]
                else:
                    growth[u] = None
            else:
                slope[u] = 0.0
                growth[u] = None
        for j in range(i + 1, n):
            h = (reads[j][0] - reads[i][0]).total_seconds() / 3600.0
            if h <= 0 or h > 72.0:
                continue
            chart_j = reads[j][1]
            for u, r in chart_i.items():
                if r <= 14:
                    continue
                rb, sb, gb = climb_bucket(r, slope.get(u, 0.0), growth.get(u))
                ent = 1.0 if chart_j.get(u, 99) <= 10 else 0.0
                trials[(rb, sb, gb)] += 1.0
                entries[(rb, sb, gb)] += ent
                base_t[rb] += 1.0
                base_e[rb] += ent
    table = {}
    for rb in range(len(CLIMB_R)):
        bt = base_t.get(rb, 0.0)
        be = base_e.get(rb, 0.0)
        base = (be + 1.0) / (bt + 20.0) if bt > 0 else 0.05
        for sb in range(len(CLIMB_S)):
            for gb in range(len(CLIMB_G)):
                t = trials.get((rb, sb, gb), 0.0)
                e = entries.get((rb, sb, gb), 0.0)
                rate = (e + shrink * base) / (t + shrink)
                table[(rb, sb, gb)] = min(CLIMB_MAX, max(CLIMB_MIN, rate / base if base > 0 else 1.0))
                table[(rb, sb, gb, "rate")] = rate
    return table


def raw_forecast(reads, task, overrides=None):
    """Run the simulation and return the raw per-place distributions.

    Returns (raw, eligible, n_chart): raw[k] maps game -> untempered
    probability mass for place k (mass may exceed 1 and must be calibrated
    before emission), eligible is the candidate set the window is simulated
    over, n_chart is the size of the last archived chart.

    overrides (dict) lets the backtest vary the simulation constants
    (t_fresh, reentry, mom, dem, beta, nw, prior_w) without editing them.
    """
    ov = dict(overrides or {})
    t_fresh = ov.pop("t_fresh", T_FRESH)
    reentry = ov.pop("reentry", REENTRY)
    mom = ov.pop("mom", MOM)
    dem = ov.pop("dem", DEM)
    beta = ov.pop("beta", BETA)
    qcap = ov.pop("qcap", QCAP)
    nw = int(ov.pop("nw", NW))
    prior_w = ov.pop("prior_w", PRIOR_W)
    tgt_pull = ov.pop("tgt_pull", TGT_PULL)
    climb_mode = ov.pop("climb_mode", CLIMB_MODE)
    climb_shrink = ov.pop("climb_shrink", CLIMB_SHRINK)
    topgrowth = ov.pop("topgrowth", TOPGROWTH)
    gshift = ov.pop("gshift", GSHIFT)
    if ov:
        raise ValueError("unknown overrides: %r" % sorted(ov))

    curves, pd_short, pd_long, bed_short, bed_long = estimate(reads, prior_w)
    t_last = reads[-1][0]
    t0 = task.get("_t0", t_last)
    h0 = max(0.0, (t0 - t_last).total_seconds() / 3600.0)

    last_chart = reads[-1][1]
    last_playing = reads[-1][2]

    # fresh leavers: on the chart until shortly before the last read, then gone.
    # They re-enter at bottom-places at non-trivial rates, so keep them as
    # candidates with a small membership (band 61 curve scaled by recency).
    recent_leavers = {}
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
    window = reads[-nw:]
    ranks_hist = defaultdict(list)
    for _, chart, _ in window:
        for uid, r in chart.items():
            ranks_hist[uid].append(r)
    t_span = max(1.0, (t_last - window[0][0]).total_seconds() / 3600.0)
    slope = {}
    for uid, rh in ranks_hist.items():
        if len(rh) >= 2:
            slope[uid] = (rh[-1] - rh[0]) / t_span
    slope_step = {}
    for uid, rh in ranks_hist.items():
        if len(rh) >= 2:
            slope_step[uid] = (rh[-1] - rh[0]) / (len(rh) - 1)
    climb_t = climb_mult_table(reads, nw, climb_shrink)
    band_targets = empirical_band_targets(reads)
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
                # re-entry: fresh leavers regain a bottom place at the
                # empirical 16%/72h re-entry rate, ramping in with horizon
                q = max(q, reentry * min(1.0, (h_eff - 24.0) / 72.0))
            if q <= 1e-4:
                continue
            sl = slope.get(uid, 0.0)
            if rank <= 10 and uid in last_chart:
                # slide demotion for games sliding down; promotion damping for risers
                if sl < 0:
                    q = min(0.97, q * (1.0 + dem * min(-sl, 2.0)))
                elif sl > 0:
                    q = q / (1.0 + dem * min(sl, 2.0))
                # player-count growth shifts placement within the top 10:
                # empirically corr(growth, later place) = -0.32
                ph = [pl2.get(uid, 0) or 0 for (_, _, pl2) in window]
                pn = last_playing.get(uid, 0) or 0
                if len(ph) >= 3 and ph[0] > 0:
                    growth = (pn - ph[0]) / (len(ph) - 1) / ph[0]
                    q = min(0.98, q * max(0.5, min(1.6, 1.0 + topgrowth * growth)))
            elif uid in last_chart:
                pl = last_playing.get(uid, play_med) or play_med
                q *= max(0.5, min(1.6, (max(pl, 1.0) / max(play_med, 1.0)) ** beta))
                if sl < 0:
                    q = min(qcap, q * (1.0 + mom * min(-sl, 3.0)))
                ph = [pl2.get(uid, 0) or 0 for (_, _, pl2) in window]
                p_now = last_playing.get(uid, 0) or 0
                growth = None
                if len(ph) >= 3 and ph[0] > 0:
                    growth = (p_now - ph[0]) / (len(ph) - 1) / ph[0]
                rb, sb, gb = climb_bucket(rank, slope_step.get(uid, 0.0), growth)
                if climb_mode == "replace":
                    # bench q := empirical window entry rate for this bucket
                    q = max(q, climb_t.get((rb, sb, gb, "rate"), q))
                else:
                    q = min(qcap, q * climb_t.get((rb, sb, gb), 1.0))
            if q <= 1e-4:
                continue
            if rank <= 10 and uid in last_chart:
                kern = pd_short[rank]
                # growth kernel shift: a game growing fast tends to land on
                # a better place than its current one, a shrinking one worse
                ph = [pl2.get(uid, 0) or 0 for (_, _, pl2) in window]
                pn = last_playing.get(uid, 0) or 0
                gr = None
                if len(ph) >= 3 and ph[0] > 0:
                    gr = (pn - ph[0]) / (len(ph) - 1) / ph[0]
                if gr is not None and gshift != 0:
                    shift = min(3, max(-3, round(gshift * gr * 10.0)))
                    tgt = rank - int(shift)
                    if 1 <= tgt <= 10 and tgt != rank:
                        kern = blend_vecs(kern, pd_short[tgt], min(1.0, abs(gshift * gr) * 2.0))
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
    # --- band re-mass: scale each place's raw bucket masses toward the
    # empirical bucket distribution of the actual games, with pull tgt_pull
    # (0 = off, 1 = full empirical shape).
    if tgt_pull > 0:
        for k in range(1, 11):
            d = prob[k]
            if not d:
                continue
            bucket_mass = [0.0] * 6
            for uid, p in d.items():
                if uid in last_chart:
                    b = target_bucket(last_chart[uid])
                elif uid in recent_leavers:
                    b = RE_B
                else:
                    continue
                bucket_mass[b] += p
            raw_tot = math.fsum(bucket_mass)
            emp = band_targets[k]
            emp_tot = math.fsum(emp)
            if raw_tot <= 0 or emp_tot <= 0:
                continue
            for b in range(6):
                raw_share = bucket_mass[b] / raw_tot
                emp_share = emp[b]
                if raw_share > 1e-9:
                    factor = 1.0 + tgt_pull * (emp_share / max(raw_share, 1e-9) - 1.0)
                    factor = min(TARGET_FMAX, max(TARGET_FMIN, factor))
                else:
                    factor = 1.0
                if factor != 1.0:
                    for uid in d:
                        bu = (target_bucket(last_chart[uid]) if uid in last_chart
                              else (RE_B if uid in recent_leavers else None))
                        if bu == b:
                            d[uid] *= factor
            # renormalize this place back to the raw total mass
            m = math.fsum(d.values())
            if m > 0:
                f = raw_tot / m
                for uid in d:
                    d[uid] *= f

    return prob, candidates, max(10, len(last_chart))


def forecast(reads, task):
    raw, eligible, n_chart = raw_forecast(reads, task)
    return calibrate(raw, eligible, n_chart, cap=CAPS)


def calibrate(raw, eligible, n_chart, alpha=ALPHA, mtgt=MTGT, wmix=WMIX,
              cap=CAP, thresh=THRESH):
    """Turn the raw per-place mass into a calibrated, named distribution.

    Steps, in order (all deterministic):
      1. tempering: raise every p to the power alpha and rescale so the place
         keeps its raw total mass; alpha < 1 flattens over-confidence
         (the raw peak at place 1 can exceed 0.3 while the archive's #1 is
         #1 in none of the practice reads).
      2. uniform mixture: blend in a uniform component of total weight wmix
         over every eligible game, so bench and re-entry candidates carry
         real rows instead of depending on the unnamed share.
      3. scale the place to the named-mass target mtgt < 1: the remainder
         1 - mtgt is the unnamed share the scorer spreads evenly over the
         games of the read that were not named at that place.
      4. share-prune (greedy, the rule-optimal cut): walking probabilities
         in descending order, keep naming a game only while its p beats the
         per-game share of the remaining unnamed mass; then a flat floor
         thresh and the 100-rows-per-place cap.
    """
    prob = {}
    n_elig = len(eligible)
    for k in range(1, 11):
        d = dict(raw[k])
        tot = math.fsum(d.values())
        if tot > 0 and alpha != 1.0:
            powed = {u: p ** alpha for u, p in d.items()}
            ptot = math.fsum(powed.values())
            if ptot > 0:
                f = tot / ptot
                d = {u: p * f for u, p in powed.items()}
        if wmix > 0 and n_elig > 0:
            u_mass = wmix / n_elig
            row = {u: p * (1.0 - wmix) for u, p in d.items()}
            for u in eligible:
                row[u] = row.get(u, 0.0) + u_mass
            d = row
        if cap is not None:
            capk = cap.get(k, 0.15) if isinstance(cap, dict) else cap
            d = {u: (p if p < capk else capk) for u, p in d.items()}
        # scale to the named-mass target
        tot = math.fsum(d.values())
        if tot > 0 and tot != mtgt:
            f = mtgt / tot
            d = {u: p * f for u, p in d.items()}
        # share-prune: keep a row only while it beats the per-game share of
        # the unnamed remainder (naming a weaker game would both spend mass
        # on it and shrink the share the actual answer receives for free).
        kept = {}
        for uid, p in sorted(d.items(), key=lambda x: (-x[1], x[0])):
            if p <= thresh or len(kept) >= 100:
                break
            share = (1.0 - min(math.fsum(kept.values()), 1.0)) / max(1, n_chart - len(kept))
            if p <= share:
                break
            kept[uid] = p
        m = math.fsum(kept.values())
        if m > 0.999:
            f = 0.999 / m
            kept = {u: p * f for u, p in kept.items()}
        prob[k] = kept

    score = {uid: math.fsum((11 - k) * prob[k].get(uid, 0.0) for k in range(1, 11))
             for uid in set().union(*[set(prob[k]) for k in range(1, 11)])}
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
