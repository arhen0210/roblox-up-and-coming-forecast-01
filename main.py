#!/usr/bin/env python3
"""roblox-up-and-coming-forecast-01: persistence-backed concentrated forecast.

The server leaderboard is the authority on concentration: a -13 log score
over 10 places means a geometric mean p of ~0.27 per place, which is only
reachable by putting large probability on a mostly-stable chart.  The chart
is computed from user growth and, over the scored hourly window, it persists
far more than it churns.  So persistence is the backbone and churn (entry
and exit) is a small correction.

Model (fully offline, stdlib only, deterministic; MAKEFUN_TRIAL_SEED is read
for interface compatibility but the model uses no RNG):

1. Parse /input/history.ndjson (one chart read per line) and /input/task.json;
   anchor every place on the LAST mark of the archive.
2. Backbone: for each place k, the game holding place k in the last mark gets
   PERSIST_AT_SAME_PLACE; the games at rank distance 1 (places k-1 / k+1, and
   bench rank 11 next to place 10) get NEIGHBOUR_W each; rank distance 2 gets
   NEIGHBOUR2_W; every other game on the last mark gets the thin FLOOR_W.
3. Exit correction (at most EXIT_MOD of the backbone): the holder's weight is
   scaled down by the empirically measured top-10 survival of its place
   (place 1 members survive ~1.00, place 10 members ~0.49 over <= 12 h).
4. Entry correction (at most ENTRY_MASS of the mass): distributed over bench
   games and fresh leavers by an empirical entry-rate table (rank band x
   rank-slope x player-growth, shrunk) times an empirical landing-place
   table, both fitted at runtime from the input archive itself.
5. Temper up: p ** TEMPER_ALPHA per place, renormalised to
   NAMED_MASS_TARGET, at most MAX_ROWS_PER_PLACE rows per place, and the
   last mark's holder of the place is always present and is the largest row.
6. top10 = the ten games with the highest place-weighted expected score.

The dev backtest (backtest.py) scores this against the archive's ~6-hour
read cadence, which underweights hourly persistence; the SERVER score is the
authority for the level of concentration (see README).

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

# ---- backbone: persistence + displacement kernel over rank distance -------
PERSIST_AT_SAME_PLACE = 0.55   # weight of the last mark's holder of place k
NEIGHBOUR_W = 0.12            # rank distance 1 (places k-1 / k+1, rank 11)
NEIGHBOUR2_W = 0.04           # rank distance 2 (places k-2 / k+2)
FLOOR_W = 0.0005              # thin floor for every other chart game
TEMPER_ALPHA = 1.5            # temper up: p ** alpha, then renormalise
MAX_ROWS_PER_PLACE = 20       # emitted rows per place (holder always kept)

# ---- emission --------------------------------------------------------------
NAMED_MASS_TARGET = 0.98      # per-place named mass; rest = unnamed share
P_ROUND = 6                   # decimals kept in the emitted p

# ---- entry/exit correction budget (kept <= 10% of the mass) ----------------
ENTRY_MASS = 0.08             # total entry-correction mass per place
ENTRY_MASS_CAP = 0.10         # hard cap on the entry correction
EXIT_MOD = 0.10               # max fractional cut of the holder's weight

# ---- empirical estimation knobs (all fitted from the input archive) --------
ENTRY_H = 48.0                # horizon for entry-rate estimation
SURV_H = 12.0                 # horizon for per-place survival estimation
T_FRESH = 48.0                # leaver horizon: off the chart < 48 h ago
NW = 5                        # trailing reads for momentum / growth
CLIMB_SHRINK = 20.0           # pseudo-counts toward the band base rate
LAND_SHRINK = 12.0            # pseudo-counts toward the landing prior
SURV_SHRINK = 10.0            # pseudo-counts toward the survival prior
SURV_PRIOR = 0.90
LEAVER_KEY = 99               # pseudo band key for fresh leavers
BENCH_BANDS = [(11, 14), (15, 17), (18, 20), (21, 25), (26, 30), (31, 38),
               (39, 45), (46, 60), (61, 97)]
SLOPE_BUCKETS = [(-1e9, -2.0), (-2.0, -0.5), (-0.5, 0.5), (0.5, 1e9)]
GROWTH_BUCKETS = [(-1e9, -0.25), (-0.25, 0.05), (0.05, 0.30), (0.30, 1e9)]
LAND_PRIOR = [0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.17, 0.20, 0.30]


def band_key(rank):
    for lo, hi in BENCH_BANDS:
        if lo <= rank <= hi:
            return lo
    return 61


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


def bucket_index(value, buckets):
    for idx, (lo, hi) in enumerate(buckets):
        if lo <= value < hi:
            return idx
    return len(buckets) - 1


def trailing_covariates(reads):
    """Rank slope (per read step) and relative player growth over the last
    NW reads: negative slope = rank number falling = climbing toward top10."""
    window = reads[-NW:]
    ranks_hist = defaultdict(list)
    plays_hist = defaultdict(list)
    for _, chart, playing in window:
        for uid, r in chart.items():
            ranks_hist[uid].append(r)
            plays_hist[uid].append(playing.get(uid, 0) or 0)
    slope = {}
    growth = {}
    for uid in sorted(ranks_hist):
        rh = ranks_hist[uid]
        ph = plays_hist[uid]
        if len(rh) >= 2:
            slope[uid] = (rh[-1] - rh[0]) / (len(rh) - 1)
            growth[uid] = ((ph[-1] - ph[0]) / (len(ph) - 1) / ph[0]) if ph[0] > 0 else None
        else:
            slope[uid] = 0.0
            growth[uid] = None
    return slope, growth


def entry_tables(reads):
    """Empirical tables fitted from the archive itself.

    Returns (rates, land, surv, leaver_rate):
      rates: (band, slope bucket, growth bucket) -> smoothed absolute entry
             rate into the top 10 within ENTRY_H hours;
      land:  band -> [P(lands at place k | entered)] for k = 1..10, shrunk
             toward LAND_PRIOR; LEAVER_KEY holds the fresh-leaver row;
      surv:  place -> smoothed P(a place-k member is still in the top 10
             after <= SURV_H hours);
      leaver_rate: smoothed re-entry rate of fresh leavers.
    """
    n = len(reads)
    trials = defaultdict(float)
    entries = defaultdict(float)
    base_t = defaultdict(float)
    base_e = defaultdict(float)
    land = defaultdict(lambda: [0.0] * 10)
    surv = [[0.0, 0.0] for _ in range(11)]   # [hits, trials] per place
    leav_t = 0.0
    leav_e = 0.0
    for i in range(n):
        ti, chart_i, _ = reads[i]
        slope, growth = trailing_covariates(reads[:i + 1])
        recent = set()
        for rec in reads[max(0, i - 8):i]:
            if (ti - rec[0]).total_seconds() <= T_FRESH * 3600:
                recent.update(rec[1])
        left = {u for u in recent if u not in chart_i}
        for j in range(i + 1, n):
            tj, chart_j, _ = reads[j]
            h = (tj - ti).total_seconds() / 3600.0
            if h <= 0:
                continue
            if h <= SURV_H:
                for u, r in chart_i.items():
                    if r <= 10:
                        surv[r][1] += 1.0
                        if chart_j.get(u, 99) <= 10:
                            surv[r][0] += 1.0
            if h > ENTRY_H:
                break
            for u, r in chart_i.items():
                if r <= 10:
                    continue
                b = band_key(r)
                sb = bucket_index(slope.get(u, 0.0), SLOPE_BUCKETS)
                g = growth.get(u)
                gb = 1 if g is None else bucket_index(g, GROWTH_BUCKETS)
                ent = 1.0 if chart_j.get(u, 99) <= 10 else 0.0
                key = (b, sb, gb)
                trials[key] += 1.0
                entries[key] += ent
                base_t[b] += 1.0
                base_e[b] += ent
                if ent:
                    land[b][chart_j[u] - 1] += 1.0
            for u in left:
                ent = 1.0 if chart_j.get(u, 99) <= 10 else 0.0
                leav_t += 1.0
                leav_e += ent
                if ent:
                    land[LEAVER_KEY][chart_j[u] - 1] += 1.0
    # smoothed absolute entry rates
    rates = {}
    band_base = {b: (base_e[b] + 1.0) / (base_t[b] + 20.0) for b in base_t}
    for b in [lo for lo, _ in BENCH_BANDS]:
        base = band_base.get(b, 0.05)
        for sb in range(len(SLOPE_BUCKETS)):
            for gb in range(len(GROWTH_BUCKETS) + 1):
                t = trials.get((b, sb, gb), 0.0)
                e = entries.get((b, sb, gb), 0.0)
                rates[(b, sb, gb)] = (e + CLIMB_SHRINK * base) / (t + CLIMB_SHRINK)
    # landing distributions
    land_out = {}
    keys = [lo for lo, _ in BENCH_BANDS] + [LEAVER_KEY]
    for b in keys:
        row = land[b]
        tot = math.fsum(row)
        land_out[b] = [(c + LAND_SHRINK * q) / (tot + LAND_SHRINK * math.fsum(LAND_PRIOR))
                       for c, q in zip(row, LAND_PRIOR)]
    # survival per place
    surv_out = {}
    for k in range(1, 11):
        hits, tot = surv[k]
        surv_out[k] = (hits + SURV_SHRINK * SURV_PRIOR) / (tot + SURV_SHRINK)
    leaver_rate = (leav_e + 1.0) / (leav_t + 20.0)
    return rates, land_out, surv_out, leaver_rate


def kernel_weight(rank, k, persist, neigh, neigh2, floor):
    d = rank - k
    ad = abs(d)
    if ad == 0:
        return persist
    if ad == 1:
        return neigh
    if ad == 2:
        return neigh2
    return floor


def forecast(reads, task, overrides=None):
    """Build the persistence-backed per-place distributions.

    Returns (prob, top10): prob[k] maps game -> emitted probability for
    place k; top10 is the list of ten distinct game ids.
    overrides lets the backtest vary the named constants without edits.
    """
    ov = dict(overrides or {})
    persist = float(ov.pop("persist", PERSIST_AT_SAME_PLACE))
    neigh = float(ov.pop("neighbour", NEIGHBOUR_W))
    neigh2 = float(ov.pop("neighbour2", NEIGHBOUR2_W))
    floor = float(ov.pop("floor", FLOOR_W))
    alpha = float(ov.pop("alpha", TEMPER_ALPHA))
    max_rows = int(ov.pop("max_rows", MAX_ROWS_PER_PLACE))
    entry_mass = min(float(ov.pop("entry_mass", ENTRY_MASS)), ENTRY_MASS_CAP)
    exit_mod = float(ov.pop("exit_mod", EXIT_MOD))
    if ov:
        raise ValueError("unknown overrides: %r" % sorted(ov))

    rates, land, surv, leaver_rate = entry_tables(reads)
    t_last, chart, playing = reads[-1]

    # holders of places 1..10 in the last mark (first by id if rank repeats)
    holders = {}
    for u in sorted(chart):
        r = chart[u]
        if 1 <= r <= 10 and r not in holders:
            holders[r] = u

    # fresh leavers: on the chart within T_FRESH before the last read, gone now
    recent = set()
    for rec in reads[:-1]:
        if (t_last - rec[0]).total_seconds() <= T_FRESH * 3600:
            recent.update(rec[1])
    leavers = {u for u in recent if u not in chart}

    slope, growth = trailing_covariates(reads)

    # --- backbone + corrections, per place ---------------------------------
    raw = {k: {} for k in range(1, 11)}
    entry_w = {k: {} for k in range(1, 11)}
    for u in sorted(chart):
        r = chart[u]
        for k in range(1, 11):
            w = kernel_weight(r, k, persist, neigh, neigh2, floor)
            if u == holders.get(k):
                # exit correction: churn at the bottom places is real
                w *= 1.0 - exit_mod * (1.0 - surv[k])
            raw[k][u] = w
            if r > 10:
                b = band_key(r)
                sb = bucket_index(slope.get(u, 0.0), SLOPE_BUCKETS)
                g = growth.get(u)
                gb = 1 if g is None else bucket_index(g, GROWTH_BUCKETS)
                entry_w[k][u] = rates.get((b, sb, gb), 0.05) * land[b][k - 1]
    for u in sorted(leavers):
        for k in range(1, 11):
            entry_w[k][u] = leaver_rate * land[LEAVER_KEY][k - 1]
    for k in range(1, 11):
        s = math.fsum(entry_w[k].values())
        if s > 0:
            f = entry_mass / s
            for u in entry_w[k]:
                raw[k][u] = raw[k].get(u, 0.0) + entry_w[k][u] * f

    # --- temper up, truncate, renormalise ----------------------------------
    prob = {}
    for k in range(1, 11):
        d = raw[k]
        tot = math.fsum(d.values())
        if tot <= 0:
            prob[k] = {}
            continue
        if alpha != 1.0:
            powed = {u: (p / tot) ** alpha for u, p in d.items()}
            ptot = math.fsum(powed.values())
            d = {u: p / ptot for u, p in powed.items()}
        else:
            d = {u: p / tot for u, p in d.items()}
        # keep at most max_rows rows, largest first
        kept = {}
        for u, p in sorted(d.items(), key=lambda x: (-x[1], x[0])):
            if len(kept) >= max_rows:
                break
            kept[u] = p
        # the last mark's holder of this place is always present and largest
        h = holders.get(k)
        if h is not None:
            if h not in kept:
                if kept:
                    worst = min(kept, key=lambda u: (kept[u], u))
                    del kept[worst]
                kept[h] = 0.0
            mx = max((p for u, p in kept.items() if u != h), default=0.0)
            if kept[h] <= mx:
                kept[h] = mx * 1.000001
        m = math.fsum(kept.values())
        if m > 0:
            f = NAMED_MASS_TARGET / m
            kept = {u: p * f for u, p in kept.items()}
        # round, then keep the per-place mass <= 1 after rounding
        rows = {u: round(p, P_ROUND) for u, p in kept.items() if p > 1e-9}
        s = math.fsum(rows.values())
        if s > 1.0 and rows:
            big = max(rows, key=lambda u: (rows[u], -u))
            rows[big] = round(rows[big] - (s - 1.0), P_ROUND)
        prob[k] = rows

    score = {}
    for u in set().union(*[set(prob[k]) for k in range(1, 11)]):
        score[u] = math.fsum((11 - k) * prob[k].get(u, 0.0) for k in range(1, 11))
    top10 = sorted(score, key=lambda u: (-score[u], u))[:10]
    return prob, top10


def main():
    task = json.load(open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8"))
    reads = parse_reads(os.path.join(INPUT_DIR, "history.ndjson"))
    if not reads:
        raise SystemExit("no usable reads in history.ndjson")
    places = int(task.get("places", 10))
    prob, top10 = forecast(reads, task)

    rows = []
    for k in range(1, places + 1):
        for u, p in sorted(prob.get(k, {}).items(), key=lambda x: (-x[1], x[0])):
            if p > 1e-9:
                rows.append({"universe_id": int(u), "place": k, "p": round(p, P_ROUND)})
    payload = {"probabilities": rows, "top10": [int(u) for u in top10[:places]]}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
        fh.write("\n")
    print("wrote %s (%d probability rows, %d top10)" % (out_path, len(rows), len(top10[:places])))


if __name__ == "__main__":
    main()
