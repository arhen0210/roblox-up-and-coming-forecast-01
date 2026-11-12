#!/usr/bin/env python3
"""Faithful scorer-side backtest for roblox-up-and-coming-forecast-01 (dev tool).

For every archive read c with at least 3 later reads inside the following
72 hours, this builds the grader's input exactly (history up to and including
read c, plus a task.json whose scored window starts at read c), runs the
forecast, then scores it against the reads that actually happened using the
competition formula including the unnamed-share rule:

    for each read, for each place k = 1..10:
        p_k = the row's p for the game actually at place k, if named there;
              else p_k = (1 - named mass at place k) / (number of games on
              that read's chart that were not named at place k)
        read_score = sum over the 10 places of ln(max(p_k, 0.0001))
        log_score  = mean of read_score over the reads that exist

IMPORTANT: the archive's read cadence is ~6 h, not 1 h.  This backtest
therefore under-weights how much the real chart persists hour to hour and
reads as far more diffuse than the server.  Its number is useful for
ranking variants; the SERVER score is the authority for the level of
concentration (see README).

Modes:
  (default)          score main.forecast on every cut with shipped constants
  --search           sweep the named constants (persist, neighbour, alpha,
                     entry_mass, exit_mod) over a grid via overrides
  --persist/--neighbour/--neighbour2/--alpha/--entry-mass/--exit-mass
                     score one parameter set
  --validate PATH    check a written forecast.json against the output schema

Usage:
  MAKEFUN_INPUT_DIR=<archive> python3 backtest.py [--search]
"""

import argparse
import json
import math
import os
import sys
from datetime import timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main as M  # noqa: E402

ARCHIVE = os.environ.get(
    "MAKEFUN_INPUT_DIR", os.path.abspath(os.path.join(HERE, "..", "rbx_input")))
WINDOW_H = 72.0
MIN_LATER = 3
LN_FLOOR = math.log(1e-4)


def at_rank(chart, k):
    for uid, r in chart.items():
        if r == k:
            return uid
    return None


def build_cuts(reads):
    cuts = []
    for c, entry in enumerate(reads):
        tc = entry[0]
        later = []
        for rec in reads[c + 1:]:
            h = (rec[0] - tc).total_seconds() / 3600.0
            if h <= 0 or h > WINDOW_H:
                continue
            later.append((rec[0], rec[1]))
        if len(later) >= MIN_LATER:
            cuts.append((c, tc, later))
    return cuts


def make_task(template, tc):
    task = dict(template)
    w = dict(template.get("window") or {})
    w["from"] = tc.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    w["until"] = (tc + timedelta(hours=WINDOW_H)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    task["window"] = w
    task["history_until"] = w["from"]
    task["_t0"] = tc
    return task


class Tally:
    def __init__(self):
        self.pair_scores = []
        self.per_place = [[] for _ in range(11)]
        self.hits = 0
        self.reads = 0
        self.agree = 0
        self.places = 0
        self.mass = []

    def add(self, prob, top10, later):
        named = {k: math.fsum(prob[k].values()) for k in range(1, 11)}
        nsets = {k: frozenset(prob[k]) for k in range(1, 11)}
        top = {int(u) for u in list(top10)[:10]}
        for _t, ch in later:
            ch_set = frozenset(ch)
            self.reads += 1
            self.hits += len(top & {u for u, r in ch.items() if r <= 10})
            rs = 0.0
            for k in range(1, 11):
                u = at_rank(ch, k)
                if u is None:
                    continue
                self.places += 1
                p = prob[k].get(u)
                if p is None:
                    m = len(ch_set - nsets[k])
                    p = (1.0 - min(named[k], 1.0)) / m if m > 0 else 0.0
                lp = math.log(p if p > 1e-4 else 1e-4)
                rs += lp
                self.per_place[k].append(lp)
                if u in nsets[k]:
                    self.agree += 1
            self.pair_scores.append(rs)
        for k in range(1, 11):
            self.mass.append((k, len(prob[k]), round(named[k], 3)))

    def mean(self):
        n = len(self.pair_scores)
        return math.fsum(self.pair_scores) / n if n else float("nan")

    def report(self, label=""):
        per = []
        for k in range(1, 11):
            v = self.per_place[k]
            per.append(math.fsum(v) / len(v) if v else float("nan"))
        print("%-58s log=%8.3f  hits/read=%.2f  agree=%.3f  reads=%d" % (
            label, self.mean(),
            self.hits / self.reads if self.reads else 0.0,
            self.agree / self.places if self.places else 0.0,
            self.reads))
        print("  per-place mean ln(p): " + " ".join(
            "%d:%6.2f" % (k, per[k - 1]) for k in range(1, 11)))
        rows = [r for r in self.mass if r[0] == 1]
        if rows:
            print("  per-place (rows,mass) at cut0: " + " ".join(
                "%d=(%d,%.2f)" % r for r in self.mass[:10]))
        return self.mean()


def run_forecast_mode(reads, template, cuts, overrides=None):
    tally = Tally()
    for c, _tc, later in cuts:
        task = make_task(template, reads[c][0])
        prob, top10 = M.forecast(reads[:c + 1], task, overrides)
        tally.add(prob, top10, later)
    return tally


def run_params(raws, params):
    tally = Tally()
    for raw, later in raws:
        prob, top10 = M.forecast(raw, {}, params)
        tally.add(prob, top10, later)
    return tally


GRID_PERSIST = [0.30, 0.40, 0.55, 0.65, 0.75]
GRID_NEIGH = [0.06, 0.12, 0.18]
GRID_ALPHA = [1.0, 1.25, 1.5, 1.75, 2.0]
GRID_ENTRY = [0.03, 0.05, 0.08, 0.10]
GRID_EXIT = [0.0, 0.10, 0.20]


def search(raws):
    """Coordinate descent: sweep one axis at a time from the shipped start."""
    best = dict(persist=M.PERSIST_AT_SAME_PLACE, neighbour=M.NEIGHBOUR_W,
                alpha=M.TEMPER_ALPHA, entry_mass=M.ENTRY_MASS,
                exit_mod=M.EXIT_MOD)
    cur = run_params(raws, best).mean()
    print("start %s -> log=%8.3f" % (best, cur))

    def eval_params(p):
        return run_params(raws, p).mean()

    for _round in range(2):
        for axis, grid in [("persist", GRID_PERSIST), ("neighbour", GRID_NEIGH),
                           ("alpha", GRID_ALPHA), ("entry_mass", GRID_ENTRY),
                           ("exit_mod", GRID_EXIT)]:
            results = []
            for v in grid:
                trial = dict(best)
                trial[axis] = v
                m = eval_params(trial)
                results.append((m, v))
                print("  %s=%-6s -> log=%8.3f%s" % (
                    axis, v, m, "   (current)" if v == best[axis] else ""))
                sys.stdout.flush()
            results.sort(key=lambda x: (-x[0], x[1]))
            m, v = results[0]
            if m > eval_params(best) + 1e-9:
                best[axis] = v
                print("  keep %s=%s (log=%8.3f)" % (axis, v, m))
            else:
                print("  keep %s=%s (no gain)" % (axis, best[axis]))
    t = run_params(raws, best)
    t.report("final %s" % best)
    return t.mean(), best


def validate_output(path):
    d = json.load(open(path, "r", encoding="utf-8"))
    assert set(d) <= {"probabilities", "top10"}, "unexpected keys: %s" % sorted(d)
    rows = d["probabilities"]
    top10 = d["top10"]
    assert isinstance(rows, list) and isinstance(top10, list)
    assert len(top10) == 10, "top10 must have 10 games, got %d" % len(top10)
    assert len(set(top10)) == 10, "top10 games must be distinct"
    seen = set()
    per_place = {}
    rows_per_place = {}
    for r in rows:
        assert set(r) == {"universe_id", "place", "p"}, "bad row keys: %s" % sorted(r)
        uid, k, p = r["universe_id"], r["place"], r["p"]
        assert isinstance(uid, int) and isinstance(k, int) and 1 <= k <= 10
        assert isinstance(p, (int, float)) and 0.0 <= p <= 1.0, "p out of range: %r" % p
        key = (uid, k)
        assert key not in seen, "duplicate (game, place): %r" % (key,)
        seen.add(key)
        per_place[k] = per_place.get(k, 0.0) + p
        rows_per_place[k] = rows_per_place.get(k, 0) + 1
        assert rows_per_place[k] <= M.MAX_ROWS_PER_PLACE, \
            "place %d has %d rows > MAX_ROWS_PER_PLACE" % (k, rows_per_place[k])
    for k, s in sorted(per_place.items()):
        assert s <= 1.0, "place %d mass %.4f > 1" % (k, s)
    print("validate %s: OK (%d rows, %d places, rows/place %s, mass %s)" % (
        path, len(rows), len(per_place),
        " ".join("%d=%d" % (k, rows_per_place[k]) for k in sorted(rows_per_place)),
        " ".join("%d=%.3f" % (k, per_place[k]) for k in sorted(per_place))))
    for k in range(1, 11):
        assert k in rows_per_place, "place %d missing entirely" % k


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--persist", type=float)
    ap.add_argument("--neighbour", type=float)
    ap.add_argument("--neighbour2", type=float)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--entry-mass", dest="entry_mass", type=float)
    ap.add_argument("--exit-mass", dest="exit_mod", type=float)
    ap.add_argument("--validate", metavar="PATH")
    args = ap.parse_args()

    if args.validate:
        validate_output(args.validate)
        return

    reads = M.parse_reads(os.path.join(ARCHIVE, "history.ndjson"))
    template = json.load(open(os.path.join(ARCHIVE, "task.json"), "r", encoding="utf-8"))
    cuts = build_cuts(reads)
    print("archive: %d reads, %d cuts, %d (cut,read) pairs" % (
        len(reads), len(cuts), sum(len(x[2]) for x in cuts)))

    if args.search:
        raws = [(reads[:c + 1], later) for c, _tc, later in cuts]
        search(raws)
    elif any(v is not None for v in (args.persist, args.neighbour, args.neighbour2,
                                     args.alpha, args.entry_mass, args.exit_mod)):
        params = {}
        if args.persist is not None:
            params["persist"] = args.persist
        if args.neighbour is not None:
            params["neighbour"] = args.neighbour
        if args.neighbour2 is not None:
            params["neighbour2"] = args.neighbour2
        if args.alpha is not None:
            params["alpha"] = args.alpha
        if args.entry_mass is not None:
            params["entry_mass"] = args.entry_mass
        if args.exit_mod is not None:
            params["exit_mod"] = args.exit_mod
        run_params([(reads[:c + 1], later) for c, _tc, later in cuts],
                   params).report("params %s" % params)
    else:
        run_forecast_mode(reads, template, cuts).report(
            "main.forecast (shipped constants)")


if __name__ == "__main__":
    main()
