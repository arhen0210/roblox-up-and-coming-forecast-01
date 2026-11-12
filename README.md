# roblox-up-and-coming-forecast-01

Persistence-backed, concentrated forecast of Roblox's Up-and-Coming chart
(top 10 by place) for the 72-hour scored window.

The server leaderboard is the authority on concentration: a -13 log score
over 10 places implies a geometric mean p of ~0.27 per place, reachable only
by putting large probability on a mostly-stable chart.  The chart is
computed from user growth and, over the hourly scored window, it persists
far more than it churns (measured on the archive: same-place rate 0.54 at
2-4 h gaps, 0.44 at 4-8 h, decaying to ~0.06 beyond 36 h).  So the model
makes persistence the backbone and churn a small correction:

- backbone: the last mark's holder of place k gets PERSIST_AT_SAME_PLACE =
  0.55; rank-distance 1 (places k-1 / k+1, and bench rank 11 beside place
  10) gets NEIGHBOUR_W = 0.12 each; rank-distance 2 gets NEIGHBOUR2_W =
  0.04; every other chart game gets a thin FLOOR_W = 0.0005;
- exit correction: the holder's weight is scaled by the empirically measured
  top-10 survival of its place (place 1 ~1.00, place 10 ~0.49 over <= 12 h),
  capped at a 10% cut;
- entry correction: <= 8% of the mass spread over bench games and fresh
  leavers by an empirical entry-rate table (rank band x rank-slope x
  player-growth, shrunk) times an empirical landing-place table, both fitted
  at runtime from the input archive itself;
- tempering: p ** TEMPER_ALPHA (1.5) per place, renormalised;
- emission: at most MAX_ROWS_PER_PLACE = 20 rows per place, named mass
  0.98, and the last mark's holder of the place is always present and is
  the largest row.

`backtest.py` (scorer-faithful rolling 72-hour backtest over the archive,
31 cuts, 295 read pairs) reads this as far MORE diffuse than the server
does, because the archive's read cadence is ~6 h, not 1 h: it punishes
exactly the concentration the leaderboard rewards.  Its number is kept for
comparison (dev backtest: -54.8 under this model; the pre-calibration
diffuse model scored -35.4 there but -36.0 on the server at rank 526/870)
— the SERVER score is the authority for the level of concentration.

Fully offline, standard library only, deterministic (byte-identical across
runs and independent of MAKEFUN_TRIAL_SEED — no RNG in the model).
Run: MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<out> python3 main.py
Dev: MAKEFUN_INPUT_DIR=<input> python3 backtest.py [--search|--persist ..]
