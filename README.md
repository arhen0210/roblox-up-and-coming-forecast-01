# roblox-up-and-coming-forecast-01

Forecast of Roblox's Up-and-Coming chart (top 10 by place) for the 72-hour
scored window, built as a rank-transition model with entry and exit:
per-rank membership curves q(rank, h) = P(top-10 membership h hours on),
estimated from every read pair in history.ndjson and shrunk toward priors
fitted offline on the same archive; place kernels for in-top10 placement;
momentum, player-count and player-growth covariates per game; an empirical
bench-climb table (rank bucket x rank-slope bucket x growth bucket) fitted
at runtime from the input archive itself.

Emission is calibrated against `backtest.py`, a scorer-faithful rolling
72-hour backtest over the archive (31 cuts, 295 read pairs):

- tempering: p^alpha (alpha = 0.85) per place, preserving place mass;
- per-place cap gradient (0.26 at place 1 down to 0.05 at place 10);
- named-mass target 1.0 with greedy share-pruning: a row is emitted only
  while its p beats the per-game share of the remaining unnamed mass, so
  the tail is carried by real rows, not the unnamed share;
- flat floor 0.008 and the 100-rows-per-place limit.

Backtest mean log score: -35.37 (was -36.14 before calibration).
Archive reference points: in-sample MLE rank->place transition table = -34.99
(the practical ceiling of a rank-only model on this data).

Fully offline, standard library only, deterministic (byte-identical across
runs; MAKEFUN_TRIAL_SEED is read but unused — no RNG in the model).
Run: MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<out> python3 main.py
Dev: MAKEFUN_INPUT_DIR=<input> python3 backtest.py [--search|--alpha ..]
