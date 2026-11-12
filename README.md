# roblox-up-and-coming-forecast-01

Forecast of Roblox's Up-and-Coming chart (top 10 by place) for the 72-hour
scored window, built as a rank-transition model with entry and exit:
per-rank membership curves q(rank, h) = P(top-10 membership h hours on),
estimated from every read pair in history.ndjson and shrunk toward priors
fitted offline on the same archive; place kernels for in-top10 placement;
momentum, player-count and fresh-leaver re-entry covariates per game.
Named mass per place is capped and low-probability rows are pruned so the
unnamed remainder carries the churn the chart shows (half of the top 10
turns over in 48 h). Fully offline, standard library only, deterministic.
Run: MAKEFUN_INPUT_DIR=<input> MAKEFUN_OUTPUT_DIR=<out> python3 main.py
