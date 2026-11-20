"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts each GTA VI trophy's Exophase earn rate at the case's target as five
quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Model — earners / tracked decomposition, fitted as logistics
------------------------------------------------------------
A trophy's rate is r = 100 * E / N, where E is the number of tracked players
who earned it and N the number of players Exophase tracks.  The case history
gives both (rates and players_tracked per read), so the program forecasts E
and N separately and recombines them.  This matters because the two move
differently: at launch N explodes (new players flood in) while E grows
steadily, so rates first FALL (dilution) and only later rise as earning
outpaces joining.  A model that extrapolates the rate itself (like the house
curve baseline) cannot see that turn; the E/N split can.

Both E(t) and N(t) are adoption curves: they rise fast, then saturate.  The
program fits each as a logistic in log space,

    E(t) = E_inf / (1 + exp(-k (t - t0))),

by grid search over (k, t0) with E_inf solved in closed form, on the case's
history subsampled to <= 32 points.  A logistic saturates, so unlike a
log-linear trend it cannot run away over the 4.5-day horizon.  A trophy
whose own earners barely moved gets no usable logistic of its own: its
median is then a 50/50 blend (in logit space) of the tier-group trend and
persistence, with a wider spread.

  * N(t): logistic fit, anchored at the fitted value at the last read; N
    never forecast below its last value.
  * E_i(t): logistic fit per trophy; E never forecast below its last value,
    nor above a tier saturation cap times N_target.
  * Story-chain monotonicity: earners can only fall along the chapter chain
    (Ch1 >= Ch2 >= Ch3 >= Ch4 >= final mission) and the platinum needs every
    trophy, so its earners are capped by the smallest other trophy's.
  * Spread (logit): the two fits' residuals plus a term for extrapolation
    distance plus half the starter's random-walk floor.

Launch prior (flat history, first 24 h)
----------------------------------------
On launch night the house reads are flat (rates and N identical for hours),
yet the rates then fall ~1.5 logits within half a day as N explodes.  Nothing
in a flat case shows that, so the program carries the launch-influx prior,
fitted on the real launch-night -> 2026-11-19 11:39 UTC Exophase snapshot
(55 trophies; N 638 -> ~10,000):

  * N(t) ~ (1 + t/2h)^2.74
  * E_i(t) ~ (1 + t/2h)^b_i,  b_i = 1.240 + 0.723 * ln(r_i_last)

The b_i regression is tight (residual std 0.174 in exponent units): trophies
that already had more early earners grow their earners faster.  The prior
fires only when the history is (near-)flat AND the case sits within 24 h of
release; a rich history (the final case) never triggers it.

The 100% trophy (id 2, "Complete Gone Fishing") is forecast as
[99.0, 99.8, 100, 100, 100]: the page shows 100.00% at ~10,000 tracked
players, so the median stays 100, but a small lower tail hedges the chance
that a flood of brand-new tracked players dips it below 100 by the target.
An all-100 row would pay up to ~4.6 logits on one trophy if it dips to 99%.

Deterministic: pure functions of the case file; no network, no clock, no
randomness.  Runs in well under the 8 s / 1 CPU / 1024 MB limits.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only.  Outside the grader's container,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of a rate's move, in logit units per square-root day (the starter's).
SIGMA = 0.5

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- Launch-influx prior (flat history, first 24 h after release) -------------
# Fitted on the real launch-night -> 2026-11-19 11:39 UTC Exophase snapshot.
PRIOR_TAU_H = 2.0      # time transform (1 + t / TAU_H)
PRIOR_ALPHA_N = 2.74   # N(t) ~ (1 + t/TAU_H)^ALPHA_N
PRIOR_B0 = 1.240       # E_i(t) ~ (1 + t/TAU_H)^b_i, b_i = B0 + B1 * ln(r_i_last)
PRIOR_B1 = 0.723
PRIOR_EXTRA_SIGMA = 0.45  # extra logit spread: the prior is one realisation

# Tier saturation caps: earners never forecast above cap * N_target (fractions).
TIER_CAP = {"platinum": 0.02, "gold": 0.05, "silver": 0.30, "bronze": 0.90}
# Story chain (chapter order) and the trophies that need the whole game.
CHAIN_IDS = [3, 4, 5, 6, 7]  # Ch1, Ch2, Ch3, Ch4, final mission
PLATINUM_ID = 1
COMPLETION_ID = 14  # Attain 100% Completion

# Global spread multiplier for the logistic / no-signal branches (calibrated
# on synthetic final-regime backtests; 1.0 = honest as fitted).  The launch
# prior branch keeps its own calibrated spread and is NOT multiplied.
SPREAD_MULT = 1.2
# Median blend (logistic branch only): the fitted trend's median is shrunk
# toward persistence in logit space.  A 4.5-day extrapolation from 2.5 days
# of history is weakly identified for late-blooming trophies, so the trend
# is trusted only in part; the launch prior branch (flat history) is NOT
# blended — the dilution there is near-certain.  Calibrated on backtests.
MEDIAN_BLEND = 0.4


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    return 100 / (1 + math.exp(-lg))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(s: str) -> float:
    return (parse_time(s) - RELEASE).total_seconds() / 3600.0


def ols(xs, ys):
    """Least-squares line y = a + b x.  Returns (a, b, rms_residual)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, resid


def _logistic_log(t, k, t0):
    """log(1 / (1 + exp(-k (t - t0)))), clamped against underflow."""
    z = k * (t - t0)
    if z >= 0:
        return -math.log1p(math.exp(-z))
    return z - math.log1p(math.exp(z))


def fit_logistic(ts, ys):
    """Fit y = log_inf + log(g(t)) with g a logistic, by grid search over
    (k, t0) with log_inf solved in closed form.  ts: hours; ys: log values.
    Returns (log_inf, k, t0, rms_residual)."""
    # Subsample to at most 32 points, evenly spaced, always keeping the ends.
    if len(ts) > 32:
        idx = sorted(set(int(round(i * (len(ts) - 1) / 31.0)) for i in range(32)))
        ts = [ts[i] for i in idx]
        ys = [ys[i] for i in idx]
    best = None

    def eval_grid(ks, t0s):
        nonlocal best
        for k in ks:
            for t0 in t0s:
                lg = [_logistic_log(t, k, t0) for t in ts]
                log_inf = sum(y - g for y, g in zip(ys, lg)) / len(ys)
                resid = math.sqrt(sum((y - (log_inf + g)) ** 2 for y, g in zip(ys, lg)) / len(ys))
                if best is None or resid < best[3]:
                    best = (log_inf, k, t0, resid)

    ks_coarse = [0.015, 0.025, 0.04, 0.065, 0.10, 0.16, 0.25, 0.40, 0.60]
    t0s_coarse = [float(v) for v in range(0, 201, 20)]
    eval_grid(ks_coarse, t0s_coarse)
    # Refine around the best (k, t0).
    _, bk, bt0, _ = best
    ks_fine = [bk * f for f in (0.7, 0.85, 1.0, 1.18, 1.4)]
    t0s_fine = [bt0 + d for d in (-10, -5, -2, 0, 2, 5, 10)]
    eval_grid(ks_fine, t0s_fine)
    return best


def r_last_for(i, hist):
    for r in reversed(hist):
        if r["rates"][i] is not None:
            return r["rates"][i]
    return None


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    hist = case["history"]
    at_h = hours_since_release(case["at"])
    target_h = hours_since_release(case["target"])
    horizon_days = case["horizon_minutes"] / 1440
    rw = SIGMA * math.sqrt(horizon_days)

    # --- time axis and N series -------------------------------------------
    times = [hours_since_release(r["at"]) for r in hist]
    ns = [float(r["players_tracked"]) for r in hist]
    n_last = ns[-1]
    t_last = times[-1]

    # Is the history (near-)flat?  Max log-E movement across trophies.
    max_log_move = 0.0
    for i in range(len(trophies)):
        es = []
        for r in hist:
            rate = r["rates"][i]
            if rate is not None:
                es.append(rate * float(r["players_tracked"]) / 100.0)
        if len(es) >= 2 and min(es) > 0:
            max_log_move = max(max_log_move, math.log(max(es) / min(es)))
    flat = max_log_move < 0.02
    early = at_h < 24.0

    # --- N forecast (logistic, unless the launch prior fires) --------------
    if flat and early:
        n_target = n_last * ((1 + target_h / PRIOR_TAU_H) / (1 + t_last / PRIOR_TAU_H)) ** PRIOR_ALPHA_N
        resid_n = 0.0
    else:
        log_inf_n, k_n, t0_n, resid_n = fit_logistic(times, [math.log(max(n, 1.0)) for n in ns])
        n_fit_last = math.exp(log_inf_n + _logistic_log(t_last, k_n, t0_n))
        n_target = math.exp(log_inf_n + _logistic_log(target_h, k_n, t0_n))
        n_target = max(n_target, n_last, n_fit_last * 0.999)
    n_target = max(n_target, n_last)

    # --- per-trophy E forecasts -------------------------------------------
    # Recent-window log-log slopes (for the tier-group fallback) and the
    # movement (signal) per trophy.
    win = [j for j, t in enumerate(times) if t >= at_h - 24.0]
    if len(win) < 6:
        win = list(range(len(hist)))
    group_slopes = {}
    move = {}
    for i, tr in enumerate(trophies):
        pts = []
        for j in win:
            rate = hist[j]["rates"][i]
            if rate is None:
                continue
            pts.append((math.log1p(times[j]), math.log(max(rate * ns[j] / 100.0, 0.05))))
        if len(pts) >= 3 and len(set(p[1] for p in pts)) >= 2:
            _, b, _ = ols([p[0] for p in pts], [p[1] for p in pts])
            move[i] = max(p[1] for p in pts) - min(p[1] for p in pts)
            group_slopes.setdefault(tr["tier"], []).append((b, move[i]))
        else:
            move[i] = 0.0
    group_mean = {}
    for tier, vals in group_slopes.items():
        known = [b for b, m in vals if m > 0.5]
        if not known:
            known = [b for b, m in vals]
        group_mean[tier] = sum(known) / len(known) if known else 0.0

    e_target = {}
    sigma = {}
    do_blend = {}
    for i, tr in enumerate(trophies):
        # Last observed earners for this trophy.
        e_last = None
        r_last = None
        for r in reversed(hist):
            if r["rates"][i] is not None:
                r_last = r["rates"][i]
                e_last = max(r_last * float(r["players_tracked"]) / 100.0, 0.05)
                break
        if e_last is None:  # trophy never appeared in any read
            e_last, r_last = 0.05, 0.01
        prior_branch = False
        if flat and early and r_last < 100:
            # Launch prior branch (own calibrated spread; not multiplied).
            b_e = PRIOR_B0 + PRIOR_B1 * math.log(max(r_last, 0.01))
            growth = ((1 + target_h / PRIOR_TAU_H) / (1 + t_last / PRIOR_TAU_H)) ** b_e
            e_t = e_last * growth
            sig = math.sqrt(rw * rw + PRIOR_EXTRA_SIGMA * PRIOR_EXTRA_SIGMA)
            prior_branch = True
        elif move[i] > 0.5:
            # Logistic fit on this trophy's earners.
            ts_e, ys_e = [], []
            for j, r in enumerate(hist):
                rate = r["rates"][i]
                if rate is None:
                    continue
                ts_e.append(times[j])
                ys_e.append(math.log(max(rate * ns[j] / 100.0, 0.05)))
            log_inf_e, k_e, t0_e, resid_e = fit_logistic(ts_e, ys_e)
            e_fit_last = math.exp(log_inf_e + _logistic_log(t_last, k_e, t0_e))
            e_t = math.exp(log_inf_e + _logistic_log(target_h, k_e, t0_e))
            e_t = max(e_t, e_last, e_fit_last * 0.999)
            extrap = abs(math.log(max(e_t, 0.05) / max(e_fit_last, 0.05)))
            sig = math.sqrt((0.5 * rw) ** 2 + resid_e ** 2 + resid_n ** 2 + (0.15 * extrap) ** 2)
            do_blend[i] = True
        else:
            # No signal of its own: borrow the tier trend, shrunk halfway
            # toward persistence, with a wide spread.
            b_g = group_mean.get(tr["tier"], 0.0)
            b_g = min(max(b_g, 0.0), 4.0)
            growth = math.exp(b_g * (math.log1p(target_h) - math.log1p(t_last)))
            growth = min(growth, 50.0)
            e_trend = e_last * growth
            m_trend = 100 * min(e_trend, TIER_CAP.get(tr["tier"], 0.90) * n_target) / n_target
            m_persist = r_last
            m = inv_logit(0.5 * logit(m_trend) + 0.5 * logit(m_persist))
            e_t = m / 100 * n_target
            sig = math.sqrt(rw * rw + 0.8 * 0.8)
        # Caps: earners never fall, never exceed the tier saturation cap.
        cap = TIER_CAP.get(tr["tier"], 0.90) * n_target
        e_t = min(max(e_t, e_last), cap)
        e_target[i] = e_t
        sigma[i] = sig if prior_branch else SPREAD_MULT * sig
        do_blend.setdefault(i, False)

    # --- story-chain / platinum monotonicity on earners --------------------
    by_id = {tr["id"]: idx for idx, tr in enumerate(trophies)}
    if all(k in by_id for k in CHAIN_IDS):
        run = math.inf
        for tid in CHAIN_IDS:
            idx = by_id[tid]
            run = min(run, e_target[idx])
            e_target[idx] = run
    if PLATINUM_ID in by_id and len(trophies) > 1:
        idx = by_id[PLATINUM_ID]
        others = min(e_target[j] for j in range(len(trophies)) if j != idx)
        e_target[idx] = min(e_target[idx], others)
    if COMPLETION_ID in by_id and all(k in by_id for k in CHAIN_IDS):
        idx = by_id[COMPLETION_ID]
        chain_min = min(e_target[by_id[k]] for k in CHAIN_IDS)
        e_target[idx] = min(e_target[idx], chain_min)

    # --- quantiles ----------------------------------------------------------
    rows = []
    for i, tr in enumerate(trophies):
        if r_last_for(i, hist) == 100:
            # The 100% trophy: median stays 100, small lower tail hedges a dip.
            rows.append({"id": tr["id"], "q": [99.0, 99.8, 100, 100, 100]})
            continue
        m = 100 * e_target[i] / n_target
        if do_blend.get(i):
            # Shrink the trend median toward persistence (last rate).
            rl = r_last_for(i, hist)
            if rl is not None and rl < 100:
                m = inv_logit(MEDIAN_BLEND * logit(m) + (1 - MEDIAN_BLEND) * logit(rl))
        lg = logit(m)
        s = sigma[i]
        qs = [round4(inv_logit(lg + z * s)) for z in Z]
        qs = [min(100, max(0.01, q)) for q in qs]
        # Keep non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
