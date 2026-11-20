"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target time.

Model, in one paragraph: the page only changes when Exophase refreshes it, so
the recorded history is first compressed into its distinct states. A trophy's
earn rate is earners / players_tracked, and the two parts are forecast
separately: the tracked base follows a damped power-law growth fit, and each
trophy's earner count follows its own power-law fit in log time with its
slope shrunk toward the pooled cross-trophy slope. The composed rate is
blended in logit space with a direct logit-trend extrapolation of the rate
itself. When the history holds a single frozen state (a refresh is overdue,
as on launch night), the forecast instead prices the coming refresh: the
tracked base multiplies while earner counts barely move, so rates dilute.
Quantiles are normal on the logit scale around the median, with a spread
that grows with the horizon, the fit residuals and the refresh risk.

Runs offline on the Python 3.13 standard library only, deterministically.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ZVALS = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
EPS = 0.01
TOP = 99.99


def clamp_rate(x):
    return min(TOP, max(EPS, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(v):
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    if text is None:
        return None
    try:
        dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def wls_slope(xs, ys, ws):
    """Weighted least-squares slope of ys on xs; None when it is unidentified."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    return sxy / sxx


def distinct_states(history):
    """Collapse consecutive identical reads: the page is frozen between refreshes."""
    states = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        when = parse_time(read.get("slot") or read.get("at"))
        if states and states[-1]["key"] == key:
            states[-1]["last_seen"] = when
            continue
        states.append({
            "key": key,
            "at": when,
            "last_seen": when,
            "players": read.get("players_tracked") or 0,
            "earned": read.get("trophies_earned") or 0,
            "rates": list(read.get("rates") or []),
        })
    return states


def last_known(rates_seq, idx):
    for rates in reversed(rates_seq):
        if idx < len(rates) and rates[idx] is not None:
            return rates[idx]
    return None


def stale_refresh_forecast(states, horizon_hours, release_ts, checkpoint_ts):
    """One frozen state: price the refresh that has not happened yet.

    Returns (player_growth, earner_growth): the tracked base multiplies by
    player_growth at the next refresh while a typical trophy's earner count
    only grows by earner_growth, so rates dilute by roughly their ratio.
    """
    players = max(states[-1]["players"], 1)
    age_h = 0.0
    if release_ts is not None and checkpoint_ts is not None:
        age_h = max(0.0, (checkpoint_ts - release_ts) / 3600.0)
    # A small base right after launch is pre-flood: the real player base
    # arrives at the first refreshes. The flood factor fades with game age.
    flood = 8.0 * math.exp(-age_h / 72.0) if players < 20000 else 0.0
    growth = 1.0 + flood * (1.0 - math.exp(-max(horizon_hours, 0.0) / 2.0))
    growth = min(growth, 12.0)
    earner_growth = 1.0 + (growth - 1.0) * 0.085
    return growth, earner_growth


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    horizon_min = float(case.get("horizon_minutes") or 0.0)
    horizon_days = horizon_min / 1440.0
    horizon_hours = horizon_min / 60.0

    states = distinct_states(history)
    t0 = states[0]["at"] if states[0]["at"] is not None else 0.0
    checkpoint_ts = parse_time(case.get("at"))
    target_ts = parse_time(case.get("target"))
    if target_ts is None and checkpoint_ts is not None:
        target_ts = checkpoint_ts + horizon_min * 60.0
    release_ts = parse_time((case.get("game") or {}).get("release_date"))

    def x_of(ts):
        if ts is None or t0 is None:
            return 0.0
        return math.log1p(max(0.0, (ts - t0) / 3600.0))

    xs = [x_of(s["at"]) for s in states]
    x_last = xs[-1]
    x_target = x_of(target_ts) if target_ts is not None else x_last + math.log1p(horizon_hours)
    dx = max(0.0, x_target - x_last)
    k = len(states)
    recency = [0.85 ** (k - 1 - j) for j in range(k)]

    # Slopes are fitted on the recent regime only (the last 36 hours of
    # distinct states): the launch surge's growth rates do not persist,
    # and extrapolating them for days overshoots badly. Levels still
    # anchor on the very last state.
    if k >= 4 and checkpoint_ts is not None:
        win = [j for j in range(k)
               if states[j]["at"] is not None and states[j]["at"] >= checkpoint_ts - 36 * 3600]
        if len(win) >= 3:
            keep = set(win)
            states = [s for j, s in enumerate(states) if j in keep]
            xs = [x for j, x in enumerate(xs) if j in keep]
            recency = [0.85 ** (len(states) - 1 - j) for j in range(len(states))]
            k = len(states)

    last_rates = [last_known([s["rates"] for s in states], i) for i in range(n)]
    p_last = max(float(states[-1]["players"] or 0), 1.0)

    # ---- tracked-base forecast -------------------------------------------
    if k >= 2:
        slope_p = wls_slope(xs, [math.log(max(float(s["players"] or 1), 1.0)) for s in states], recency)
        if slope_p is None:
            slope_p = 0.0
        slope_p = min(max(slope_p, 0.0), 3.0)
        p_target = p_last * math.exp(0.8 * slope_p * dx)
        p_target = min(max(p_target, p_last * 1.0), p_last * 10.0)
        stale = False
    else:
        growth, earner_growth = stale_refresh_forecast(states, horizon_hours, release_ts, checkpoint_ts)
        p_target = p_last * growth
        stale = True

    # ---- per-trophy earner and logit-trend slopes ------------------------
    earners = []
    for s in states:
        row = []
        for i in range(n):
            r = s["rates"][i] if i < len(s["rates"]) else None
            row.append(None if r is None else max(r, 0.0) / 100.0 * float(s["players"] or 0))
        earners.append(row)

    pooled_e = []
    pooled_l = []
    slopes_e = []
    slopes_l = []
    decels = []
    resid_l = []
    for i in range(n):
        pts_e = [(xs[j], math.log(max(earners[j][i], 0.4)), recency[j])
                 for j in range(k) if earners[j][i] is not None]
        if len(pts_e) >= 2:
            s_e = wls_slope([p[0] for p in pts_e], [p[1] for p in pts_e], [p[2] for p in pts_e])
        else:
            s_e = None
        slopes_e.append(s_e)
        if s_e is not None:
            pooled_e.append(s_e)
        pts_l = [(xs[j], logit(states[j]["rates"][i]), recency[j])
                 for j in range(k)
                 if i < len(states[j]["rates"]) and states[j]["rates"][i] is not None
                 and 0.0 < states[j]["rates"][i] < 100.0]
        decel = 1.0
        if len(pts_l) >= 4:
            half = len(pts_l) // 2
            s_a = wls_slope([p[0] for p in pts_l[:half]], [p[1] for p in pts_l[:half]],
                            [p[2] for p in pts_l[:half]])
            s_b = wls_slope([p[0] for p in pts_l[half:]], [p[1] for p in pts_l[half:]],
                            [p[2] for p in pts_l[half:]])
            if s_a is not None and s_b is not None and abs(s_a) > 0.05 and s_a * s_b > 0:
                decel = min(max(abs(s_b) / abs(s_a), 0.3), 1.15)
        decels.append(decel)
        if len(pts_l) >= 2:
            s_l = wls_slope([p[0] for p in pts_l], [p[1] for p in pts_l], [p[2] for p in pts_l])
            if s_l is not None and len(pts_l) >= 3:
                mx = sum(p[0] for p in pts_l) / len(pts_l)
                my = sum(p[1] for p in pts_l) / len(pts_l)
                for px, py, _ in pts_l:
                    resid_l.append(py - (my + s_l * (px - mx)))
        else:
            s_l = None
        slopes_l.append(s_l)
        if s_l is not None:
            pooled_l.append(s_l)

    def pooled(vals, fallback):
        if not vals:
            return fallback
        vals = sorted(vals)
        return vals[len(vals) // 2]

    pool_e = pooled(pooled_e, 0.6)
    pool_l = pooled(pooled_l, 0.0)
    resid_sd = 0.0
    if len(resid_l) >= 8:
        mean_r = sum(resid_l) / len(resid_l)
        resid_sd = math.sqrt(sum((v - mean_r) ** 2 for v in resid_l) / len(resid_l))
    resid_sd = min(resid_sd, 1.2)

    base_spread = 0.5 * math.sqrt(max(horizon_days, 0.0))

    medians = []
    sigmas = []
    for i in range(n):
        r_last = last_rates[i]
        if r_last is None:
            r_last = 1.0
        e_last = max(r_last, 0.0) / 100.0 * p_last
        if stale:
            if r_last >= 99.9:
                e_pred = e_last * (p_target / p_last)
            else:
                # Newly tracked players earn early content first: story
                # trophies gain earners fastest, grind trophies slowest,
                # and almost nobody new holds the platinum yet.
                desc = (trophies[i].get("description") or "").lower()
                if trophies[i].get("tier") == "platinum":
                    tilt = 0.3
                elif "complete chapter" in desc or "complete the final mission." in desc:
                    tilt = 1.8
                elif "final mission for" in desc:
                    tilt = 0.7
                elif trophies[i].get("tier") in ("gold", "silver"):
                    tilt = 0.6
                else:
                    tilt = 1.0
                e_pred = max(e_last, 0.4) * (1.0 + (earner_growth - 1.0) * tilt)
            rate_med = 100.0 * e_pred / p_target
            sigma = math.sqrt(base_spread ** 2 + 0.85 ** 2 + 0.25 / max(e_pred, 1.0))
        else:
            s_e = slopes_e[i]
            if s_e is None:
                s_e = pool_e
            s_e = (2.0 * min(max(s_e, -0.3), 3.0) + 2.0 * pool_e) / 4.0
            e_pred = max(e_last, 0.4) * math.exp(s_e * dx)
            e_pred = min(e_pred, p_target * 0.999)
            rate_comp = 100.0 * e_pred / p_target
            lg = logit(r_last)
            # Growth fitted on the launch surge cannot compound for days:
            # damp every projected logit move, blend it with persistence
            # and cap the total move, so saturation wins over momentum.
            move_comp = logit(rate_comp) - lg
            s_l = slopes_l[i]
            if s_l is not None and 0.0 < r_last < 100.0:
                s_l = (2.0 * min(max(s_l, -2.5), 2.5) + 2.0 * pool_l) / 4.0
                move = 0.6 * move_comp + 0.4 * (s_l * dx)
            else:
                move = move_comp
            move = min(max(0.65 * move * decels[i], -1.8), 1.8)
            med = lg + move
            rate_med = inv_logit(med)
            sigma = math.sqrt(base_spread ** 2 + (resid_sd * (0.6 + dx)) ** 2
                              + 0.04 + 0.3 / max(e_pred, 1.0) + (0.35 * move) ** 2)
        medians.append(rate_med)
        sigmas.append(min(max(sigma, 0.3), 2.0))

    # ---- cross-trophy structure ------------------------------------------
    # The platinum needs every other trophy: its rate cannot top the rarest.
    plat = [i for i, t in enumerate(trophies) if t.get("tier") == "platinum"]
    if plat:
        others = [medians[i] for i in range(n) if i not in plat]
        if others:
            cap = min(others)
            for i in plat:
                medians[i] = min(medians[i], cap)
    # Story chapters finish in order: chapter k+1 cannot be commoner than k.
    chain = []
    for i, t in enumerate(trophies):
        desc = (t.get("description") or "").lower()
        if "complete chapter" in desc or "complete the final mission" in desc:
            chain.append(i)
    run_min = None
    for i in chain:
        if run_min is None or medians[i] < run_min:
            run_min = medians[i]
        else:
            medians[i] = run_min

    rows = []
    for i, t in enumerate(trophies):
        med = logit(medians[i])
        qs = [round4(inv_logit(med + z * sigmas[i])) for z in ZVALS]
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
