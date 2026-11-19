"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's Exophase
earn rate at the case's target, as five quantiles on the logit scale.

Model (three signals combined in logit space, written for this entry):
  1. Rate trend: per trophy, logit(rate) fit on log-time since release over
     the whole history (recency-weighted), slope shrunk toward its tier's
     mean. Launch-week rates move almost linearly in log-time, so this is the
     workhorse.
  2. Counts decomposition: earners E_i(t) fit on log players-tracked and
     log-time (two regressors, shrunk toward "earners track the base"), and
     players N(t) fit with a saturating (concave in log-time) curve blended
     with a straight log-time line. Dividing the two counts separates
     dilution (base growth) from progression (players earning more), which
     matters when the base's growth bends over the horizon.
  3. Persistence: the last observed rate, a small anchor against extrapolation
     running away.
  The three are blended 0.8 / 0.1 / 0.1. Spreads come from the trend fit's
  residuals plus the N-forecast uncertainty (including the disagreement
  between the saturating and linear N fits) plus a random-walk term in the
  horizon, floored so a quiet history never yields an overconfident band.
  A history of identical reads (Exophase has not refreshed yet) takes a
  separate path: a pending refresh is assumed, the median shifts down by the
  expected dilution and the band widens. A trophy everyone already has is
  handled by the lag of new players behind the tracked base.

Standard library only, deterministic (no randomness, no unordered iteration
over output-affecting data), runs in well under a second per case.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Recency half-life for the trend fits, in hours: the freshest leg of the
# rising trend carries the most weight.
TAU_HOURS = 12.0
# Slope shrinkage: a trophy's own slope gets weight n / (n + SHRINK_K).
SHRINK_K = 3.0
# Blend weights: trend / decomposition / persistence.
W_TREND = 0.8
W_DECOMP = 0.1
# Base random-walk volatility of a rate's logit, per square-root day.
RW_SIGMA = 0.20
# Extra spread for the answer depending on when Exophase last refreshed.
REFRESH_SIGMA = 0.15


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, halves away from zero; whole numbers as ints."""
    scaled = math.floor(x * 10000.0 + 0.5)
    value = scaled / 10000.0
    if value == int(value):
        return int(value)
    return value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_epoch(case):
    """Hours are measured from the game's release (midnight UTC of its release
    date); fall back to the first read when the date is missing."""
    game = case.get("game") or {}
    raw = game.get("release_date")
    if raw:
        try:
            day = datetime.fromisoformat(raw)
            return day.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    hist = case.get("history") or []
    if hist:
        return parse_time(hist[0]["at"])
    return parse_time(case["at"])


def distinct_reads(case):
    """Collapse consecutive identical reads (same players tracked and same
    rates): Exophase only changes the page at a refresh, so a run of identical
    reads is one observation, timed at its first appearance."""
    out = []
    prev_key = None
    for read in case.get("history") or []:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or ()))
        if key == prev_key:
            continue
        prev_key = key
        out.append(read)
    return out


def wls(design, target, weights):
    """Weighted least squares for a small design (list of row lists).
    Returns (coeffs, residual_std) or None when degenerate."""
    n = len(design)
    if n == 0:
        return None
    k = len(design[0])
    sw = sum(weights)
    if sw <= 0:
        return None
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for row, y, w in zip(design, target, weights):
        for i in range(k):
            atb[i] += w * row[i] * y
            for j in range(k):
                ata[i][j] += w * row[i] * row[j]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(ata[r][col]))
        if abs(ata[piv][col]) < 1e-12:
            return None
        ata[col], ata[piv] = ata[piv], ata[col]
        atb[col], atb[piv] = atb[piv], atb[col]
        for r in range(col + 1, k):
            f = ata[r][col] / ata[col][col]
            for c in range(col, k):
                ata[r][c] -= f * ata[col][c]
            atb[r] -= f * atb[col]
    coef = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = atb[i] - sum(ata[i][j] * coef[j] for j in range(i + 1, k))
        coef[i] = s / ata[i][i]
    resid_sq = 0.0
    for row, y, w in zip(design, target, weights):
        pred = sum(c * v for c, v in zip(coef, row))
        resid_sq += w * (y - pred) ** 2
    dof = max(n - k, 1)
    resid = math.sqrt(resid_sq / sw * n / dof)
    return coef, resid


def recency_weights(times, t_last):
    return [math.exp(-(t_last - t) / TAU_HOURS) for t in times]


def fit_players(reads, epoch):
    """Forecast log players-tracked at any hour. Adoption bends: a straight
    line in log-time overestimates the future, a concave quadratic
    underestimates it, so the forecast blends the two in log space and the
    spread counts their disagreement. Returns (predict_fn, resid) where
    predict_fn(hours) -> (log N, std), or None."""
    xs, ys = [], []
    t_last = None
    for read in reads:
        n = read.get("players_tracked")
        if not n or n <= 0:
            continue
        t = (parse_time(read["at"]) - epoch).total_seconds() / 3600.0
        if t_last is None or t > t_last:
            t_last = t
        xs.append(t)
        ys.append(math.log(n))
    if len(xs) < 2 or t_last is None:
        return None
    ws = recency_weights(xs, t_last)
    lx = [math.log1p(max(t, 0.0)) for t in xs]
    lin = wls([[1.0, v] for v in lx], ys, ws)
    if lin is None:
        return None
    (lin_coef, lin_resid) = lin
    quad = None
    if len(set(round(v, 6) for v in lx)) >= 4:
        solved = wls([[1.0, v, v * v] for v in lx], ys, ws)
        if solved is not None and solved[0][2] <= 0.0:
            quad = solved
    mean_lx = sum(w * v for w, v in zip(ws, lx)) / sum(ws)
    sxx = sum(w * (v - mean_lx) ** 2 for w, v in zip(ws, lx))
    cap = ys[-1] + math.log(25.0)

    def predict(t_hours):
        v = math.log1p(max(t_hours, 0.0))
        v_lin = lin_coef[0] + lin_coef[1] * v
        if quad is not None:
            qc = quad[0]
            v_quad = qc[0] + qc[1] * v + qc[2] * v * v
            value = 0.5 * (v_lin + v_quad)
            disagree = abs(v_lin - v_quad) / 2.0
            resid = max(lin_resid, quad[1])
        else:
            value = v_lin
            disagree = 0.0
            resid = lin_resid
        value = min(value, cap)
        lev = 1.0 + (1.0 / len(xs))
        if sxx > 0:
            lev += (v - mean_lx) ** 2 / sxx
        std = math.sqrt((resid * math.sqrt(lev)) ** 2 + disagree ** 2)
        return value, max(std, 0.10)

    return predict, lin_resid


def fit_rate_trend(history, epoch, idx, checkpoint):
    """Per-trophy fit of logit(rate) on log-time. Launch-week rates are not
    one straight line: they fall hard while the tracked base explodes, then
    rise as the base matures, so the fit uses only the last 24 hours of reads
    (the maturation leg), widening to 48h and then the full history when the
    window holds too few reads. Returns (slope, intercept, resid_std,
    n_points) or None."""
    cutoff_24 = checkpoint.timestamp() - 24 * 3600
    cutoff_48 = checkpoint.timestamp() - 48 * 3600
    for cutoff in (cutoff_24, cutoff_48, None):
        xs, ys = [], []
        t_last = None
        for read in history:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            t = (parse_time(read["at"]) - epoch).total_seconds() / 3600.0
            if cutoff is not None and parse_time(read["at"]).timestamp() < cutoff:
                continue
            if t_last is None or t > t_last:
                t_last = t
            xs.append(t)
            ys.append(logit_pct(rates[idx]))
        if len(xs) < 3 or t_last is None:
            continue
        ws = recency_weights(xs, t_last)
        lx = [math.log1p(max(t, 0.0)) for t in xs]
        if len(set(round(v, 6) for v in lx)) < 2:
            continue
        solved = wls([[1.0, v] for v in lx], ys, ws)
        if solved is None:
            continue
        (intercept, slope), resid = solved
        sw = sum(ws)
        x_mean = sum(w * v for w, v in zip(ws, lx)) / sw
        sxx = sum(w * (v - x_mean) ** 2 for w, v in zip(ws, lx))
        return slope, intercept, resid, len(xs), x_mean, sxx
    return None


def fit_earners2(reads, epoch, idx):
    """Per-trophy fit of log earners on log players-tracked and log-time.
    Earners grow with the base (coefficient near 1 for easy trophies) and
    with maturation (the time term); both coefficients are shrunk toward
    (1, 0) to tame the collinearity. Returns (b_n, b_t, intercept, resid,
    n_points) or None."""
    xs_n, xs_t, ys, ts = [], [], [], []
    t_last = None
    for read in reads:
        rates = read.get("rates") or []
        n = read.get("players_tracked")
        if idx >= len(rates) or rates[idx] is None or not n or n <= 0:
            continue
        t = (parse_time(read["at"]) - epoch).total_seconds() / 3600.0
        if t_last is None or t > t_last:
            t_last = t
        ts.append(t)
        xs_n.append(math.log(n))
        xs_t.append(math.log1p(max(t, 0.0)))
        ys.append(math.log(max(rates[idx] * n / 100.0, 0.3)))
    if len(ys) < 3 or t_last is None:
        return None
    ws = recency_weights(ts, t_last)
    design = [[1.0, a, b] for a, b in zip(xs_n, xs_t)]
    solved = wls(design, ys, ws)
    if solved is None:
        return None
    (intercept, b_n, b_t), resid = solved
    w_own = len(ys) / (len(ys) + 4.0)
    b_n = w_own * b_n + (1.0 - w_own) * 1.0
    b_t = w_own * b_t + (1.0 - w_own) * 0.0
    # Refit the intercept around the shrunk slopes.
    sw = sum(ws)
    intercept = sum(w * (y - b_n * a - b_t * b)
                    for w, y, a, b in zip(ws, ys, xs_n, xs_t)) / sw
    return b_n, b_t, intercept, resid, len(ys)


def last_known_rate(case, idx):
    """The trophy's rate in the most recent read that lists it."""
    for read in reversed(case.get("history") or []):
        rates = read.get("rates") or []
        if idx < len(rates) and rates[idx] is not None:
            return rates[idx]
    return None


def forecast_case(case):
    trophies = case.get("trophies") or []
    horizon_days = max(case.get("horizon_minutes", 0) / 1440.0, 0.0)
    epoch = release_epoch(case)
    target = parse_time(case["target"])
    t_target = (target - epoch).total_seconds() / 3600.0
    x_target = math.log1p(max(t_target, 0.0))
    reads = distinct_reads(case)
    last = (case.get("history") or [None])[-1]

    n_model = fit_players(reads, epoch) if len(reads) >= 2 else None
    log_n_target = None
    n_std = 0.5
    if n_model is not None:
        predict_n, _ = n_model
        log_n_target, n_std = predict_n(t_target)
        last_n = last.get("players_tracked") if last else None
        if last_n:
            log_n_target = max(log_n_target, math.log(last_n))

    # Per-trophy trend fits, with tier-mean slopes for shrinkage. When the
    # page has not refreshed at all (every read identical), there is no trend
    # to fit: the pending refresh is the whole story, so every trophy takes
    # the stale-page path below instead.
    trends = {}
    tier_slopes = {}
    stale = len(reads) <= 1
    checkpoint = parse_time(case["at"])
    history = case.get("history") or []
    if not stale:
        for i, trophy in enumerate(trophies):
            fit = fit_rate_trend(history, epoch, i, checkpoint)
            if fit is not None:
                trends[i] = fit
                tier_slopes.setdefault(trophy.get("tier", "bronze"), []).append(fit[0])
    all_slopes = [f[0] for f in trends.values()]
    global_slope = sum(all_slopes) / len(all_slopes) if all_slopes else 0.0
    tier_mean = {}
    for tier, slopes in tier_slopes.items():
        tier_mean[tier] = (sum(slopes) / len(slopes) if len(slopes) >= 3
                           else global_slope)

    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = last_known_rate(case, i)
        if last_rate is None:
            last_rate = 0.5
        lg_last = logit_pct(last_rate)

        if last_rate >= 99.99 and n_model is not None:
            # Everyone has it: its earners lag the tracked base by the time
            # a new player needs to earn it (~2h for an intro trophy).
            predict_n, _ = n_model
            log_now, _ = predict_n(t_target)
            log_lag, _ = predict_n(max(t_target - 2.0, 0.0))
            median = 100.0 * math.exp(log_lag - log_now)
            median = min(max(median, 90.0), 99.9)
            sigma = 0.35 + 0.10 * math.sqrt(horizon_days)
            rows.append({"id": trophy["id"], "q": [
                round4(inv_logit_pct(logit_pct(median) + z * sigma))
                for z in Z]})
            continue

        if i not in trends:
            # Degenerate history (a page Exophase has not refreshed yet, or a
            # trophy with no usable series): assume a refresh is coming, which
            # dilutes every rate as the tracked base jumps, and say so widely.
            # The jump scales with how long the page has been stale and how
            # early in the launch it is (a young base grows explosively; a
            # mature one barely moves). A saturated trophy cannot fall far,
            # so it keeps its level.
            if last_rate >= 99.99:
                median_lg = logit_pct(99.9)
                sigma = 0.5
            else:
                stale_hours = 0.0
                if history:
                    stale_hours = max(
                        (checkpoint - parse_time(history[0]["at"])).total_seconds() / 3600.0,
                        0.0)
                t_since_release = max(
                    (checkpoint - epoch).total_seconds() / 3600.0, 0.0)
                growth = 1.2 * 12.0 / (12.0 + t_since_release)
                shift = -math.log1p(growth * stale_hours)
                shift = max(shift, -1.8)
                median_lg = lg_last + shift
                sigma = 0.5 * math.sqrt(horizon_days) + 1.0
            rows.append({"id": trophy["id"], "q": [
                round4(inv_logit_pct(median_lg + z * sigma)) for z in Z]})
            continue

        slope, intercept, resid, n_pts, x_mean, sxx = trends[i]
        tier = trophy.get("tier", "bronze")
        w_own = n_pts / (n_pts + SHRINK_K)
        slope_s = w_own * slope + (1.0 - w_own) * tier_mean.get(tier, global_slope)
        lg_trend = intercept + slope_s * x_target

        # Decomposition signal: earners forecast divided by the N forecast.
        lg_decomp = None
        efit = fit_earners2(reads, epoch, i)
        if efit is not None and log_n_target is not None:
            b_n, b_t, e_intercept, _, _ = efit
            log_e = e_intercept + b_n * log_n_target + b_t * x_target
            e_last = max(last_rate * (last.get("players_tracked") or 0) / 100.0,
                         0.3)
            log_e = max(log_e, math.log(e_last))
            lg_decomp = logit_pct(100.0 * math.exp(log_e - log_n_target))

        if lg_decomp is None:
            median_lg = (W_TREND * lg_trend
                         + (1.0 - W_TREND) * lg_last)
        else:
            median_lg = (W_TREND * lg_trend + W_DECOMP * lg_decomp
                         + (1.0 - W_TREND - W_DECOMP) * lg_last)

        # Prediction interval for the trend: the fit's scatter plus the
        # slope's own uncertainty, which grows with the extrapolation
        # distance (a 24h window with few distinct reads extrapolates
        # 4.5 days, so this term dominates for quiet trophies).
        leverage = 0.0
        if sxx > 1e-9:
            leverage = min((x_target - x_mean) ** 2 / sxx, 9.0)
        pred_var = resid * resid * (1.0 + 1.0 / n_pts + leverage)
        var = pred_var + n_std * n_std
        var += (RW_SIGMA ** 2) * horizon_days + REFRESH_SIGMA ** 2
        sigma = math.sqrt(var)
        floor = 0.30 + 0.12 * math.sqrt(horizon_days)
        sigma = min(max(sigma, floor), 2.5)
        rows.append({"id": trophy["id"], "q": [
            round4(inv_logit_pct(median_lg + z * sigma)) for z in Z]})

    rows.sort(key=lambda r: r["id"])
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
