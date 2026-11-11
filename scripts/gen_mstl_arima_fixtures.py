"""Generate chapkit-eval's MSTL + ARIMA weather parity fixtures from chap-core's own provider.

Run with a chap-core environment (made with chap-core 2.7.0, statsmodels 0.15), from the
chapkit-rs root:

    $CHAP_CORE/.venv/bin/python -I scripts/gen_mstl_arima_fixtures.py

(`make eval-fixtures` runs it too). The series come from a fixed seed.

Writes crates/chapkit-eval/fixtures/mstl_arima.json with, per series (monthly and weekly, seasonal
covariates of different lengths and noise, a random walk, a trend, a constant series (KPSS's
constant-series shortcut), short ones, one where most candidates fail the root check): the history, every stage of chap-core's `_forecast_series` (the MSTL seasonal component, the
number of differences, each (p, q) candidate's fit: AICc, log-likelihood, parameters and whether
its roots pass, the chosen order and parameters, the trend forecast) and the provider's forecast,
with the seconds chap-core took for it. A series of exactly two cycles is listed under `errors`.
"""

import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import MSTL

from chap_core.assessment.weather_providers import mstl_arima as m

OUT = Path(os.environ.get("FIXTURES_OUT") or Path(__file__).resolve().parent.parent / "crates/chapkit-eval/fixtures")


def num(v):
    v = float(v)
    return None if math.isnan(v) else (1e999 if v == math.inf else v)


def series(rng, kind: str, n: int, per: int) -> np.ndarray:
    t = np.arange(n)
    s = np.sin(2 * np.pi * t / per)
    if kind == "rainfall":
        return np.round(np.maximum(0, 120 + 80 * s + rng.normal(0, 15, n)), 2)
    if kind == "temperature":
        return np.round(24 + 5 * s + rng.normal(0, 1, n), 2)
    if kind == "humidity":
        return np.round(60 + 20 * np.cos(2 * np.pi * t / per) + rng.normal(0, 5, n), 2)
    if kind == "walk":
        return np.round(50 + 5 * s + np.cumsum(rng.normal(0, 2, n)), 3)
    if kind == "trend":
        return np.round(10 + 0.3 * t + 3 * s + rng.normal(0, 0.5, n), 3)
    if kind == "constant":
        return np.full(n, 7.5)
    if kind == "spike":
        # A dry series with one storm: most candidates fail the root check.
        return np.where(t == n - 6, 100.0, 0.0)
    if kind == "linear":
        return t.astype(float)
    raise ValueError(kind)


def stages(y: np.ndarray, per: int, h: int) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seasonal = np.asarray(MSTL(y, periods=per).fit().seasonal, dtype=float)
        deseasonalised = y - seasonal
        d = m._n_differences(deseasonalised)
        candidates = []
        best, best_score = None, np.inf
        for p in range(m.MAX_P + 1):
            for q in range(m.MAX_Q + 1):
                try:
                    fitted = ARIMA(deseasonalised, order=(p, d, q)).fit()
                except Exception as e:  # noqa: BLE001 - as chap-core: skip orders that fail
                    candidates.append({"p": p, "q": q, "failed": type(e).__name__})
                    continue
                ok = bool(m._is_well_conditioned(fitted))
                candidates.append(
                    {
                        "p": p,
                        "q": q,
                        "aicc": num(fitted.aicc),
                        "llf": num(fitted.llf),
                        "params": [num(v) for v in fitted.params],
                        "iterations": int(fitted.mle_retvals.get("iterations", -1)),
                        "converged": bool(fitted.mle_retvals.get("converged", False)),
                        "well_conditioned": ok,
                    }
                )
                if ok and np.isfinite(fitted.aicc) and fitted.aicc < best_score:
                    best_score, best = fitted.aicc, fitted
        fallback = best is None
        if fallback:
            best = ARIMA(deseasonalised, order=m.FALLBACK_ARIMA_ORDER).fit()
        chosen = m._select_model(deseasonalised)
        assert chosen.model.order == best.model.order, (chosen.model.order, best.model.order)
        trend_forecast = np.asarray(best.forecast(h), dtype=float)
    start = time.perf_counter()
    forecast = m._forecast_series(y, h, per, None)
    seconds = time.perf_counter() - start
    return {
        "seasonal": [num(v) for v in seasonal],
        "d": int(d),
        "candidates": candidates,
        "order": list(best.model.order),
        "fallback": fallback,
        "params": [num(v) for v in best.params],
        "trend_forecast": [num(v) for v in trend_forecast],
        "forecast": [num(v) for v in forecast],
        "seconds": seconds,
    }


def main() -> None:
    rng = np.random.default_rng(7)
    cases = [
        ("monthly_rainfall_60", "rainfall", 60, 12, 6),
        ("monthly_temperature_48", "temperature", 48, 12, 6),
        ("monthly_humidity_37", "humidity", 37, 12, 3),
        ("monthly_walk_72", "walk", 72, 12, 6),
        ("monthly_trend_50", "trend", 50, 12, 6),
        ("monthly_constant_40", "constant", 40, 12, 6),
        ("monthly_rainfall_short_25", "rainfall", 25, 12, 3),
        ("monthly_spike_short_26", "spike", 26, 12, 3),
        ("monthly_linear_short_26", "linear", 26, 12, 3),
        ("weekly_rainfall_160", "rainfall", 160, 52, 8),
        ("weekly_temperature_110", "temperature", 110, 52, 8),
        ("weekly_walk_130", "walk", 130, 52, 8),
        ("weekly_humidity_short_105", "humidity", 105, 52, 4),
        ("weekly_constant_120", "constant", 120, 52, 4),
    ]
    out = []
    total = 0.0
    for name, kind, n, per, h in cases:
        y = series(rng, kind, n, per)
        st = stages(y, per, h)
        total += st["seconds"]
        out.append({"name": name, "period": per, "horizon": h, "y": [float(v) for v in y]} | st)
        print(f"{name}: d={st['d']} order={st['order']} fallback={st['fallback']} {st['seconds']:.2f}s")
    errors = []
    y = series(rng, "rainfall", 24, 12)
    try:
        m._forecast_series(y, 3, 12, None)
    except ValueError as e:
        errors.append({"name": "monthly_two_cycles_24", "period": 12, "y": [float(v) for v in y], "error": str(e)})
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "mstl_arima.json"
    path.write_text(json.dumps({"chap_core": "2.7.0", "cases": out, "errors": errors, "seconds": total}) + "\n")
    print(f"wrote {path} ({total:.1f}s in chap-core's provider)")


if __name__ == "__main__":
    main()
