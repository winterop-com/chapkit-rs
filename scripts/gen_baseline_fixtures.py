"""Generate golden outputs from chap-core's built-in baselines.

Run with chap-core's environment, from the chapkit-rs root:

    ../chap-core/.venv/bin/python scripts/gen_baseline_fixtures.py

Writes tests/fixtures/baselines_<case>.json with the
train/historic/future frames (chapkit {columns, data} format) and the samples
each baseline predicts, so the Rust ports can be checked number for number.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from chap_core.models.builtin import get_builtin_model
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

OUT = Path(__file__).resolve().parent.parent / "tests/fixtures"
MODELS = ["global_median", "seasonal_median", "seasonal_shift_persistence"]


def monthly(i: int) -> str:
    return f"{2018 + i // 12}-{i % 12 + 1:02d}"


def weekly(i: int) -> str:
    # 2020 has an ISO week 53; start in 2019W40 so the series crosses it.
    year, week = 2019, 40 + i
    while True:
        weeks = 53 if year in (2015, 2020, 2026) else 52
        if week <= weeks:
            return f"{year}W{week:02d}"
        week -= weeks
        year += 1


def frames(period, n_train: int, horizon: int, rng: np.random.Generator):
    rows = []
    for li, loc in enumerate(["A", "B", "C"]):
        for i in range(n_train + horizon):
            season = math.sin(2 * math.pi * i / 12)
            cases = float(rng.poisson(max(0.5, 20 + 10 * season + 5 * li)))
            # holes: B misses some values, C's last historic value is missing
            if loc == "B" and i % 7 == 3:
                cases = float("nan")
            if loc == "C" and i == n_train - 1:
                cases = float("nan")
            rows.append({"time_period": period(i), "location": loc, "disease_cases": cases, "rainfall": 100 + 30 * season})
    df = pd.DataFrame(rows)
    train = df[df.time_period.isin([period(i) for i in range(n_train)])]
    future = df[df.time_period.isin([period(i) for i in range(n_train, n_train + horizon)])].drop(columns="disease_cases")
    return train.reset_index(drop=True), future.reset_index(drop=True)


def wire(df: pd.DataFrame) -> dict:
    data = [[None if isinstance(v, float) and math.isnan(v) else v for v in row] for row in df.itertuples(index=False)]
    return {"columns": list(df.columns), "data": data}


def run(case: str, period, n_train: int, horizon: int) -> None:
    rng = np.random.default_rng(42)
    train, future = frames(period, n_train, horizon, rng)
    # an extra location "D" only in the future, to exercise the fallbacks
    extra = future[future.location == "A"].copy()
    extra["location"] = "D"
    future_all = pd.concat([future, extra], ignore_index=True)
    historic_extra = train[train.location == "A"].copy()
    historic_extra["location"] = "D"
    historic = pd.concat([train, historic_extra], ignore_index=True)

    out = {"train": wire(train), "historic": wire(historic), "future": wire(future_all), "predictions": {}}
    for name in MODELS:
        model = get_builtin_model(name)()
        model.train(DataSet.from_pandas(train))
        pred = model.predict(DataSet.from_pandas(historic), DataSet.from_pandas(future_all))
        out["predictions"][name] = {
            f"{loc}|{period.id}": [float(v) for v in row]
            for loc, samples in pred.items()
            for period, row in zip(samples.time_period, samples.samples, strict=True)
        }
    path = OUT / f"baselines_{case}.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path} ({len(future_all)} future rows)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    run("monthly", monthly, 40, 6)
    run("weekly", weekly, 80, 8)
