"""Generate chapkit-eval's parity fixtures from chap-core's own backtest and metrics.

Run with a chap-core environment (the fixtures were last made with chap-core 2.7.0), from the
chapkit-rs root:

    make eval-fixtures CHAP_CORE=../../chap-core

which runs `$CHAP_CORE/.venv/bin/python scripts/gen_eval_fixtures.py`. The data is generated
from a fixed seed, so the output is the same on every run of the same chap-core release.

Writes crates/chapkit-eval/fixtures/eval_<case>.json, each with:

- `data`: the dataset (chapkit `{columns, data}`), with gaps: a missing and a zero target in the
  evaluation window, and a covariate never observed in one season for one location (so the
  climatology falls back to its regression);
- `params`: n_periods, n_splits, stride, n_retrain (and masked ranges, if any);
- `dropped`: the locations chap-core's REST backtest leaves out (`validate_and_filter_dataset_for_evaluation`:
  no observed target before the test windows); the `untrained` case has one;
- `prepared`: the dataset as chap-core's REST API loads it (`DataSet.from_pandas(df,
  fill_missing=True)`): sorted, gaps filled, every location padded to the same range. The
  `messy_*` cases have a gap inside one location, a location starting later and one ending
  earlier, a period missing for every location and shuffled rows (weekly across 2020W53);
- `splits`: per weather provider, chap-core's `train_test_generator` windows: the last train
  period, each split's last historic period, its future periods and the masked future frame;
- `models`: per model (chap-core's `seasonal_median`; `covariate`, a deterministic test model
  that reads history and future covariates; `since_last`, one that counts the periods since the
  last observation, as chap-core's persistence does; both repeated in Rust; `signed`, one with samples
  below -1, scored from its forecasts only), the flat forecasts of chap-core's
  `backtest` with retraining and the metrics: the aggregate as chap-core's REST API stores it
  (`compute_all_aggregated_metrics_from_backtest`, observations from the whole dataset), and the
  breakdowns by horizon and by location.
"""

import dataclasses
import datetime
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from chap_core.api_types import BacktestParams
from chap_core.assessment.dataset_splitting import train_test_generator
from chap_core.assessment.evaluation import Evaluation
from chap_core.assessment.flat_representations import DataDimension
from chap_core.assessment.metrics import available_metrics, compute_all_aggregated_metrics_from_backtest
from chap_core.database.dataset_tables import DataSet as DataSetDB
from chap_core.database.dataset_tables import MaskedRange, Observation
from chap_core.database.model_templates_and_config_tables import ConfiguredModelDB, ModelTemplateDB
from chap_core.datatypes import Samples
from chap_core.models.builtin import get_builtin_model
from chap_core.rest_api.db_worker_functions import validate_and_filter_dataset_for_evaluation
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet

OUT = Path(os.environ.get("FIXTURES_OUT") or Path(__file__).resolve().parent.parent / "crates/chapkit-eval/fixtures")
PROVIDERS = ["climatology", "damped_persistence", "observed", "mstl_arima"]


def monthly(i: int) -> str:
    return f"{2018 + i // 12}-{i % 12 + 1:02d}"


def weekly(i: int) -> str:
    # ISO weeks from 2019W01; 2020 has a week 53.
    year, week, _ = (datetime.date.fromisocalendar(2019, 1, 1) + datetime.timedelta(weeks=i)).isocalendar()
    return f"{year}W{week:02d}"


def dataset(period, n: int, per_year: int, locations: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for li, loc in enumerate(locations):
        population = int(rng.integers(50_000, 500_000))
        for i in range(n):
            season = math.sin(2 * math.pi * (i / per_year - 0.1 * li))
            rainfall = round(max(0.0, 120 + 80 * season + rng.normal(0, 15)), 2)
            temperature = round(24 + 5 * season + rng.normal(0, 1), 2)
            humidity = round(60 + 20 * season + rng.normal(0, 5), 2)
            cases = float(rng.poisson(max(0.5, 20 + 12 * season + 0.05 * rainfall + 4 * li)))
            p = period(i)
            # Never observed in one season for one location: the climatology's regression path.
            if li == 1 and i % per_year == 0:
                humidity = float("nan")
            # A missing and a zero target in the evaluation window.
            if li == 0 and i == n - 2:
                cases = float("nan")
            if li == 1 and i == n - 3:
                cases = 0.0
            rows.append(
                {
                    "time_period": p,
                    "location": loc,
                    "disease_cases": cases,
                    "rainfall": rainfall,
                    "mean_temperature": temperature,
                    "humidity": humidity,
                    "population": population,
                }
            )
    return pd.DataFrame(rows)


def cell(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else float(v)
    return v


def wire(df: pd.DataFrame) -> dict:
    return {"columns": list(df.columns), "data": [[cell(v) for v in row] for row in df.itertuples(index=False)]}


def frame(ds: DataSet) -> pd.DataFrame:
    """A chap-core DataSet as one frame, rows by location then period, periods as chap-core ids."""
    rows = []
    for loc, d in ds.items():
        fields = [f.name for f in dataclasses.fields(d) if f.name != "time_period"]
        for i, p in enumerate(d.time_period):
            rows.append({"time_period": p.id, "location": loc} | {f: getattr(d, f)[i] for f in fields})
    return pd.DataFrame(rows)


class Covariate:
    """Deterministic test model, repeated in crates/chapkit-eval/tests/parity.rs.

    Per sample k: max(0, 0.5 * mean + 0.5 * last + 0.1 * (rainfall - 120) + 0.5 * (mean_temperature - 24) + OFFSETS[k]),
    with `mean` the location's mean observed target in the training data and `last` its last
    observed target in the historic data.
    """

    OFFSETS = np.array([-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0])

    def train(self, data: DataSet):
        means = {}
        for loc, d in data.items():
            y = np.asarray(d.disease_cases, dtype=float)
            means[loc] = float(np.mean(y[np.isfinite(y)]))
        return CovariatePredictor(means)


class CovariatePredictor:
    def __init__(self, means):
        self.means = means

    def predict(self, historic: DataSet, future: DataSet) -> DataSet:
        out = {}
        for loc, f in future.items():
            y = np.asarray(historic[loc].disease_cases, dtype=float)
            last = float(y[np.isfinite(y)][-1])
            base = (
                0.5 * self.means[loc]
                + 0.5 * last
                + 0.1 * (np.asarray(f.rainfall, dtype=float) - 120)
                + 0.5 * (np.asarray(f.mean_temperature, dtype=float) - 24)
            )
            out[loc] = Samples(f.time_period, np.maximum(base[:, None] + Covariate.OFFSETS[None, :], 0.0))
        return DataSet(out)


class SinceLast:
    """Deterministic test model, repeated in crates/chapkit-eval/tests/parity.rs.

    For the h-th future period (0-based) of a location: max(0, last + 2 * (since + h) + OFFSETS[k]),
    with `last` the last observed target in the historic data and `since` the number of historic
    periods from it to the end of the history (1 when the last period is observed), as chap-core's
    persistence baseline counts them: gaps and padding count.
    """

    def train(self, data: DataSet):
        return self

    def predict(self, historic: DataSet, future: DataSet) -> DataSet:
        out = {}
        for loc, f in future.items():
            y = np.asarray(historic[loc].disease_cases, dtype=float)
            observed = np.flatnonzero(np.isfinite(y))
            last, since = float(y[observed[-1]]), len(y) - observed[-1]
            steps = np.arange(len(f.time_period), dtype=float)
            base = last + 2.0 * (since + steps)
            out[loc] = Samples(f.time_period, np.maximum(base[:, None] + Covariate.OFFSETS[None, :], 0.0))
        return DataSet(out)


class Signed:
    """Deterministic test model whose samples go below -1, where log1p is NaN (the log-Winkler
    metrics are then NaN, as numpy's percentile propagates it). Scored only from its forecasts.

    Per sample k: last - mean + OFFSETS[k], `mean` and `last` as in `Covariate`.
    """

    def train(self, data: DataSet):
        return SignedPredictor(Covariate().train(data).means)


class SignedPredictor(CovariatePredictor):
    def predict(self, historic: DataSet, future: DataSet) -> DataSet:
        out = {}
        for loc, f in future.items():
            y = np.asarray(historic[loc].disease_cases, dtype=float)
            last = float(y[np.isfinite(y)][-1])
            base = np.full(len(f.time_period), last - self.means[loc])
            out[loc] = Samples(f.time_period, base[:, None] + Covariate.OFFSETS[None, :])
        return DataSet(out)


class Builtin:
    """chap-core's estimator shape around a built-in baseline: train returns the predictor."""

    def __init__(self, name: str):
        self.name = name

    def train(self, data: DataSet):
        model = get_builtin_model(self.name)()
        model.train(data)
        return model


def splits(ds: DataSet, params: dict, provider: str) -> dict:
    train, gen = train_test_generator(
        ds, params["n_periods"], params["n_splits"], params["stride"], future_weather_provider=provider
    )
    out = []
    for historic, masked, future in gen:
        out.append(
            {
                "historic_last": historic.period_range[-1].id,
                "future_periods": [p.id for p in future.period_range],
                "masked_future": wire(frame(masked)),
            }
        )
    return {"train_last": train.period_range[-1].id, "splits": out}


def metrics_of(ds: DataSet, estimator, params: dict, stored: DataSet) -> dict:
    model_template = ModelTemplateDB(id="fixture", name="fixture", version="1")
    configured = ConfiguredModelDB(id="fixture", model_template_id="fixture", model_template=model_template)
    evaluation = Evaluation.create(
        configured_model=configured,
        estimator=estimator,
        dataset=ds,
        backtest_params=BacktestParams(
            n_periods=params["n_periods"],
            n_splits=params["n_splits"],
            stride=params["stride"],
            n_retrain=params["n_retrain"],
            future_weather_provider="climatology",
        ),
        historical_context_years=0,
    )
    backtest = evaluation.to_backtest()
    # As chap-core's REST API stores aggregate_metrics: the backtest's dataset is the whole
    # stored dataset, so the observations are every observed target, not only the test windows.
    observations = [
        Observation(period=p.id, org_unit=loc, value=float(v), feature_name="disease_cases")
        for loc, d in stored.items()
        for p, v in zip(d.time_period, d.disease_cases, strict=True)
        if np.isfinite(v)
    ]
    backtest.dataset = DataSetDB(name="fixture", observations=observations)
    aggregate = compute_all_aggregated_metrics_from_backtest(backtest)

    flat = Evaluation.from_backtest(backtest).to_flat()
    by = {}
    for dim in (DataDimension.horizon_distance, DataDimension.location):
        by[dim.value] = {}
        for metric_id, cls in available_metrics.items():
            metric = cls()
            df = metric.get_metric(flat.observations, flat.forecasts, dimensions=(dim,))
            by[dim.value][metric_id] = {str(k): cell(v) for k, v in zip(df[dim.value], df["metric"], strict=True)}

    forecasts = []
    keys = ["location", "time_period", "horizon_distance"]
    for (loc, period, h), g in flat.forecasts.groupby(keys, sort=True):
        forecasts.append(
            {
                "location": loc,
                "time_period": period,
                "horizon": int(h),
                "samples": [float(v) for v in g.sort_values("sample")["forecast"]],
            }
        )
    return {
        "forecasts": forecasts,
        "aggregate": aggregate,
        "by_horizon": by["horizon_distance"],
        "by_location": by["location"],
    }


def messy(df: pd.DataFrame, periods: list[str], gap: range, late: int, early: int, everywhere: int, seed: int) -> pd.DataFrame:
    """`df` with a gap in the first location, the second starting late, the third ending early,
    one period missing everywhere, and the rows shuffled."""
    locs = sorted(df.location.unique())
    drop = (
        ((df.location == locs[0]) & df.time_period.isin([periods[i] for i in gap]))
        | ((df.location == locs[1]) & df.time_period.isin(periods[:late]))
        | ((df.location == locs[2]) & df.time_period.isin(periods[early:]))
        | (df.time_period == periods[everywhere])
    )
    kept = df[~drop]
    return kept.iloc[np.random.default_rng(seed).permutation(len(kept))].reset_index(drop=True)


def run(case: str, df: pd.DataFrame, params: dict, masked: list[dict] | None = None) -> None:
    # As chap-core's REST API loads a dataset (worker_functions.py, crud.py).
    ds = DataSet.from_pandas(df, fill_missing=True)
    if masked:
        ds = ds.apply_masked_ranges([MaskedRange(**r) for r in masked])
    stored = ds
    # As resolve_backtest_specification: locations without a target before the test windows go.
    ds = validate_and_filter_dataset_for_evaluation(
        ds, "disease_cases", params["n_periods"], params["n_splits"], params["stride"]
    )
    out = {
        "chap_core": "2.7.0",
        "params": params | ({"masked_ranges": masked} if masked else {}),
        "data": wire(df),
        "prepared": wire(frame(stored)),
        "dropped": sorted(set(stored.keys()) - set(ds.keys())),
        "splits": {p: splits(ds, params, p) for p in PROVIDERS},
        "models": {
            "seasonal_median": metrics_of(ds, Builtin("seasonal_median"), params, stored),
            "covariate": metrics_of(ds, Covariate(), params, stored),
            "since_last": metrics_of(ds, SinceLast(), params, stored),
            "signed": metrics_of(ds, Signed(), params, stored),
        },
    }
    path = OUT / f"eval_{case}.json"
    path.write_text(json.dumps(out, indent=None) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    run(
        "monthly",
        dataset(monthly, 48, 12, ["A", "B", "C"], seed=1),
        {"n_periods": 3, "n_splits": 4, "stride": 2, "n_retrain": 2},
    )
    run(
        "weekly",
        dataset(weekly, 130, 52, ["A", "B"], seed=2),
        {"n_periods": 4, "n_splits": 5, "stride": 1, "n_retrain": 2},
    )
    run(
        "masked",
        dataset(monthly, 36, 12, ["A", "B"], seed=3),
        {"n_periods": 2, "n_splits": 3, "stride": 1, "n_retrain": 3},
        masked=[{"start": "2018-03", "end": "2018-05"}],
    )
    # A gap in A at the end of the first split's history (2021-02, 2021-03), C ending 8 periods early.
    run(
        "messy_monthly",
        messy(dataset(monthly, 48, 12, ["A", "B", "C", "D"], seed=4), [monthly(i) for i in range(48)], range(37, 39), 6, 40, 20, seed=4),
        {"n_periods": 3, "n_splits": 4, "stride": 2, "n_retrain": 2},
    )
    # C has no target before the test windows (observed only inside them), so chap-core drops it;
    # its observations still set crps_norm's range.
    untrained = dataset(monthly, 36, 12, ["A", "B", "C"], seed=6)
    untrained.loc[(untrained.location == "C") & (untrained.index % 36 < 29), "disease_cases"] = float("nan")
    untrained.loc[(untrained.location == "C") & (untrained.index % 36 == 33), "disease_cases"] = 500.0
    run("untrained", untrained, {"n_periods": 3, "n_splits": 3, "stride": 2, "n_retrain": 1})
    # A's gap covers 2020W52 to 2021W01, across 2020W53.
    run(
        "messy_weekly",
        messy(dataset(weekly, 130, 52, ["A", "B", "C"], seed=5), [weekly(i) for i in range(130)], range(103, 106), 10, 120, 60, seed=5),
        {"n_periods": 4, "n_splits": 5, "stride": 1, "n_retrain": 2},
    )
