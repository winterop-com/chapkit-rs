//! Synthetic chap-style datasets, the Rust counterpart of chapkit's `TestDataGenerator`.
//!
//! Panel data (periods x locations) with the columns chap-core expects:
//! `time_period`, `location`, `disease_cases` (target), `population`, `rainfall`,
//! `mean_temperature`, then `feature_0..N`, the model's required covariates and any
//! extra covariates. Climate covariates follow an annual cycle with a per-location
//! phase, and `disease_cases` depends on the season and on rainfall, so models have
//! something real to learn. Used by `$generate-sample-data`, tests and benchmarks.

use serde_json::{Value, json};

use crate::rng::Rng;
use crate::{Column, DataFrame, Error, Period, Result};

/// Period resolution of generated data.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Resolution {
    /// `2020-01`, ...
    #[default]
    Monthly,
    /// `2020-W01`, ...
    Weekly,
}

impl Resolution {
    fn per_year(self) -> usize {
        match self {
            Resolution::Monthly => 12,
            Resolution::Weekly => 52,
        }
    }

    /// The labels of `count` periods from absolute period `start` (0 = the first of
    /// `start_year`). Weeks follow the ISO calendar (53 in some years), as [`Period::succ`] does.
    fn period_ids(self, start_year: i32, start: usize, count: usize) -> Result<Vec<String>> {
        let mut p = match self {
            Resolution::Monthly => Period::Month {
                year: start_year,
                month: 1,
            },
            Resolution::Weekly => Period::Week {
                year: start_year,
                week: 1,
                sunday: false,
            },
        };
        let mut out = Vec::with_capacity(count);
        for i in 0..start + count {
            if i >= start {
                out.push(match p {
                    Period::Month { year, month } => format!("{year}-{month:02}"),
                    Period::Week { year, week, .. } => format!("{year}-W{week:02}"),
                });
            }
            p = p
                .succ()
                .ok_or_else(|| Error::invalid(format!("no period after {p}")))?;
        }
        Ok(out)
    }
}

/// What to generate.
#[derive(Debug, Clone)]
pub struct Spec {
    /// Locations (`location_0..N`).
    pub num_locations: usize,
    /// Periods.
    pub num_periods: usize,
    /// Generic `feature_N` covariates.
    pub num_features: usize,
    /// Covariates the model requires (added after the features).
    pub required_covariates: Vec<String>,
    /// Covariates a config asks for (added after the required ones).
    pub additional_covariates: Vec<String>,
    /// Period resolution.
    pub resolution: Resolution,
    /// First year.
    pub start_year: i32,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            num_locations: 5,
            num_periods: 12,
            num_features: 3,
            required_covariates: Vec::new(),
            additional_covariates: Vec::new(),
            resolution: Resolution::Monthly,
            start_year: 2020,
        }
    }
}

struct LocationParams {
    population: i64,
    phase: f64,
    base_cases: f64,
    amplitude: f64,
}

/// Synthetic data generator. The same seed gives the same data.
pub struct Generator {
    rng: Rng,
}

impl Generator {
    /// A generator; `None` seeds from the clock (fresh data each time).
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: seed.map_or_else(Rng::from_time, Rng::new),
        }
    }

    fn columns(spec: &Spec) -> Vec<String> {
        let mut cols: Vec<String> = [
            "time_period",
            "location",
            "disease_cases",
            "population",
            "rainfall",
            "mean_temperature",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        cols.extend((0..spec.num_features).map(|i| format!("feature_{i}")));
        for c in spec
            .required_covariates
            .iter()
            .chain(&spec.additional_covariates)
        {
            if !cols.contains(c) {
                cols.push(c.clone());
            }
        }
        cols
    }

    fn params(&mut self, n: usize) -> Vec<LocationParams> {
        (0..n)
            .map(|_| LocationParams {
                population: self.rng.int_range(50_000, 1_500_000),
                phase: self.rng.unit(),
                base_cases: self.rng.uniform(20.0, 70.0),
                amplitude: self.rng.uniform(0.45, 0.7),
            })
            .collect()
    }

    /// Generate `count` periods starting at absolute period `start`, with targets.
    fn block(
        &mut self,
        spec: &Spec,
        params: &[LocationParams],
        start: usize,
        count: usize,
    ) -> Result<DataFrame> {
        let names = Self::columns(spec);
        let n_extra = names.len() - 6;
        let rows = count * spec.num_locations;
        let (mut tp, mut loc, mut cases, mut pop) = (
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
        );
        let (mut rain, mut temp) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
        let mut extra: Vec<Vec<f64>> = (0..n_extra).map(|_| Vec::with_capacity(rows)).collect();
        let per = spec.resolution.per_year();
        let labels = spec.resolution.period_ids(spec.start_year, start, count)?;
        for (p, label) in (start..start + count).zip(&labels) {
            // The season: 52 weeks a year, week 53 continuing the cycle.
            let year_fraction = (p % per) as f64 / per as f64;
            for (l, lp) in params.iter().enumerate() {
                tp.push(label.clone());
                loc.push(format!("location_{l}"));
                let season = (std::f64::consts::TAU * (year_fraction - lp.phase)).sin();
                let rainfall = (150.0 + 130.0 * season + self.rng.uniform(-25.0, 25.0)).max(0.0);
                let temperature = 24.0
                    + 7.0 * (std::f64::consts::TAU * (year_fraction - 0.08)).sin()
                    + self.rng.uniform(-1.5, 1.5);
                let c = lp.base_cases * (1.0 + lp.amplitude * season)
                    + 0.06 * rainfall
                    + self.rng.gauss(0.0, 4.0);
                cases.push(Some(c.round().max(0.0)));
                pop.push(Some(lp.population));
                rain.push(round2(rainfall));
                temp.push(round2(temperature));
                for (i, col) in extra.iter_mut().enumerate() {
                    let phase = (i + 1) as f64 / (n_extra + 1) as f64;
                    let v = 50.0
                        + 35.0 * (std::f64::consts::TAU * (year_fraction - phase)).sin()
                        + self.rng.uniform(-8.0, 8.0);
                    col.push(round2(v.clamp(0.0, 100.0)));
                }
            }
        }
        let mut df = DataFrame::new();
        df.push_column("time_period", Column::from_strs(tp))?;
        df.push_column("location", Column::from_strs(loc))?;
        df.push_column("disease_cases", Column::Float(cases))?;
        df.push_column("population", Column::Int(pop))?;
        df.push_column("rainfall", Column::from_f64(rain))?;
        df.push_column("mean_temperature", Column::from_f64(temp))?;
        for (name, col) in names.into_iter().skip(6).zip(extra) {
            df.push_column(name, Column::from_f64(col))?;
        }
        Ok(df)
    }

    /// Training data: `spec.num_periods` periods with observed `disease_cases`.
    pub fn training_data(&mut self, spec: &Spec) -> Result<DataFrame> {
        let params = self.params(spec.num_locations);
        self.block(spec, &params, 0, spec.num_periods)
    }

    /// Prediction data: `historic_periods` observed periods, then `spec.num_periods`
    /// future periods continuing the same calendar, with `disease_cases` null.
    pub fn prediction_data(
        &mut self,
        spec: &Spec,
        historic_periods: usize,
    ) -> Result<(DataFrame, DataFrame)> {
        let params = self.params(spec.num_locations);
        let historic = self.block(spec, &params, 0, historic_periods)?;
        let mut future = self.block(spec, &params, historic_periods, spec.num_periods)?;
        future.set_column("disease_cases", Column::Float(vec![None; future.n_rows()]))?;
        Ok((historic, future))
    }

    /// A GeoJSON FeatureCollection tiling `n` locations as a grid near Nairobi
    /// (`properties.id` matches `location_N`), with a `bbox`.
    pub fn geo(n: usize, points: bool) -> Value {
        let cols = (n as f64).sqrt().ceil().max(1.0) as usize;
        let rows = n.div_ceil(cols).max(1);
        let cell = 0.8;
        let (origin_lon, origin_lat) = (
            36.8 - cols as f64 * cell / 2.0,
            -1.3 - rows as f64 * cell / 2.0,
        );
        let features: Vec<Value> = (0..n)
            .map(|i| {
                let (x, y) = (origin_lon + (i % cols) as f64 * cell, origin_lat + (i / cols) as f64 * cell);
                let geometry = if points {
                    json!({"type": "Point", "coordinates": [x + cell / 2.0, y + cell / 2.0]})
                } else {
                    json!({"type": "Polygon", "coordinates": [[[x, y], [x + cell, y], [x + cell, y + cell], [x, y + cell], [x, y]]]})
                };
                json!({"type": "Feature", "geometry": geometry, "properties": {"id": format!("location_{i}")}})
            })
            .collect();
        let bbox = [
            origin_lon,
            origin_lat,
            origin_lon + cols as f64 * cell,
            origin_lat + rows as f64 * cell,
        ];
        json!({"type": "FeatureCollection", "features": features, "bbox": bbox})
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Period;

    #[test]
    fn training_data_has_chap_columns_and_shape() {
        let spec = Spec {
            required_covariates: vec!["humidity".into(), "rainfall".into()],
            ..Spec::default()
        };
        let df = Generator::new(Some(1)).training_data(&spec).unwrap();
        assert_eq!(df.n_rows(), 60);
        assert_eq!(
            df.names(),
            [
                "time_period",
                "location",
                "disease_cases",
                "population",
                "rainfall",
                "mean_temperature",
                "feature_0",
                "feature_1",
                "feature_2",
                "humidity"
            ]
        );
        let cases = df.f64_column("disease_cases").unwrap();
        assert!(cases.iter().all(|c| c.is_finite() && *c >= 0.0));
        assert_eq!(Generator::new(Some(1)).training_data(&spec).unwrap(), df);
    }

    #[test]
    fn prediction_future_continues_history() {
        let spec = Spec {
            num_periods: 3,
            resolution: Resolution::Weekly,
            ..Spec::default()
        };
        let (h, f) = Generator::new(Some(2)).prediction_data(&spec, 52).unwrap();
        let tp = |df: &DataFrame, r| {
            Period::parse(df.column("time_period").unwrap().str_at(r).unwrap()).unwrap()
        };
        assert_eq!(tp(&h, h.n_rows() - 1).to_string(), "2020W52");
        // 2020 has an ISO week 53.
        assert_eq!(tp(&f, 0).to_string(), "2020W53");
        assert!(
            f.f64_column("disease_cases")
                .unwrap()
                .iter()
                .all(|v| v.is_nan())
        );
    }

    /// 2020 has an ISO week 53: weekly labels follow the calendar, as `Period::succ` does.
    #[test]
    fn weekly_labels_include_iso_week_53() {
        let spec = Spec {
            num_locations: 1,
            num_periods: 54,
            resolution: Resolution::Weekly,
            ..Spec::default()
        };
        let df = Generator::new(Some(3)).training_data(&spec).unwrap();
        let tp = df.column("time_period").unwrap();
        let label = |r| tp.str_at(r).unwrap();
        assert_eq!(label(0), "2020-W01");
        assert_eq!(label(52), "2020-W53");
        assert_eq!(label(53), "2021-W01");
        for r in 1..df.n_rows() {
            assert_eq!(
                Period::parse(label(r - 1)).unwrap().succ(),
                Period::parse(label(r))
            );
        }
    }

    #[test]
    fn geo_grid() {
        let g = Generator::geo(5, false);
        assert_eq!(g["features"].as_array().unwrap().len(), 5);
        assert_eq!(g["features"][4]["properties"]["id"], "location_4");
    }
}
