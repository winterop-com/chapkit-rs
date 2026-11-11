//! chap-core's `forecast_ahead`: train on the whole dataset and forecast the periods after its
//! last one, with the covariates of those periods from a weather provider.

use chapkit_core::{DataFrame, Error, Result, cols};

use crate::prepare::{Prepare, prepare};
use crate::split::Panel;
use crate::weather::{WeatherProvider, masked_future};

/// What a forecast ahead is made from.
#[derive(Debug, Clone)]
pub struct Ahead {
    /// The dataset as prepared ([`prepare`]): the model is trained on it and given it as history.
    pub historic: DataFrame,
    /// The covariates of the forecast periods from the weather provider, without the target.
    /// Rows by location, then period.
    pub future: DataFrame,
    /// The forecast periods, as [`chapkit_core::Period`] displays them.
    pub periods: Vec<String>,
}

/// Prepare `data` and build the future frame of the `periods` periods after its last period, as
/// chap-core's `forecast_with_predicted_weather`. A provider that looks ahead
/// ([`WeatherProvider::leaks_future_data`]) has nothing to read and is refused.
pub fn forecast_ahead(
    data: &DataFrame,
    periods: usize,
    options: &Prepare,
    weather: &dyn WeatherProvider,
) -> Result<Ahead> {
    if periods == 0 {
        return Err(Error::invalid("periods must be at least 1"));
    }
    if weather.leaks_future_data() {
        return Err(Error::invalid(format!(
            "weather provider {} reads the observed future and cannot forecast ahead",
            weather.id()
        )));
    }
    data.require(cols::LOCATION)?;
    let historic = prepare(data, options)?;
    let panel = Panel::new(&historic, options.fill_gaps)?;
    let last = *panel
        .periods
        .last()
        .ok_or_else(|| Error::invalid("the dataset has no rows"))?;
    let mut ahead = Vec::with_capacity(periods);
    let mut p = last;
    for _ in 0..periods {
        p = p
            .succ()
            .ok_or_else(|| Error::invalid(format!("cannot step past {p}: Sunday-start weeks")))?;
        ahead.push(p);
    }
    let historic_periods: Vec<_> = panel
        .row_period
        .iter()
        .filter_map(|&i| panel.periods.get(i).copied())
        .collect();
    let labels: Vec<String> = ahead.iter().map(ToString::to_string).collect();
    let future = masked_future(
        weather,
        &historic,
        &historic_periods,
        &labels,
        &ahead,
        &DataFrame::new(),
    )?;
    Ok(Ahead {
        historic,
        future,
        periods: labels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weather::{Climatology, Observed};
    use chapkit_core::Column;

    fn data() -> DataFrame {
        let (mut tp, mut loc, mut y, mut rain) = (vec![], vec![], vec![], vec![]);
        for l in ["B", "A"] {
            for m in 1..=12 {
                tp.push(format!("2024-{m:02}"));
                loc.push(l.to_owned());
                y.push(Some(f64::from(m)));
                rain.push(Some(f64::from(m) * 10.0));
            }
        }
        DataFrame::from_columns(vec![
            ("time_period", Column::from_strs(tp)),
            ("location", Column::from_strs(loc)),
            ("disease_cases", Column::Float(y)),
            ("rainfall", Column::Float(rain)),
        ])
        .unwrap()
    }

    #[test]
    fn forecasts_the_periods_after_the_last_with_climatology() {
        let a = forecast_ahead(&data(), 3, &Prepare::default(), &Climatology).unwrap();
        assert_eq!(a.periods, ["2025-01", "2025-02", "2025-03"]);
        assert_eq!(a.historic.n_rows(), 24);
        assert_eq!(a.future.n_rows(), 6);
        assert!(!a.future.has("disease_cases"));
        let loc = a.future.require("location").unwrap();
        assert_eq!(loc.str_at(0), Some("A"));
        // The climatology of January is the one January observed.
        assert_eq!(a.future.f64_column("rainfall").unwrap()[0], 10.0);
        assert_eq!(a.future.f64_column("rainfall").unwrap()[2], 30.0);
    }

    #[test]
    fn refuses_a_provider_that_looks_ahead() {
        let e = forecast_ahead(&data(), 3, &Prepare::default(), &Observed).unwrap_err();
        assert!(e.to_string().contains("cannot forecast ahead"), "{e}");
        assert!(forecast_ahead(&data(), 0, &Prepare::default(), &Climatology).is_err());
    }
}
