use std::sync::atomic::{AtomicUsize, Ordering};

use chapkit_dataframe::{Column, DataFrame, cols};

use crate::{Error, Result};

/// The default of [`max_prediction_cells`]: 250 million cells, about 2 GB of samples.
pub const DEFAULT_MAX_PREDICTION_CELLS: usize = 250_000_000;

static MAX_PREDICTION_CELLS: AtomicUsize = AtomicUsize::new(DEFAULT_MAX_PREDICTION_CELLS);

/// The most cells (future rows times samples) [`prediction_frame`] and
/// [`predict_by_location`](crate::locations::predict_by_location) build; a larger forecast is an
/// error before the model is asked for it. The service sets it from
/// `CHAPKIT_MAX_PREDICTION_CELLS`.
pub fn max_prediction_cells() -> usize {
    MAX_PREDICTION_CELLS.load(Ordering::Relaxed)
}

/// Set [`max_prediction_cells`] for the process.
pub fn set_max_prediction_cells(cells: usize) {
    MAX_PREDICTION_CELLS.store(cells, Ordering::Relaxed);
}

/// The most samples a forecast has, whatever its row count: every sample is a column, allocated even for
/// an empty future. The model packages cap `n_samples` at the same value; chap-core asks for 100 to 1,000.
pub const MAX_PREDICTION_SAMPLES: usize = 100_000;

/// Refuse a forecast of `rows` future rows with `n_samples` samples each over
/// [`max_prediction_cells`], or with more than [`MAX_PREDICTION_SAMPLES`] samples.
pub(crate) fn check_prediction_cells(rows: usize, n_samples: usize) -> Result<()> {
    if n_samples > MAX_PREDICTION_SAMPLES {
        return Err(Error::invalid(format!(
            "n_samples is {n_samples}, more than the limit of {MAX_PREDICTION_SAMPLES}"
        )));
    }
    let (cells, max) = (rows.saturating_mul(n_samples), max_prediction_cells());
    if cells > max {
        return Err(Error::invalid(format!(
            "the forecast would have {cells} cells ({rows} future rows x {n_samples} samples), \
             more than the limit of {max} (CHAPKIT_MAX_PREDICTION_CELLS)"
        )));
    }
    Ok(())
}

/// Build a chapkit prediction frame: `time_period`, `location`, `sample_0..n_samples`,
/// one output row per row of `future`.
///
/// `fill(row, samples)` is called once per future row and writes that row's samples.
pub fn prediction_frame(
    future: &DataFrame,
    n_samples: usize,
    mut fill: impl FnMut(usize, &mut [f64]) -> Result<()>,
) -> Result<DataFrame> {
    if n_samples == 0 {
        return Err(Error::invalid("n_samples must be at least 1"));
    }
    let n = future.n_rows();
    check_prediction_cells(n, n_samples)?;
    let mut samples: Vec<Vec<Option<f64>>> =
        (0..n_samples).map(|_| Vec::with_capacity(n)).collect();
    let mut buf = vec![0.0; n_samples];
    for row in 0..n {
        fill(row, &mut buf)?;
        for (col, v) in samples.iter_mut().zip(&buf) {
            col.push(Some(*v));
        }
    }
    let mut out = DataFrame::new();
    out.push_column(
        cols::TIME_PERIOD,
        future.require(cols::TIME_PERIOD)?.clone(),
    )?;
    out.push_column(cols::LOCATION, future.require(cols::LOCATION)?.clone())?;
    for (i, col) in samples.into_iter().enumerate() {
        out.push_column(cols::sample(i), Column::Float(col))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Refused before anything is allocated; 100,000 locations x 12 periods x 100 samples fit.
    #[test]
    fn prediction_cells_are_bounded() {
        let future = DataFrame::from_columns(vec![
            (cols::TIME_PERIOD, Column::from_strs(["2024-01"])),
            (cols::LOCATION, Column::from_strs(["A"])),
        ])
        .unwrap();
        let rows = DEFAULT_MAX_PREDICTION_CELLS / MAX_PREDICTION_SAMPLES + 1;
        let e = check_prediction_cells(rows, MAX_PREDICTION_SAMPLES)
            .unwrap_err()
            .to_string();
        assert!(e.contains("CHAPKIT_MAX_PREDICTION_CELLS"), "{e}");
        let e = crate::locations::predict_by_location(&future, usize::MAX, |_, _| Ok(vec![]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("more than the limit"), "{e}");
        assert!(check_prediction_cells(100_000 * 12, 100).is_ok());
    }

    /// An empty future still allocates one column per sample, so the samples are bounded on their own.
    #[test]
    fn empty_future_bounds_samples() {
        let future = DataFrame::from_columns(vec![
            (cols::TIME_PERIOD, Column::from_strs(Vec::<&str>::new())),
            (cols::LOCATION, Column::from_strs(Vec::<&str>::new())),
        ])
        .unwrap();
        let e = prediction_frame(&future, usize::MAX, |_, _| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(e.contains("n_samples"), "{e}");
        let out = prediction_frame(&future, 3, |_, _| Ok(())).unwrap();
        assert_eq!((out.n_rows(), out.n_cols()), (0, 5));
    }
}
