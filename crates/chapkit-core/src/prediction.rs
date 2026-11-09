use chapkit_dataframe::{Column, DataFrame, cols};

use crate::{Error, Result};

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
