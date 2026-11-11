//! fable's `train_arima` (fable 0.5.0): the unit-root tests for `D` and `d`, the stepwise search
//! over the remaining orders by an information criterion, and the re-estimation of the best
//! approximate (CSS) fits by CSS-ML.

use serde::{Deserialize, Serialize};

use super::{Arima, ArimaSpec, Method, fit_spec, roots_ok, stats};
use crate::unitroot::{diff, na_contiguous, ndiffs, nsdiffs};
use crate::{Error, Result};

/// The information criterion the search minimises.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Ic {
    #[default]
    Aicc,
    Aic,
    Bic,
}

/// fable's `ARIMA()` search settings; [`AutoArima::new`] gives fable's defaults.
///
/// Known differences between fable and R's `forecast::auto.arima` / statsforecast's `AutoARIMA`,
/// which this follows fable in: `D` from the STL seasonal strength (>= 0.64) rather than the OCSB
/// or Canova-Hansen test; `d` as the largest number of differences whose predecessor fails KPSS
/// rather than differencing until KPSS passes; up to 94 models in the stepwise search, starting
/// from (2,d,2)(1,D,1), the null, AR and MA models and moving to every model within squared
/// distance 2; `p + q + P + Q <= 6` and `constant + d + D <= 2`; candidate models with a root
/// within 1.01 of the unit circle rejected; fable's `sigma2` (residual sum of squares over
/// `nstar - npar + 1`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutoArima {
    /// Seasonal period (1 for none).
    pub period: usize,
    pub p: Vec<usize>,
    pub d: Vec<usize>,
    pub q: Vec<usize>,
    pub seasonal_p: Vec<usize>,
    pub seasonal_d: Vec<usize>,
    pub seasonal_q: Vec<usize>,
    pub p_init: usize,
    pub q_init: usize,
    pub seasonal_p_init: usize,
    pub seasonal_q_init: usize,
    /// Whether to try models with and without fable's constant.
    pub constant: Vec<bool>,
    /// `p + q + P + Q` at most this.
    pub max_order: usize,
    pub ic: Ic,
    pub stepwise: bool,
    pub greedy: bool,
    /// CSS during the search, CSS-ML for the best; `None`: when the series is longer than 150 or
    /// the period above 12.
    pub approximation: Option<bool>,
    /// KPSS level for `d`.
    pub kpss_alpha: f64,
    /// Seasonal strength from which `D = 1`.
    pub seasonal_strength: f64,
}

impl AutoArima {
    /// fable's defaults for a seasonal period: `pdq(0:5, 0:2, 0:5) + PDQ(0:2, 0:1, 0:2)`, with and
    /// without a constant, AICc, stepwise and greedy.
    pub fn new(period: usize) -> Self {
        Self {
            period,
            p: (0..=5).collect(),
            d: (0..=2).collect(),
            q: (0..=5).collect(),
            seasonal_p: (0..=2).collect(),
            seasonal_d: (0..=1).collect(),
            seasonal_q: (0..=2).collect(),
            p_init: 2,
            q_init: 2,
            seasonal_p_init: 1,
            seasonal_q_init: 1,
            constant: vec![true, false],
            max_order: 6,
            ic: Ic::Aicc,
            stepwise: true,
            greedy: true,
            approximation: None,
            kpss_alpha: 0.05,
            seasonal_strength: 0.64,
        }
    }

    /// One model, every order given (no search, no tests).
    pub(crate) fn exact(spec: ArimaSpec) -> Self {
        Self {
            period: spec.period,
            p: vec![spec.p],
            d: vec![spec.d],
            q: vec![spec.q],
            seasonal_p: vec![spec.seasonal_p],
            seasonal_d: vec![spec.seasonal_d],
            seasonal_q: vec![spec.seasonal_q],
            p_init: spec.p,
            q_init: spec.q,
            seasonal_p_init: spec.seasonal_p,
            seasonal_q_init: spec.seasonal_q,
            constant: vec![spec.constant],
            ..Self::new(spec.period)
        }
    }
}

/// Choose and fit an ARIMA model for `y` (`NaN` for missing) as fable's `ARIMA(y)` does.
pub fn auto_arima(y: &[f64], settings: &AutoArima) -> Result<Arima> {
    super::check_series(y)?;
    if settings.period == 0 {
        return Err(Error::invalid("the seasonal period must be at least 1"));
    }
    train(y, settings, None, None, false)
}

/// A row of fable's `model_opts`: `(p, d, q, P, D, Q, constant)`.
type Row = [usize; 7];

fn spec_of(row: &Row, period: usize) -> ArimaSpec {
    ArimaSpec {
        p: row[0],
        d: row[1],
        q: row[2],
        seasonal_p: row[3],
        seasonal_d: row[4],
        seasonal_q: row[5],
        period,
        constant: row[6] == 1,
    }
}

/// `x[which.min(abs(x - init))]`.
fn nearest(x: &[usize], init: usize) -> usize {
    x.iter()
        .copied()
        .min_by_key(|v| v.abs_diff(init))
        .unwrap_or(init)
}

fn unique(rows: &[Row], col: usize) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for r in rows {
        if !out.contains(&r[col]) {
            out.push(r[col]);
        }
    }
    out
}

/// R's `is.constant`: every value equal to the first (true for an empty series; false with a
/// missing value).
fn is_constant(x: &[f64]) -> bool {
    x.first().is_none_or(|f| x.iter().all(|v| v == f))
}

struct Search<'a> {
    y: &'a [f64],
    period: usize,
    ic: Ic,
    method: Method,
    offset: f64,
    /// fable wraps `arima` in `possibly()` when there was more than one candidate.
    tolerant: bool,
    best: Option<Arima>,
    best_ic: f64,
}

impl Search<'_> {
    /// `compare_arima`: fit one row and return its criterion (`Inf` when it failed or was rejected).
    fn compare(&mut self, row: &Row, fixed: Option<&[f64]>, check_roots: bool) -> Result<f64> {
        let spec = spec_of(row, self.period);
        let fit = match fit_spec(self.y, &spec, self.method, fixed, self.offset) {
            Ok(fit) => Some(fit),
            Err(e) if !self.tolerant => return Err(e),
            Err(_) => None,
        };
        let fit = fit.filter(|f| !check_roots || roots_ok(f.phi(), f.theta()));
        let Some(fit) = fit else {
            return Ok(f64::INFINITY);
        };
        let ic = match self.ic {
            Ic::Aicc => fit.aicc,
            Ic::Aic => fit.aic,
            Ic::Bic => fit.bic,
        };
        let ic = if ic.is_nan() { f64::INFINITY } else { ic };
        if ic < self.best_ic || self.best.is_none() {
            self.best = Some(fit);
            self.best_ic = ic;
        }
        Ok(ic)
    }
}

/// R's `order()` of `keys` (ascending, stable).
fn order_by<K: PartialOrd>(n: usize, key: impl Fn(usize) -> K) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|a, b| {
        key(*a)
            .partial_cmp(&key(*b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// `which.min`, ignoring `None`: the first smallest.
fn which_min(v: &[Option<f64>]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, x) in v.iter().enumerate() {
        if let Some(x) = x
            && best.is_none_or(|(_, b)| *x < b)
        {
            best = Some((i, *x));
        }
    }
    best.map(|(i, _)| i)
}

/// fable's `train_arima` for a univariate series without regressors. `fixed`: every coefficient
/// given (refit without re-estimation). `method`: forced (no approximation).
#[allow(clippy::too_many_lines)]
pub(crate) fn train(
    y: &[f64],
    s: &AutoArima,
    fixed: Option<&[f64]>,
    method: Option<Method>,
    refit: bool,
) -> Result<Arima> {
    let n = y.len();
    let period = s.period;
    // fable's specials drop orders above a third of the data (P and Q: of the data in periods).
    // In a refit they filter the default ranges, which always keep 0, and the model's orders then
    // replace them: no limit applies.
    let fits = |v: &usize, limit: usize| refit || *v <= limit;
    let p: Vec<usize> = s.p.iter().copied().filter(|v| fits(v, n / 3)).collect();
    let q: Vec<usize> = s.q.iter().copied().filter(|v| fits(v, n / 3)).collect();
    let (sp, sd, sq) = if period <= 1 {
        (vec![0], vec![0], vec![0])
    } else {
        let sp: Vec<usize> = s
            .seasonal_p
            .iter()
            .copied()
            .filter(|v| fits(v, n / 3 / period))
            .collect();
        let sq: Vec<usize> = s
            .seasonal_q
            .iter()
            .copied()
            .filter(|v| fits(v, n / 3 / period))
            .collect();
        if sp.is_empty() || sq.is_empty() {
            return Err(Error::invalid(
                "not enough data to estimate a model with those options of P and Q",
            ));
        }
        (sp, s.seasonal_d.clone(), sq)
    };
    if p.is_empty() || q.is_empty() || s.d.is_empty() || sd.is_empty() || s.constant.is_empty() {
        return Err(Error::invalid("not enough data for the orders allowed"));
    }
    let p_init = nearest(&p, s.p_init);
    let q_init = nearest(&q, s.q_init);
    let sp_init = nearest(&sp, s.seasonal_p_init);
    let sq_init = nearest(&sq, s.seasonal_q_init);

    // expand.grid: p fastest, then d, q, P, D, Q, constant.
    let mut rows: Vec<Row> = Vec::new();
    for &c in &s.constant {
        for &qq in &sq {
            for &dd in &sd {
                for &pp in &sp {
                    for &q1 in &q {
                        for &d1 in &s.d {
                            for &p1 in &p {
                                rows.push([p1, d1, q1, pp, dd, qq, usize::from(c)]);
                            }
                        }
                    }
                }
            }
        }
    }
    let tolerant = rows.len() > 1;
    if tolerant {
        rows.retain(|r| r[0] + r[2] + r[3] + r[5] <= s.max_order && r[6] + r[1] + r[4] <= 2);
        if rows.is_empty() {
            return Err(Error::invalid(
                "there are no ARIMA models to choose from after imposing the order constraint",
            ));
        }
    }

    let mut seas_d_big = unique(&rows, 4);
    if seas_d_big.len() > 1 {
        let chosen = nsdiffs(na_contiguous(y), period, s.seasonal_strength, &seas_d_big)
            .ok_or_else(|| {
                Error::estimation("could not find an appropriate number of seasonal differences")
            })?;
        rows.retain(|r| r[4] == chosen);
        seas_d_big = vec![chosen];
    }
    let big_d = *seas_d_big.first().ok_or_else(|| {
        Error::estimation("could not find an appropriate number of seasonal differences")
    })?;
    let dx = if big_d == 0 {
        y.to_vec()
    } else {
        diff(y, period, big_d)
    };
    let mut seas_d = unique(&rows, 1);
    if seas_d.len() > 1 {
        let chosen = ndiffs(na_contiguous(&dx), s.kpss_alpha, &seas_d).ok_or_else(|| {
            Error::estimation("could not find an appropriate number of non-seasonal differences")
        })?;
        rows.retain(|r| r[1] == chosen);
        seas_d = vec![chosen];
    }
    let small_d = *seas_d.first().ok_or_else(|| {
        Error::estimation("could not find an appropriate number of non-seasonal differences")
    })?;
    // fable differences at the seasonal lag here (`diff(x, lag = period, differences = d)`).
    let dx = if small_d == 0 {
        y.to_vec()
    } else {
        diff(y, period.max(1), small_d)
    };

    let (method, mut approximation) = match method {
        Some(m) => (m, false),
        None => {
            let approximation = s
                .approximation
                .unwrap_or((n > 150 || period > 12) && rows.len() > 1);
            (
                if approximation {
                    Method::Css
                } else {
                    Method::CssMl
                },
                approximation,
            )
        }
    };
    let mut offset = 0.0;
    if method == Method::Css {
        // fable: -2 * loglik - NROW(data) * log(sigma2), where `data` is not a local variable and
        // resolves to the function `utils::data` (NROW of which is 1).
        let include_mean = s.constant.iter().all(|c| *c);
        let fit = stats::arima(
            y,
            [0, 0, 0, 0, period, small_d, 0],
            None,
            include_mean,
            None,
            Method::CssMl,
        )?;
        offset = -2.0 * fit.loglik - fit.sigma2.ln();
    }
    let constants: Vec<usize> = unique(&rows, 6);

    let mut search = Search {
        y,
        period,
        ic: s.ic,
        method,
        offset,
        tolerant,
        best: None,
        best_ic: f64::INFINITY,
    };
    let mut est_ic: Vec<Option<f64>> = vec![None; rows.len()];

    if is_constant(&dx) {
        rows.retain(|r| r[0] == 0 && r[2] == 0 && r[3] == 0 && r[5] == 0);
        if rows.is_empty() {
            return Err(Error::estimation(
                "no ARIMA models without AR or MA terms are available for constant differenced data",
            ));
        }
        approximation = false;
        let has_constant = (big_d > 0 && small_d == 0) || small_d < 2;
        if small_d >= 2 && big_d == 0 {
            return Err(Error::estimation(
                "data follows a simple polynomial and is unsuitable for ARIMA modelling",
            ));
        }
        let scale = if big_d > 0 && small_d == 0 {
            period as f64
        } else {
            1.0
        };
        let kept: Vec<f64> = dx
            .iter()
            .filter(|v| !v.is_nan())
            .map(|v| v / scale)
            .collect();
        let fixed_c = has_constant.then(|| vec![kept.iter().sum::<f64>() / kept.len() as f64]);
        let row = [0, small_d, 0, 0, big_d, 0, usize::from(has_constant)];
        let check = rows.len() > 1;
        search.compare(&row, fixed_c.as_deref(), check)?;
    } else if s.stepwise {
        let as_row = |v: [usize; 7]| -> Option<usize> { rows.iter().position(|r| *r == v) };
        let c1 = constants.first().copied().unwrap_or(0);
        let max_p = usize::from(p.iter().max().is_some_and(|m| *m > 0));
        let max_q = usize::from(q.iter().max().is_some_and(|m| *m > 0));
        let max_sp = usize::from(sp.iter().max().is_some_and(|m| *m > 0));
        let max_sq = usize::from(sq.iter().max().is_some_and(|m| *m > 0));
        let initial_opts = [
            [p_init, small_d, q_init, sp_init, big_d, sq_init, c1],
            [0, small_d, 0, 0, big_d, 0, c1],
            [max_p, small_d, 0, max_sp, big_d, 0, c1],
            [0, small_d, max_q, 0, big_d, max_sq, c1],
        ];
        let mut step_order: Vec<usize> = Vec::new();
        for o in initial_opts {
            if let Some(i) = as_row(o)
                && !step_order.contains(&i)
            {
                step_order.push(i);
            }
        }
        if step_order.is_empty() {
            let sq_dist = |r: &Row| {
                r[0].abs_diff(p_init).pow(2)
                    + r[2].abs_diff(q_init).pow(2)
                    + r[3].abs_diff(sp_init).pow(2)
                    + r[5].abs_diff(sq_init).pow(2)
            };
            let order = order_by(rows.len(), |i| {
                let r = &rows[i];
                (sq_dist(r), r[3], r[5], r[0], r[2])
            });
            step_order = order.into_iter().take(1).collect();
        }
        let mut best_ic = f64::INFINITY;
        let mut initial = true;
        let mut k = 0;
        let check = rows.len() > 1;
        while let Some(&first) = step_order.first() {
            if k >= 94 {
                break;
            }
            k += 1;
            let row = rows[first];
            let ic = search.compare(&row, fixed, check)?;
            est_ic[first] = Some(ic);
            let mut current: Option<Row> = None;
            let update_step;
            if s.greedy && !initial {
                update_step = ic < best_ic;
                if update_step {
                    best_ic = ic;
                    current = Some(row);
                }
            } else {
                update_step = step_order.len() == 1;
                if update_step {
                    best_ic = est_ic
                        .iter()
                        .flatten()
                        .copied()
                        .fold(f64::INFINITY, f64::min);
                    current = which_min(&est_ic).map(|i| rows[i]);
                }
            }
            if update_step && let Some(current) = current {
                initial = false;
                let dist: Vec<usize> = rows
                    .iter()
                    .map(|r| {
                        r.iter()
                            .zip(&current)
                            .map(|(a, b)| a.abs_diff(*b).pow(2))
                            .sum()
                    })
                    .collect();
                let near = dist.iter().filter(|d| **d <= 2).count();
                let order = order_by(rows.len(), |i| {
                    let r = &rows[i];
                    (dist[i], r[3], r[5], r[0], r[2])
                });
                step_order = order
                    .into_iter()
                    .take(near)
                    .filter(|i| est_ic[*i].is_none())
                    .collect();
            } else {
                step_order.remove(0);
            }
        }
    } else {
        let check = rows.len() > 1;
        for i in 0..rows.len() {
            let row = rows[i];
            est_ic[i] = Some(search.compare(&row, fixed, check)?);
        }
    }

    if approximation && search.best.is_some() {
        let approx_best = search.best.take();
        search.method = Method::CssMl;
        search.best_ic = f64::INFINITY;
        let evaluated = est_ic.iter().filter(|v| v.is_some()).count();
        let order = order_by(rows.len(), |i| {
            (est_ic[i].is_none(), est_ic[i].unwrap_or(0.0))
        });
        let order: Vec<usize> = order
            .into_iter()
            .filter(|i| est_ic[*i].is_some())
            .take(evaluated)
            .collect();
        let check = rows.len() > 1;
        let mut found = false;
        for i in order {
            let row = rows[i];
            if search.compare(&row, fixed, check)?.is_finite() {
                found = true;
                break;
            }
        }
        if !found {
            search.best = approx_best;
        }
    }
    search.best.ok_or_else(|| {
        Error::estimation(
            "could not find an appropriate ARIMA model (candidates with characteristic roots \
             near the unit circle are not selected)",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        assert_eq!(nearest(&[0, 1, 2, 3], 2), 2);
        assert_eq!(nearest(&[0, 1], 2), 1);
        assert!(is_constant(&[]) && is_constant(&[2.0, 2.0]) && !is_constant(&[1.0, 2.0]));
        assert!(!is_constant(&[1.0, f64::NAN]));
        assert_eq!(which_min(&[None, Some(3.0), Some(1.0), Some(1.0)]), Some(2));
        assert_eq!(order_by(4, |i| [3, 1, 2, 1][i]), vec![1, 3, 2, 0]);
    }
}
