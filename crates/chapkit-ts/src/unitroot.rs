//! Unit-root tests and the number of differences: the KPSS test (urca's `ur.kpss` as feasts calls
//! it, and statsmodels' `kpss` with its automatic lag), feasts' STL seasonal strength, and fable's
//! `unitroot_ndiffs` / `unitroot_nsdiffs`.

use crate::stl::Stl;

/// KPSS critical values for level stationarity (10%, 5%, 2.5%, 1%).
const KPSS_CRIT: [f64; 4] = [0.347, 0.463, 0.574, 0.739];
const KPSS_PVALS: [f64; 4] = [0.10, 0.05, 0.025, 0.01];

/// R's `diff(x, lag, differences)`: empty when `lag * differences >= x.len()`.
pub fn diff(x: &[f64], lag: usize, differences: usize) -> Vec<f64> {
    if lag * differences >= x.len() {
        return Vec::new();
    }
    let mut out = x.to_vec();
    for _ in 0..differences {
        out = out.windows(lag + 1).map(|w| w[lag] - w[0]).collect();
    }
    out
}

/// R's `na.contiguous`: the longest stretch without missing values (the first of equals).
pub fn na_contiguous(x: &[f64]) -> &[f64] {
    let (mut best, mut best_len) = (0, 0);
    let mut start = 0;
    for i in 0..=x.len() {
        if i == x.len() || x[i].is_nan() {
            if i - start > best_len {
                best = start;
                best_len = i - start;
            }
            start = i + 1;
        }
    }
    &x[best..best + best_len]
}

/// Linear interpolation of the KPSS p-value table, clamped at its ends (R's `approx(rule = 2)`).
fn kpss_pvalue(stat: f64) -> f64 {
    if stat.is_nan() {
        return f64::NAN;
    }
    if stat <= KPSS_CRIT[0] {
        return KPSS_PVALS[0];
    }
    if stat >= KPSS_CRIT[3] {
        return KPSS_PVALS[3];
    }
    let j = KPSS_CRIT
        .iter()
        .rposition(|c| *c <= stat)
        .unwrap_or(0)
        .min(2);
    let (x0, x1, y0, y1) = (
        KPSS_CRIT[j],
        KPSS_CRIT[j + 1],
        KPSS_PVALS[j],
        KPSS_PVALS[j + 1],
    );
    y0 + (y1 - y0) * ((stat - x0) / (x1 - x0))
}

fn kpss_stat(res: &[f64], lags: usize, s_hat: impl Fn(&[f64], usize) -> f64) -> f64 {
    let n = res.len() as f64;
    let mut s = 0.0;
    let mut eta = 0.0;
    for r in res {
        s += r;
        eta += s * s;
    }
    eta / (n * n) / s_hat(res, lags)
}

fn autocov(res: &[f64], lag: usize) -> f64 {
    res[lag..].iter().zip(res).map(|(a, b)| a * b).sum()
}

/// urca's `ur.kpss(x, type = "mu", lags = "short")` with feasts' p-value: `(statistic, p-value)`.
/// Missing values are dropped first.
pub fn kpss(x: &[f64]) -> (f64, f64) {
    let y: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = y.len();
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let lmax = (4.0 * (n as f64 / 100.0).powf(0.25)).trunc() as usize;
    let mean = y.iter().sum::<f64>() / n as f64;
    let res: Vec<f64> = y.iter().map(|v| v - mean).collect();
    let stat = kpss_stat(&res, lmax, |res, lmax| {
        let nf = res.len() as f64;
        let s2 = res.iter().map(|r| r * r).sum::<f64>() / nf;
        let mut acc = 0.0;
        for i in 1..=lmax.min(res.len().saturating_sub(1)) {
            acc += (1.0 - i as f64 / (lmax as f64 + 1.0)) * autocov(res, i);
        }
        s2 + 2.0 / nf * acc
    });
    (stat, kpss_pvalue(stat))
}

/// statsmodels' `kpss(x, regression="c", nlags="auto")`: `(statistic, p-value)`, or `None` where
/// statsmodels raises (the automatic lag is not a finite number).
pub fn kpss_statsmodels(x: &[f64]) -> Option<(f64, f64)> {
    let n = x.len();
    if n == 0 {
        return None;
    }
    let nf = n as f64;
    let mean = x.iter().sum::<f64>() / nf;
    let res: Vec<f64> = x.iter().map(|v| v - mean).collect();
    // Hobijn et al. (1998)'s automatic lag.
    let covlags = nf.powf(2.0 / 9.0) as usize;
    let mut s0 = res.iter().map(|r| r * r).sum::<f64>() / nf;
    let mut s1 = 0.0;
    for i in 1..=covlags.min(n.saturating_sub(1)) {
        let prod = autocov(&res, i) / (nf / 2.0);
        s0 += prod;
        s1 += i as f64 * prod;
    }
    let s_hat = s1 / s0;
    let gamma = 1.1447 * (s_hat * s_hat).powf(1.0 / 3.0);
    let lags = gamma * nf.powf(1.0 / 3.0);
    if !lags.is_finite() {
        return None;
    }
    let lags = (lags.trunc() as usize).min(n - 1);
    let stat = kpss_stat(&res, lags, |res, lags| {
        let mut s = res.iter().map(|r| r * r).sum::<f64>();
        for i in 1..=lags {
            s += 2.0 * autocov(res, i) * (1.0 - i as f64 / (lags as f64 + 1.0));
        }
        s / res.len() as f64
    });
    Some((stat, kpss_pvalue(stat)))
}

fn var(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)
}

/// feasts' `feat_stl(x, period)` seasonal strength: `max(0, min(1, 1 - var(remainder) /
/// var(remainder + seasonal)))` from two STL passes (`s.window = 11`, R's other defaults) over the
/// longest run without missing values. `None` when that run is at most two periods long (feasts
/// then fits no seasonal component); `NaN` for a constant series.
pub fn seasonal_strength(x: &[f64], period: usize) -> Option<f64> {
    let x = na_contiguous(x);
    if x.len() <= 2 * period || period < 2 {
        return None;
    }
    let stl = Stl::r(period, 11);
    let mut deseas = x.to_vec();
    let mut seas = vec![0.0; x.len()];
    let mut trend = vec![0.0; x.len()];
    for _ in 0..2 {
        for (d, s) in deseas.iter_mut().zip(&seas) {
            *d += s;
        }
        let fit = stl.fit(&deseas).ok()?;
        seas = fit.seasonal;
        trend = fit.trend;
        for (d, s) in deseas.iter_mut().zip(&seas) {
            *d -= s;
        }
    }
    let remainder: Vec<f64> = deseas.iter().zip(&trend).map(|(d, t)| d - t).collect();
    let with_season: Vec<f64> = remainder.iter().zip(&seas).map(|(r, s)| r + s).collect();
    let s = 1.0 - var(&remainder) / var(&with_season);
    // R's min/max propagate NaN.
    Some(if s.is_nan() { s } else { s.clamp(0.0, 1.0) })
}

/// fable's choice of `d` (`feasts::unitroot_ndiffs` with the KPSS p-value): among `differences`
/// (ascending), the largest `k` for which the series differenced `k - 1` times fails KPSS
/// (`p < alpha`); the smallest candidate when none does.
pub fn ndiffs(x: &[f64], alpha: f64, differences: &[usize]) -> Option<usize> {
    let kept: Vec<usize> = differences
        .iter()
        .copied()
        .filter(|&k| diff(x, 1, k).iter().any(|v| !v.is_nan()))
        .collect();
    let first = *kept.first()?;
    let mut best = first;
    for &k in kept.iter().skip(1) {
        let p = kpss(&diff(x, 1, k - 1)).1;
        if p < alpha {
            best = best.max(k);
        }
    }
    Some(best)
}

/// fable's choice of `D` (`feasts::unitroot_nsdiffs` with `ur_seasonal_strength(threshold)`):
/// among `differences` (ascending), the largest `k` for which the series seasonally differenced
/// `k - 1` times has a seasonal strength of at least `threshold`.
pub fn nsdiffs(x: &[f64], period: usize, threshold: f64, differences: &[usize]) -> Option<usize> {
    if period <= 1 {
        return differences.iter().copied().min();
    }
    let kept: Vec<usize> = differences
        .iter()
        .copied()
        .filter(|&k| diff(x, period, k).iter().any(|v| !v.is_nan()))
        .collect();
    let first = *kept.first()?;
    let mut best = first;
    for &k in kept.iter().skip(1) {
        // Seasonal enough to difference: a strength at or above the threshold. No seasonal
        // component (too short) or NaN counts as not seasonal.
        if seasonal_strength(&diff(x, period, k - 1), period).is_some_and(|s| s >= threshold) {
            best = best.max(k);
        }
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffs() {
        assert_eq!(diff(&[1.0, 4.0, 9.0, 16.0], 1, 2), vec![2.0, 2.0]);
        assert_eq!(diff(&[1.0, 2.0, 3.0, 5.0], 2, 1), vec![2.0, 3.0]);
        assert_eq!(diff(&[1.0, 2.0], 2, 1), Vec::<f64>::new());
        assert_eq!(
            na_contiguous(&[1.0, f64::NAN, 2.0, 3.0, f64::NAN]),
            &[2.0, 3.0]
        );
    }

    #[test]
    fn pvalue_table() {
        assert_eq!(kpss_pvalue(0.1), 0.10);
        assert_eq!(kpss_pvalue(2.0), 0.01);
        assert!((kpss_pvalue(0.463) - 0.05).abs() < 1e-15);
        assert!((kpss_pvalue(0.4050) - 0.075).abs() < 1e-12);
    }
}
