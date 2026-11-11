//! STL: seasonal-trend decomposition by LOESS (R.B. Cleveland, W.S. Cleveland, McRae and
//! Terpenning, 1990), a port of the netlib code behind R's `stats::stl` (R's `stl.c`) and
//! statsmodels' `STL`. The robustness iterations are not ported (`robust = FALSE`, the default in
//! both).

use crate::{Error, Result};

/// STL settings: the spans, local degrees and jumps of the seasonal, trend and low-pass smoothers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stl {
    /// The seasonal period (`np`).
    pub period: usize,
    /// Seasonal smoother span (`s.window`), odd.
    pub s_window: usize,
    pub s_degree: usize,
    pub t_window: usize,
    pub t_degree: usize,
    pub l_window: usize,
    pub l_degree: usize,
    pub s_jump: usize,
    pub t_jump: usize,
    pub l_jump: usize,
    /// Inner iterations.
    pub inner: usize,
}

/// R's `nextodd`: round, then up to the next odd number.
fn next_odd(x: f64) -> usize {
    let x = x.round() as usize;
    if x.is_multiple_of(2) { x + 1 } else { x }
}

impl Stl {
    /// R's `stl(x, s.window)` defaults: `s.degree = 0`, `t.window =
    /// nextodd(ceiling(1.5 * period / (1 - 1.5 / s.window)))`, `l.window = nextodd(period)`, jumps of a
    /// tenth of each span, two inner iterations.
    pub fn r(period: usize, s_window: usize) -> Self {
        let t_window = next_odd((1.5 * period as f64 / (1.0 - 1.5 / s_window as f64)).ceil());
        let l_window = next_odd(period as f64);
        let jump = |w: usize| w.div_ceil(10);
        Self {
            period,
            s_window,
            s_degree: 0,
            t_window,
            t_degree: 1,
            l_window,
            l_degree: 1,
            s_jump: jump(s_window),
            t_jump: jump(t_window),
            l_jump: jump(l_window),
            inner: 2,
        }
    }

    /// statsmodels' `STL(endog, period, seasonal)` defaults: degree 1 everywhere, jumps of 1, the
    /// trend span the next odd integer from `ceil(1.5 * period / (1 - 1.5 / seasonal))`, the
    /// low-pass span the next odd integer above `period`, five inner iterations.
    pub fn statsmodels(period: usize, seasonal: usize) -> Self {
        let mut t_window = (1.5 * period as f64 / (1.0 - 1.5 / seasonal as f64)).ceil() as usize;
        t_window += usize::from(t_window.is_multiple_of(2));
        let mut l_window = period + 1;
        l_window += usize::from(l_window.is_multiple_of(2));
        Self {
            period,
            s_window: seasonal,
            s_degree: 1,
            t_window,
            t_degree: 1,
            l_window,
            l_degree: 1,
            s_jump: 1,
            t_jump: 1,
            l_jump: 1,
            inner: 5,
        }
    }

    /// The seasonal and trend components of `y` (no missing values; more than two periods).
    pub fn fit(&self, y: &[f64]) -> Result<Decomposition> {
        let n = y.len();
        if self.period < 2 || n <= 2 * self.period {
            return Err(Error::invalid(
                "STL needs a period of at least 2 and more than two full periods",
            ));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid("STL does not support missing values"));
        }
        let odd = |w: usize| {
            let w = w.max(3);
            if w.is_multiple_of(2) { w + 1 } else { w }
        };
        let s = Stl {
            s_window: odd(self.s_window),
            t_window: odd(self.t_window),
            l_window: odd(self.l_window),
            ..*self
        };
        let mut seasonal = vec![0.0; n];
        let mut trend = vec![0.0; n];
        s.step(y, &mut seasonal, &mut trend);
        Ok(Decomposition { seasonal, trend })
    }

    /// `stlstp`: the inner loop.
    fn step(&self, y: &[f64], season: &mut [f64], trend: &mut [f64]) {
        let n = y.len();
        let np = self.period;
        let n2p = n + 2 * np;
        let mut w1 = vec![0.0; n2p];
        let mut w2 = vec![0.0; n2p];
        let mut w3 = vec![0.0; n2p];
        let mut w4 = vec![0.0; n2p];
        let mut w5 = vec![0.0; n2p];
        for _ in 0..self.inner {
            for i in 0..n {
                w1[i] = y[i] - trend[i];
            }
            self.seasonal_smooth(&w1[..n], &mut w2, &mut w3, &mut w4, &mut w5);
            fts(&w2, n2p, np, &mut w3, &mut w1);
            ess(
                &w3[..n],
                self.l_window,
                self.l_degree,
                self.l_jump,
                &mut w1[..n],
                &mut w5,
            );
            for i in 0..n {
                season[i] = w2[np + i] - w1[i];
            }
            for i in 0..n {
                w1[i] = y[i] - season[i];
            }
            ess(
                &w1[..n],
                self.t_window,
                self.t_degree,
                self.t_jump,
                trend,
                &mut w3,
            );
        }
    }

    /// `stlss`: smooth each cycle-subseries, extended by one value at each end.
    fn seasonal_smooth(
        &self,
        y: &[f64],
        season: &mut [f64],
        w1: &mut [f64],
        w2: &mut [f64],
        w4: &mut [f64],
    ) {
        let n = y.len();
        let np = self.period;
        let ns = self.s_window;
        for j in 0..np {
            let k = (n - (j + 1)) / np + 1;
            for i in 0..k {
                w1[i] = y[i * np + j];
            }
            ess(&w1[..k], ns, self.s_degree, self.s_jump, &mut w2[1..=k], w4);
            let nright = ns.min(k);
            w2[0] = est(&w1[..k], ns, self.s_degree, 0.0, 1, nright, w4).unwrap_or(w2[1]);
            let nleft = (k as i64 - ns as i64 + 1).max(1) as usize;
            w2[k + 1] =
                est(&w1[..k], ns, self.s_degree, (k + 1) as f64, nleft, k, w4).unwrap_or(w2[k]);
            for m in 0..k + 2 {
                season[m * np + j] = w2[m];
            }
        }
    }
}

/// The seasonal and trend components.
#[derive(Debug, Clone, PartialEq)]
pub struct Decomposition {
    pub seasonal: Vec<f64>,
    pub trend: Vec<f64>,
}

/// `x * x * x` as R's `R_pow_di(x, 3)` computes it.
fn cube(x: f64) -> f64 {
    x * (x * x)
}

/// `stlest`: the local fit at `xs` (1-based position) over `nleft..=nright`.
fn est(
    y: &[f64],
    len: usize,
    ideg: usize,
    xs: f64,
    nleft: usize,
    nright: usize,
    w: &mut [f64],
) -> Option<f64> {
    let n = y.len();
    let range = n as f64 - 1.0;
    let mut h = (xs - nleft as f64).max(nright as f64 - xs);
    if len > n {
        h += ((len - n) / 2) as f64;
    }
    let h9 = h * 0.999;
    let h1 = h * 0.001;
    let mut a = 0.0;
    for j in nleft - 1..nright {
        let r = ((j + 1) as f64 - xs).abs();
        if r <= h9 {
            w[j] = if r <= h1 {
                1.0
            } else {
                cube(1.0 - cube(r / h))
            };
            a += w[j];
        } else {
            w[j] = 0.0;
        }
    }
    if a <= 0.0 {
        return None;
    }
    for wj in &mut w[nleft - 1..nright] {
        *wj /= a;
    }
    if h > 0.0 && ideg > 0 {
        let mut a = 0.0;
        for j in nleft - 1..nright {
            a += w[j] * (j + 1) as f64;
        }
        let mut b = xs - a;
        let mut c = 0.0;
        for j in nleft - 1..nright {
            let d = (j + 1) as f64 - a;
            c += w[j] * (d * d);
        }
        if c.sqrt() > range * 0.001 {
            b /= c;
            for j in nleft - 1..nright {
                w[j] *= b * ((j + 1) as f64 - a) + 1.0;
            }
        }
    }
    let mut ys = 0.0;
    for j in nleft - 1..nright {
        ys += w[j] * y[j];
    }
    Some(ys)
}

/// `stless`: LOESS of `y` with span `len`, fitted every `njump` points and interpolated between.
fn ess(y: &[f64], len: usize, ideg: usize, njump: usize, ys: &mut [f64], res: &mut [f64]) {
    let n = y.len();
    if n < 2 {
        if let (Some(o), Some(v)) = (ys.first_mut(), y.first()) {
            *o = *v;
        }
        return;
    }
    let newnj = njump.min(n - 1).max(1);
    let (mut nleft, mut nright);
    if len >= n {
        nleft = 1;
        nright = n;
        for i in (0..n).step_by(newnj) {
            ys[i] = est(y, len, ideg, (i + 1) as f64, nleft, nright, res).unwrap_or(y[i]);
        }
    } else {
        let nsh = len.div_ceil(2);
        if newnj == 1 {
            nleft = 1;
            nright = len;
            for i in 0..n {
                if i + 1 > nsh && nright != n {
                    nleft += 1;
                    nright += 1;
                }
                ys[i] = est(y, len, ideg, (i + 1) as f64, nleft, nright, res).unwrap_or(y[i]);
            }
        } else {
            nleft = 1;
            nright = len;
            for i in (0..n).step_by(newnj) {
                if i + 1 < nsh {
                    nleft = 1;
                    nright = len;
                } else if i >= n - nsh {
                    nleft = n - len + 1;
                    nright = n;
                } else {
                    nleft = i + 1 - nsh + 1;
                    nright = len + i + 1 - nsh;
                }
                ys[i] = est(y, len, ideg, (i + 1) as f64, nleft, nright, res).unwrap_or(y[i]);
            }
        }
    }
    if newnj != 1 {
        let mut i = 0;
        while i < n - newnj {
            let delta = (ys[i + newnj] - ys[i]) / newnj as f64;
            for j in i + 1..i + newnj {
                ys[j] = ys[i] + delta * (j - i) as f64;
            }
            i += newnj;
        }
        let k = (n - 1) / newnj * newnj;
        if k != n - 1 {
            ys[n - 1] = est(y, len, ideg, n as f64, nleft, nright, res).unwrap_or(y[n - 1]);
            let delta = (ys[n - 1] - ys[k]) / (n - 1 - k) as f64;
            for j in k + 1..n - 1 {
                ys[j] = ys[k] + delta * (j - k) as f64;
            }
        }
    }
}

/// `stlma`: running mean of span `len` over the first `n` values.
fn ma(x: &[f64], n: usize, len: usize, ave: &mut [f64]) {
    let flen = len as f64;
    let mut v: f64 = x[..len].iter().sum();
    ave[0] = v / flen;
    let newn = n - len + 1;
    let (mut k, mut m) = (len, 0);
    for a in ave.iter_mut().take(newn).skip(1) {
        v += x[k] - x[m];
        *a = v / flen;
        k += 1;
        m += 1;
    }
}

/// `stlfts`: moving averages of spans `np`, `np` and 3 (the low-pass filter).
fn fts(x: &[f64], n: usize, np: usize, trend: &mut [f64], work: &mut [f64]) {
    ma(x, n, np, trend);
    ma(trend, n - np + 1, np, work);
    ma(work, n - 2 * np + 2, 3, trend);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let r = Stl::r(12, 11);
        assert_eq!(
            (r.t_window, r.l_window, r.s_jump, r.t_jump, r.l_jump),
            (21, 13, 2, 3, 2)
        );
        let r = Stl::r(52, 11);
        assert_eq!(
            (r.t_window, r.l_window, r.t_jump, r.l_jump),
            (91, 53, 10, 6)
        );
        let s = Stl::statsmodels(12, 11);
        assert_eq!((s.t_window, s.l_window), (21, 13));
        let s = Stl::statsmodels(52, 11);
        assert_eq!((s.t_window, s.l_window), (91, 53));
    }

    #[test]
    fn recovers_a_pure_season() {
        let y: Vec<f64> = (0..60)
            .map(|i| 5.0 + (2.0 * std::f64::consts::PI * f64::from(i) / 12.0).sin())
            .collect();
        let d = Stl::r(12, 11).fit(&y).unwrap();
        for (i, v) in y.iter().enumerate() {
            assert!((d.seasonal[i] + d.trend[i] - v).abs() < 0.1);
        }
        assert!(Stl::r(12, 11).fit(&y[..24]).is_err());
    }
}
