//! R's `stats::arima` (R >= 4.0, `SSinit = "Gardner1980"`, `transform.pars = TRUE`,
//! `kappa = 1e6`, `optim.method = "BFGS"` with R's default controls).

use super::kalman::{StateSpace, css, inv_trans, trans_pars, undo_pars};
use crate::optim::bfgs;
use crate::poly::{conv, roots, roots_outside};
use crate::{Error, Result};

/// How `stats::arima` estimates the coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum Method {
    /// Conditional sum of squares, then exact maximum likelihood from there (R's default).
    #[default]
    CssMl,
    /// Exact maximum likelihood.
    Ml,
    /// Conditional sum of squares only.
    Css,
}

/// What `stats::arima` returns, as far as fable and forecasting use it.
#[derive(Debug, Clone)]
pub(crate) struct Fit {
    /// `ar.., ma.., sar.., sma.., xreg..`.
    pub coef: Vec<f64>,
    /// Which coefficients were estimated (not fixed).
    pub mask: Vec<bool>,
    pub sigma2: f64,
    pub loglik: f64,
    /// `None` for CSS.
    pub aic: Option<f64>,
    pub residuals: Vec<f64>,
    /// `n.used`: observations minus the differencing.
    pub nobs: usize,
    pub model: StateSpace,
    pub method: Method,
}

/// The model orders: R's `arma` vector `(p, q, P, Q, period, d, D)`.
pub(crate) type Arma = [usize; 7];

/// `Delta`: the differencing polynomial `(1 - B)^d (1 - B^s)^D`, negated and without its leading 1.
pub(crate) fn delta(d: usize, sd: usize, period: usize) -> Vec<f64> {
    let mut delta = vec![1.0];
    for _ in 0..d {
        delta = conv(&delta, &[1.0, -1.0]);
    }
    for _ in 0..sd {
        let mut s = vec![0.0; period + 1];
        s[0] = 1.0;
        s[period] = -1.0;
        delta = conv(&delta, &s);
    }
    delta.iter().skip(1).map(|v| -v).collect()
}

/// `arCheck`: the AR polynomial has all roots outside the unit circle.
fn ar_check(ar: &[f64]) -> bool {
    let c: Vec<f64> = ar.iter().map(|v| -v).collect();
    roots_outside(&c, 1.0)
}

/// `maInvert`: reflect MA roots inside the unit circle to make the polynomial invertible.
fn ma_invert(ma: &[f64]) -> Vec<f64> {
    let q = ma.len();
    let q0 = ma.iter().rposition(|v| *v != 0.0).map_or(0, |i| i + 1);
    if q0 == 0 {
        return ma.to_vec();
    }
    let mut c = vec![1.0];
    c.extend_from_slice(&ma[..q0]);
    let mut rs = roots(&c);
    if rs.iter().all(|z| z.norm() >= 1.0) {
        return ma.to_vec();
    }
    if q0 == 1 {
        let mut out = vec![1.0 / ma[0]];
        out.resize(q, 0.0);
        return out;
    }
    for z in &mut rs {
        if z.norm() < 1.0 {
            *z = z.recip();
        }
    }
    // x <- 1; for (r in roots) x <- c(x, 0) - c(0, x) / r, in complex arithmetic.
    let mut x = vec![(1.0, 0.0)];
    for z in &rs {
        let inv = z.recip();
        let mut next = x.clone();
        next.push((0.0, 0.0));
        for (i, xi) in x.iter().enumerate() {
            // next[i + 1] -= x[i] / z
            let (re, im) = (xi.0 * inv.re - xi.1 * inv.im, xi.0 * inv.im + xi.1 * inv.re);
            next[i + 1].0 -= re;
            next[i + 1].1 -= im;
        }
        x = next;
    }
    let mut out: Vec<f64> = x.iter().skip(1).map(|v| v.0).collect();
    out.resize(q, 0.0);
    out
}

/// One regression column's least squares fit through the origin: `(coef, standard error)`, or
/// `None` when the column has no non-zero value among the rows used (`rank == 0`).
fn lm1(y: &[f64], x: &[f64]) -> Option<(f64, f64)> {
    let rows: Vec<(f64, f64)> = y
        .iter()
        .zip(x)
        .filter(|(y, x)| !y.is_nan() && !x.is_nan())
        .map(|(y, x)| (*y, *x))
        .collect();
    let sxx: f64 = rows.iter().map(|(_, x)| x * x).sum();
    if sxx == 0.0 {
        return None;
    }
    let coef = rows.iter().map(|(y, x)| x * y).sum::<f64>() / sxx;
    let rss: f64 = rows.iter().map(|(y, x)| (y - coef * x).powi(2)).sum();
    let df = rows.len() as f64 - 1.0;
    Some((coef, (rss / df / sxx).sqrt()))
}

/// `stats::arima(x, order = c(p, d, q), seasonal = list(order = c(P, D, Q), period), xreg,
/// include.mean, fixed, method)`. `fixed` uses `NaN` for the free coefficients. Only a single
/// regression column is supported (fable's constant, or the mean).
#[allow(clippy::too_many_lines)]
pub(crate) fn arima(
    x: &[f64],
    arma: Arma,
    xreg: Option<&[f64]>,
    include_mean: bool,
    fixed: Option<&[f64]>,
    method: Method,
) -> Result<Fit> {
    let n = x.len();
    let [mp, mq, msp, msq, period, d, sd] = arma;
    let narma = mp + mq + msp + msq;
    let delta = delta(d, sd, period);
    let nd = d + sd;
    let mut xreg: Option<Vec<f64>> = xreg.map(<[f64]>::to_vec);
    if let Some(x) = &xreg
        && x.len() != n
    {
        return Err(Error::invalid("lengths of 'x' and 'xreg' do not match"));
    }
    if include_mean && nd == 0 {
        if xreg.is_some() {
            return Err(Error::invalid("only one regression column is supported"));
        }
        xreg = Some(vec![1.0; n]);
    }
    let ncxreg = usize::from(xreg.is_some());
    let mut method = method;
    if method == Method::CssMl
        && (x.iter().any(|v| v.is_nan())
            || xreg.as_ref().is_some_and(|x| x.iter().any(|v| v.is_nan())))
    {
        method = Method::Ml;
    }
    let mut ncond = 0;
    if matches!(method, Method::Css | Method::CssMl) {
        ncond = d + sd * period + mp + period * msp;
    }
    let fixed: Vec<f64> = match fixed {
        Some(f) if f.len() != narma + ncxreg => {
            return Err(Error::invalid("wrong length for 'fixed'"));
        }
        Some(f) => f.to_vec(),
        None => vec![f64::NAN; narma + ncxreg],
    };
    let mask: Vec<bool> = fixed.iter().map(|v| v.is_nan()).collect();
    let no_optim = !mask.iter().any(|m| *m);
    let mut transform = !no_optim;
    if transform
        && (mask[..mp].iter().any(|m| !m) || mask[mp + mq..mp + mq + msp].iter().any(|m| !m))
    {
        transform = false;
    }
    let mut init0 = vec![0.0; narma];
    let mut parscale = vec![1.0; narma];
    let mut n_used = x.iter().filter(|v| !v.is_nan()).count() as isize - delta.len() as isize;
    if let Some(xr) = &xreg {
        let mut dx = x.to_vec();
        let mut dxr = xr.clone();
        if d > 0 {
            dx = crate::unitroot::diff(&dx, 1, d);
            dxr = crate::unitroot::diff(&dxr, 1, d);
        }
        if period > 1 && sd > 0 {
            dx = crate::unitroot::diff(&dx, period, sd);
            dxr = crate::unitroot::diff(&dxr, period, sd);
        }
        let fit = if dx.len() > 1 { lm1(&dx, &dxr) } else { None };
        let (coef, se) = match fit {
            Some(f) => f,
            None => lm1(x, xr).ok_or_else(|| Error::estimation("the regression is singular"))?,
        };
        n_used = x
            .iter()
            .zip(xr)
            .filter(|(a, b)| !a.is_nan() && !b.is_nan())
            .count() as isize
            - delta.len() as isize;
        init0.push(coef);
        parscale.push(10.0 * se);
    }
    if n_used <= 0 {
        return Err(Error::invalid("too few non-missing observations"));
    }
    let n_used = n_used as usize;
    let mut init = init0.clone();
    let mut coef = fixed.clone();
    let free = |v: &[f64]| -> Vec<f64> {
        v.iter()
            .zip(&mask)
            .filter(|(_, m)| **m)
            .map(|(v, _)| *v)
            .collect()
    };
    let parscale_free = free(&parscale);
    let place = |base: &[f64], p: &[f64]| -> Vec<f64> {
        let mut out = base.to_vec();
        let mut it = p.iter();
        for (o, m) in out.iter_mut().zip(&mask) {
            if *m && let Some(v) = it.next() {
                *o = *v;
            }
        }
        out
    };
    let regress = |par: &[f64]| -> Vec<f64> {
        match &xreg {
            Some(xr) => {
                let b = par[narma];
                x.iter().zip(xr).map(|(x, r)| x - r * b).collect()
            }
            None => x.to_vec(),
        }
    };
    let css_value = |par: &[f64]| -> f64 {
        let (phi, theta) = trans_pars(par, &arma, false);
        let (s, _) = css(&regress(par), &arma, &phi, &theta, ncond);
        0.5 * s.ln()
    };
    let ml_value = |par: &[f64], trans: bool| -> f64 {
        let (phi, theta) = trans_pars(par, &arma, trans);
        let mut model = StateSpace::new(&phi, &theta, &delta, 1e6);
        let (ssq, sumlog, nu, _) = model.likelihood(&regress(par), false);
        let nu = nu as f64;
        0.5 * ((ssq / nu).ln() + sumlog / nu)
    };

    let (value, sigma2, residuals, model);
    if method == Method::Css {
        let res_value = if no_optim {
            css_value(&coef)
        } else {
            let f = |p: &[f64]| Ok(css_value(&place(&fixed, p)));
            let res = bfgs(&f, &free(&init), &parscale_free, 100)?;
            coef = place(&coef, &res.par);
            res.value
        };
        let (phi, theta) = trans_pars(&coef, &arma, false);
        let mut m = StateSpace::new(&phi, &theta, &delta, 1e6);
        let xr = regress(&coef);
        m.likelihood(&xr, false);
        let (s2, resid) = css(&xr, &arma, &phi, &theta, ncond);
        value = res_value;
        sigma2 = s2;
        residuals = resid;
        model = m;
    } else {
        if method == Method::CssMl {
            // Without free coefficients optim is not run (its CSS value is not used).
            if !no_optim {
                let f = |p: &[f64]| Ok(css_value(&place(&fixed, p)));
                let res = bfgs(&f, &free(&init), &parscale_free, 100)?;
                if res.converged {
                    init = place(&init, &res.par);
                }
            }
            if mp > 0 && !ar_check(&init[..mp]) {
                return Err(Error::estimation("non-stationary AR part from CSS"));
            }
            if msp > 0 && !ar_check(&init[mp + mq..mp + mq + msp]) {
                return Err(Error::estimation(
                    "non-stationary seasonal AR part from CSS",
                ));
            }
        }
        if transform {
            init = inv_trans(&init, &arma);
            if mq > 0 {
                let v = ma_invert(&init[mp..mp + mq]);
                init[mp..mp + mq].copy_from_slice(&v);
            }
            if msq > 0 {
                let s = mp + mq + msp;
                let v = ma_invert(&init[s..s + msq]);
                init[s..s + msq].copy_from_slice(&v);
            }
        }
        let res_value = if no_optim {
            ml_value(&coef, transform)
        } else {
            let f = |p: &[f64]| Ok(ml_value(&place(&fixed, p), transform));
            let res = bfgs(&f, &free(&init), &parscale_free, 100)?;
            coef = place(&coef, &res.par);
            let mut value = res.value;
            if transform {
                let before = coef.clone();
                if mq > 0 && mask[mp..mp + mq].iter().all(|m| *m) {
                    let v = ma_invert(&coef[mp..mp + mq]);
                    coef[mp..mp + mq].copy_from_slice(&v);
                }
                if msq > 0 {
                    let s = mp + mq + msp;
                    if mask[s..s + msq].iter().all(|m| *m) {
                        let v = ma_invert(&coef[s..s + msq]);
                        coef[s..s + msq].copy_from_slice(&v);
                    }
                }
                if free(&coef) != free(&before) {
                    // optim with maxit = 0: the value at the inverted coefficients.
                    value = ml_value(&coef, true);
                }
                coef = undo_pars(&coef, &arma);
            }
            value
        };
        let (phi, theta) = trans_pars(&coef, &arma, false);
        let mut m = StateSpace::new(&phi, &theta, &delta, 1e6);
        let (ssq, _, _, resid) = m.likelihood(&regress(&coef), true);
        value = res_value;
        sigma2 = ssq / n_used as f64;
        residuals = resid;
        model = m;
    }
    let nu = n_used as f64;
    let value2 = 2.0 * nu * value + nu + nu * (2.0 * std::f64::consts::PI).ln();
    let k = mask.iter().filter(|m| **m).count() as f64;
    let aic = (method != Method::Css).then_some(value2 + 2.0 * k + 2.0);
    Ok(Fit {
        coef,
        mask,
        sigma2,
        loglik: -0.5 * value2,
        aic,
        residuals,
        nobs: n_used,
        model,
        method,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differencing_polynomial() {
        assert_eq!(delta(1, 0, 12), vec![1.0]);
        assert_eq!(delta(2, 0, 12), vec![2.0, -1.0]);
        let d = delta(1, 1, 4);
        assert_eq!(d, vec![1.0, 0.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn ma_inversion() {
        // 1 + 2B -> 1 + 0.5B
        assert_eq!(ma_invert(&[2.0]), vec![0.5]);
        // (1 + 2B)(1 + 0.25B) = 1 + 2.25B + 0.5B^2 -> (1 + 0.5B)(1 + 0.25B) = 1 + 0.75B + 0.125B^2
        let v = ma_invert(&[2.25, 0.5]);
        assert!((v[0] - 0.75).abs() < 1e-12 && (v[1] - 0.125).abs() < 1e-12);
        assert_eq!(ma_invert(&[0.3, 0.0]), vec![0.3, 0.0]);
    }
}
