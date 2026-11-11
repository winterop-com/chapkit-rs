//! statsmodels' `tsa.arima.model.ARIMA(y, order=(p, d, q)).fit()` (statsmodels 0.15): the
//! state-space model of `SARIMAX` with the constant as a regression (`trend='c'` when `d == 0`),
//! starting values by conditional least squares, exact likelihood by the Kalman filter with an
//! approximate diffuse prior on the differenced states, and L-BFGS-B on the transformed
//! parameters. What chap-core's MSTL + ARIMA weather provider fits.

mod lbfgsb;

use crate::poly::{roots, roots_outside};
use crate::{Error, Result};

/// The variance of the approximate diffuse prior (`initial_variance`).
const DIFFUSE: f64 = 1e6;
/// `maxiter` of `MLEModel.fit`.
const MAXITER: usize = 50;

/// A fitted statsmodels ARIMA.
#[derive(Debug, Clone, PartialEq)]
pub struct SmArima {
    pub p: usize,
    pub d: usize,
    pub q: usize,
    /// `const` (when `d == 0`), `ar.L1..`, `ma.L1..`, `sigma2`.
    pub params: Vec<f64>,
    pub llf: f64,
    pub aicc: f64,
    /// The predicted state after the last observation, where forecasts start.
    state: Vec<f64>,
}

/// `constrain_stationary_univariate`.
fn constrain(u: &[f64]) -> Vec<f64> {
    let n = u.len();
    let r: Vec<f64> = u.iter().map(|v| v / (1.0 + v * v).sqrt()).collect();
    let mut y = vec![vec![0.0; n]; n];
    for k in 0..n {
        for i in 0..k {
            y[k][i] = y[k - 1][i] + r[k] * y[k - 1][k - i - 1];
        }
        y[k][k] = r[k];
    }
    y.last()
        .map_or_else(Vec::new, |row| row.iter().map(|v| -v).collect())
}

/// `unconstrain_stationary_univariate`.
fn unconstrain(c: &[f64]) -> Vec<f64> {
    let n = c.len();
    if n == 0 {
        return Vec::new();
    }
    let mut y = vec![vec![0.0; n]; n];
    y[n - 1] = c.iter().map(|v| -v).collect();
    for k in (1..n).rev() {
        for i in 0..k {
            y[k - 1][i] = (y[k][i] - y[k][k] * y[k][k - i - 1]) / (1.0 - y[k][k] * y[k][k]);
        }
    }
    (0..n)
        .map(|i| {
            let r = y[i][i];
            r / (1.0 - r * r).sqrt()
        })
        .collect()
}

/// Least squares by the pseudo-inverse (numpy's `pinv`, `rcond = 1e-15`), via a one-sided Jacobi
/// SVD. `cols` are the regressors.
fn pinv_solve(cols: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = cols.len();
    let mut a: Vec<Vec<f64>> = cols.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    for _ in 0..60 {
        let mut rotated = false;
        for p in 0..n {
            for q in p + 1..n {
                let alpha: f64 = a[p].iter().map(|x| x * x).sum();
                let beta: f64 = a[q].iter().map(|x| x * x).sum();
                let gamma: f64 = a[p].iter().zip(&a[q]).map(|(x, y)| x * y).sum();
                if gamma.abs() <= 1e-15 * (alpha * beta).sqrt() || gamma == 0.0 {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for k in 0..a[p].len() {
                    let (x, y) = (a[p][k], a[q][k]);
                    a[p][k] = c * x - s * y;
                    a[q][k] = s * x + c * y;
                }
                for row in &mut v {
                    let (x, y) = (row[p], row[q]);
                    row[p] = c * x - s * y;
                    row[q] = s * x + c * y;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let sigma: Vec<f64> = a
        .iter()
        .map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt())
        .collect();
    let cutoff = 1e-15 * sigma.iter().fold(0.0f64, |m, s| m.max(*s));
    let mut out = vec![0.0; n];
    for j in 0..n {
        if sigma[j] > cutoff {
            let w = a[j].iter().zip(y).map(|(a, y)| a * y).sum::<f64>() / (sigma[j] * sigma[j]);
            for (o, row) in out.iter_mut().zip(&v) {
                *o += row[j] * w;
            }
        }
    }
    out
}

/// `SARIMAX._conditional_sum_squares` for `k_trend = 0`: AR and MA starting values (an AR(2q)
/// for the residuals first when there is an MA part) and the residuals.
fn conditional_sum_squares(
    endog: &[f64],
    p: usize,
    q: usize,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    if p + q == 0 {
        return (Vec::new(), Vec::new(), None);
    }
    let n = endog.len();
    let k = 2 * q;
    let r = (k + q).max(p);
    // lagmat raises when a lag is not below the number of observations.
    if (q > 0 && (k >= n || q >= n - k)) || (p > 0 && p >= n) || r >= n {
        let mean = endog.iter().sum::<f64>() / n as f64;
        let mut resid = vec![0.0; 2 * q];
        resid.extend(endog.iter().map(|v| v - mean));
        return (vec![0.0; p], vec![0.0; q], Some(resid));
    }
    let mut ma_resid = Vec::new();
    if q > 0 {
        let cols: Vec<Vec<f64>> = (0..k)
            .map(|j| (k..n).map(|t| endog[t - 1 - j]).collect())
            .collect();
        let b = pinv_solve(&cols, &endog[k..]);
        ma_resid = (k..n)
            .map(|t| endog[t] - (0..k).map(|j| endog[t - 1 - j] * b[j]).sum::<f64>())
            .collect();
    }
    let mut cols: Vec<Vec<f64>> = (0..p)
        .map(|j| (r..n).map(|t| endog[t - 1 - j]).collect())
        .collect();
    cols.extend((0..q).map(|j| {
        (r..n)
            .map(|t| ma_resid[t - k - 1 - j])
            .collect::<Vec<f64>>()
    }));
    let params = pinv_solve(&cols, &endog[r..]);
    let resid: Vec<f64> = (0..n - r)
        .map(|i| endog[r + i] - cols.iter().zip(&params).map(|(c, b)| c[i] * b).sum::<f64>())
        .collect();
    (params[..p].to_vec(), params[p..].to_vec(), Some(resid))
}

/// The state space: design `Z`, transition `T`, selection `R`, row-major, for parameters
/// `(phi, theta)`.
struct Ssm {
    k: usize,
    d: usize,
    r: usize,
    t: Vec<f64>,
    rsel: Vec<f64>,
    z: Vec<f64>,
}

impl Ssm {
    fn new(d: usize, phi: &[f64], theta: &[f64]) -> Self {
        let r = phi.len().max(theta.len() + 1);
        let k = d + r;
        let mut t = vec![0.0; k * k];
        for i in 0..d {
            for j in i..d {
                t[i * k + j] = 1.0;
            }
            t[i * k + d] = 1.0;
        }
        for (i, v) in phi.iter().enumerate() {
            t[(d + i) * k + d] = *v;
        }
        for i in 0..r - 1 {
            t[(d + i) * k + d + i + 1] = 1.0;
        }
        let mut rsel = vec![0.0; k];
        rsel[d] = 1.0;
        for (i, v) in theta.iter().enumerate() {
            rsel[d + 1 + i] = *v;
        }
        let mut z = vec![0.0; k];
        for v in z.iter_mut().take(d + 1) {
            *v = 1.0;
        }
        Self {
            k,
            d,
            r,
            t,
            rsel,
            z,
        }
    }

    /// The stationary covariance of the ARMA block: `P = T P T' + R R' sigma2`, by solving
    /// `(I - T kron T) vec(P) = vec(R R' sigma2)`.
    fn stationary(&self, sigma2: f64) -> Result<Vec<f64>> {
        let (k, d, r) = (self.k, self.d, self.r);
        let ta = |i: usize, j: usize| self.t[(d + i) * k + d + j];
        let m = r * r;
        let mut a = vec![0.0; m * m];
        let mut b = vec![0.0; m];
        for i in 0..r {
            for j in 0..r {
                let row = i * r + j;
                b[row] = self.rsel[d + i] * self.rsel[d + j] * sigma2;
                for kk in 0..r {
                    for l in 0..r {
                        a[row * m + kk * r + l] -= ta(i, kk) * ta(j, l);
                    }
                }
                a[row * m + row] += 1.0;
            }
        }
        // Gaussian elimination with partial pivoting.
        for col in 0..m {
            let piv = (col..m)
                .max_by(|x, y| a[x * m + col].abs().total_cmp(&a[y * m + col].abs()))
                .unwrap_or(col);
            if a[piv * m + col] == 0.0 {
                return Err(Error::estimation(
                    "singular stationary covariance equations",
                ));
            }
            if piv != col {
                for c in 0..m {
                    a.swap(piv * m + c, col * m + c);
                }
                b.swap(piv, col);
            }
            for row in col + 1..m {
                let f = a[row * m + col] / a[col * m + col];
                if f != 0.0 {
                    for c in col..m {
                        a[row * m + c] -= f * a[col * m + c];
                    }
                    b[row] -= f * b[col];
                }
            }
        }
        let mut x = vec![0.0; m];
        for row in (0..m).rev() {
            let mut s = b[row];
            for c in row + 1..m {
                s -= a[row * m + c] * x[c];
            }
            x[row] = s / a[row * m + row];
        }
        Ok(x)
    }

    /// statsmodels' conventional Kalman filter (`TIMING_INIT_PREDICTED`, forced symmetry,
    /// steady-state detection at `1e-19` when time-invariant): the log-likelihood from
    /// observation `d` on (the diffuse ones burnt), and the predicted state after the last
    /// observation.
    fn filter(
        &self,
        y: &[f64],
        intercept: f64,
        sigma2: f64,
        time_invariant: bool,
    ) -> Result<(f64, Vec<f64>)> {
        let (k, d, r) = (self.k, self.d, self.r);
        let mut a = vec![0.0; k];
        let mut pm = vec![0.0; k * k];
        for i in 0..d {
            pm[i * k + i] = DIFFUSE;
        }
        let p0 = self.stationary(sigma2)?;
        for i in 0..r {
            for j in 0..r {
                pm[(d + i) * k + d + j] = p0[i * r + j];
            }
        }
        let rqr: Vec<f64> = (0..k * k)
            .map(|ij| self.rsel[ij / k] * self.rsel[ij % k] * sigma2)
            .collect();
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let mut llf = 0.0;
        let mut converged = false;
        let (mut f_conv, mut det_conv) = (0.0, 0.0);
        let mut pz = vec![0.0; k];
        let mut pf = vec![0.0; k * k];
        let mut tmp = vec![0.0; k * k];
        for (t, &yt) in y.iter().enumerate() {
            let fc = intercept + self.z.iter().zip(&a).map(|(z, a)| z * a).sum::<f64>();
            let v = yt - fc;
            for i in 0..k {
                pz[i] = (0..k).map(|j| pm[i * k + j] * self.z[j]).sum();
            }
            let (f, det) = if converged {
                (f_conv, det_conv)
            } else {
                let f: f64 = self.z.iter().zip(&pz).map(|(z, p)| z * p).sum();
                (f, f.ln())
            };
            // A forecast error variance below 1e-12 makes statsmodels switch to its univariate
            // filter for the period, which skips an observation whose variance is not above 1e-10.
            let skip = f < 1e-12;
            let s = if skip { 0.0 } else { 1.0 / f };
            let tmp2 = s * v;
            let af: Vec<f64> = (0..k).map(|i| a[i] + pz[i] * tmp2).collect();
            if t >= d && !skip {
                llf += -0.5 * (ln2pi + det) - 0.5 * v * tmp2;
            }
            for (i, ai) in a.iter_mut().enumerate() {
                *ai = (0..k).map(|j| self.t[i * k + j] * af[j]).sum();
            }
            if !converged {
                for i in 0..k {
                    for j in 0..k {
                        pf[i * k + j] = pm[i * k + j] - pz[i] * s * pz[j];
                    }
                }
                for i in 0..k {
                    for j in 0..k {
                        tmp[i * k + j] = (0..k).map(|l| self.t[i * k + l] * pf[l * k + j]).sum();
                    }
                }
                let mut next = rqr.clone();
                for i in 0..k {
                    for j in 0..k {
                        next[i * k + j] += (0..k)
                            .map(|l| tmp[i * k + l] * self.t[j * k + l])
                            .sum::<f64>();
                    }
                }
                for i in 0..k {
                    for j in i + 1..k {
                        let m = (next[i * k + j] + next[j * k + i]) * 0.5;
                        next[i * k + j] = m;
                        next[j * k + i] = m;
                    }
                }
                if time_invariant && t >= 1 {
                    let diff: f64 = pm.iter().zip(&next).map(|(a, b)| (a - b) * (a - b)).sum();
                    if diff < 1e-19 {
                        converged = true;
                        f_conv = f;
                        det_conv = det;
                        // P stays at this period's predicted covariance from here on.
                        continue;
                    }
                }
                pm = next;
            }
        }
        Ok((llf, a))
    }
}

/// The model of `(p, d, q)` with parameters `params` (constrained).
fn split(params: &[f64], p: usize, d: usize, q: usize) -> (f64, &[f64], &[f64], f64) {
    let c = usize::from(d == 0);
    let intercept = if c == 1 { params[0] } else { 0.0 };
    (
        intercept,
        &params[c..c + p],
        &params[c + p..c + p + q],
        params[c + p + q],
    )
}

fn loglike(y: &[f64], params: &[f64], p: usize, d: usize, q: usize) -> Result<(f64, Vec<f64>)> {
    let (intercept, phi, theta, sigma2) = split(params, p, d, q);
    Ssm::new(d, phi, theta).filter(y, intercept, sigma2, d > 0)
}

fn transform(x: &[f64], p: usize, d: usize, q: usize) -> Vec<f64> {
    let c = usize::from(d == 0);
    let mut out = x[..c].to_vec();
    out.extend(constrain(&x[c..c + p]));
    out.extend(constrain(&x[c + p..c + p + q]).iter().map(|v| -v));
    out.push(x[c + p + q] * x[c + p + q]);
    out
}

fn untransform(params: &[f64], p: usize, d: usize, q: usize) -> Vec<f64> {
    let c = usize::from(d == 0);
    let mut out = params[..c].to_vec();
    out.extend(unconstrain(&params[c..c + p]));
    let ma: Vec<f64> = params[c + p..c + p + q].iter().map(|v| -v).collect();
    out.extend(unconstrain(&ma));
    out.push(params[c + p + q].sqrt());
    out
}

/// `SARIMAX.start_params` for ARIMA(p, d, q) with the constant as a regression when `d == 0`.
fn start_params(y: &[f64], p: usize, d: usize, q: usize) -> Vec<f64> {
    let n = y.len();
    let mut endog = y.to_vec();
    for _ in 0..d {
        endog = endog.windows(2).map(|w| w[1] - w[0]).collect();
    }
    let mut out = Vec::new();
    if d == 0 {
        let mean = pinv_solve(&[vec![1.0; endog.len()]], &endog)[0];
        for v in &mut endog {
            *v -= mean;
        }
        out.push(mean);
    }
    let (mut ar, mut ma, resid) = conditional_sum_squares(&endog, p, q);
    // is_invertible: companion eigenvalues (inverse roots) below 1 - 1e-10.
    let bound = 1.0 / (1.0 - 1e-10);
    if p > 0 && !roots_outside(&ar.iter().map(|v| -v).collect::<Vec<_>>(), bound) {
        ar.fill(0.0);
    }
    if q > 0 && !roots_outside(&ma, bound) {
        ma.fill(0.0);
    }
    let variance = match resid {
        Some(r) if r.len() > q.max(1) => {
            let tail = &r[q..];
            tail.iter().map(|v| v * v).sum::<f64>() / tail.len() as f64
        }
        Some(_) => {
            let m = endog.iter().sum::<f64>() / endog.len() as f64;
            endog.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / endog.len() as f64
        }
        None => {
            let ss: f64 = endog.iter().map(|v| v * v).sum();
            if d == 0 { ss } else { ss / n as f64 }
        }
    };
    out.extend(ar);
    out.extend(ma);
    out.push(variance.max(1e-10));
    out
}

impl SmArima {
    /// `ARIMA(y, order=(p, d, q)).fit()`.
    pub fn fit(y: &[f64], p: usize, d: usize, q: usize) -> Result<Self> {
        let n = y.len();
        if n <= d + 1 || y.iter().any(|v| !v.is_finite()) {
            return Err(Error::invalid(
                "ARIMA needs a longer series without missing values",
            ));
        }
        let x0 = untransform(&start_params(y, p, d, q), p, d, q);
        let nf = n as f64;
        let objective =
            |x: &[f64]| -> Result<f64> { Ok(-loglike(y, &transform(x, p, d, q), p, d, q)?.0 / nf) };
        let x = lbfgsb::minimize(&objective, &x0, MAXITER)?;
        let params = transform(&x, p, d, q);
        let (llf, state) = loglike(y, &params, p, d, q)?;
        let k = params.len() as f64;
        let nobs = (n - d) as f64;
        let dof = nobs - k - 1.0;
        let aicc = if dof > 0.0 {
            -2.0 * llf + 2.0 * k * nobs / dof
        } else {
            f64::INFINITY
        };
        Ok(Self {
            p,
            d,
            q,
            params,
            llf,
            aicc,
            state,
        })
    }

    fn arma(&self) -> (f64, &[f64], &[f64], f64) {
        split(&self.params, self.p, self.d, self.q)
    }

    /// The smallest modulus of the AR and MA roots (`arroots`, `maroots`); infinite without any.
    pub fn min_root_modulus(&self) -> f64 {
        let (_, phi, theta, _) = self.arma();
        let ar: Vec<f64> = std::iter::once(1.0).chain(phi.iter().map(|v| -v)).collect();
        let ma: Vec<f64> = std::iter::once(1.0).chain(theta.iter().copied()).collect();
        roots(&ar)
            .into_iter()
            .chain(roots(&ma))
            .map(crate::poly::Complex::norm)
            .fold(f64::INFINITY, f64::min)
    }

    /// The next `h` forecasts (means).
    pub fn forecast(&self, h: usize) -> Vec<f64> {
        let (intercept, phi, theta, _) = self.arma();
        let ssm = Ssm::new(self.d, phi, theta);
        let k = ssm.k;
        let mut a = self.state.clone();
        let mut out = Vec::with_capacity(h);
        for _ in 0..h {
            out.push(intercept + ssm.z.iter().zip(&a).map(|(z, a)| z * a).sum::<f64>());
            a = (0..k)
                .map(|i| (0..k).map(|j| ssm.t[i * k + j] * a[j]).sum())
                .collect();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transforms_round_trip() {
        let c = [0.5, -0.3, 0.1];
        let back = constrain(&unconstrain(&c));
        for (a, b) in c.iter().zip(&back) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn pinv_least_squares() {
        let x1 = vec![1.0, 2.0, 3.0, 4.0];
        let x2 = vec![1.0, 1.0, 1.0, 1.0];
        let y = [3.0, 5.0, 7.0, 9.0];
        let b = pinv_solve(&[x1.clone(), x2], &y);
        assert!((b[0] - 2.0).abs() < 1e-12 && (b[1] - 1.0).abs() < 1e-12);
        // Rank deficient: the minimum-norm solution.
        let b = pinv_solve(&[x1.clone(), x1], &[2.0, 4.0, 6.0, 8.0]);
        assert!((b[0] - 1.0).abs() < 1e-12 && (b[1] - 1.0).abs() < 1e-12);
    }
}
