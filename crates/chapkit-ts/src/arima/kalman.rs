//! R's ARIMA state space (`stats::makeARIMA`) and the C code `stats::arima` runs on it
//! (`src/library/stats/src/arima.c`): Gardner et al.'s initial covariance (`getQ0`), the exact
//! likelihood by Kalman filter with a diffuse prior on the differenced states (`ARIMA_Like`), the
//! conditional sum of squares (`ARIMA_CSS`), the parameter transforms and `KalmanFore`.

use serde::{Deserialize, Serialize};

/// The state space of an ARIMA model as `makeARIMA` builds it. Matrices are column-major. After
/// [`StateSpace::likelihood`], `a` and `p` hold the filtered state at the last observation, which
/// is where forecasts start (as R's `arima` leaves its `model`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StateSpace {
    #[serde(with = "crate::serde_float::many")]
    pub phi: Vec<f64>,
    /// MA coefficients, zero-padded to `r - 1`.
    #[serde(with = "crate::serde_float::many")]
    pub theta: Vec<f64>,
    /// The differencing polynomial (`Delta`), sign as in `makeARIMA`.
    #[serde(with = "crate::serde_float::many")]
    pub delta: Vec<f64>,
    #[serde(with = "crate::serde_float::many")]
    pub a: Vec<f64>,
    #[serde(with = "crate::serde_float::many")]
    pub p: Vec<f64>,
    #[serde(with = "crate::serde_float::many")]
    pub pn: Vec<f64>,
}

impl StateSpace {
    /// `makeARIMA(phi, theta, Delta, kappa)` with Gardner's initial covariance.
    pub fn new(phi: &[f64], theta: &[f64], delta: &[f64], kappa: f64) -> Self {
        let p = phi.len();
        let q = theta.len();
        let r = p.max(q + 1);
        let d = delta.len();
        let rd = r + d;
        let mut theta = theta.to_vec();
        theta.resize(theta.len().max(r - 1), 0.0);
        let mut pn = vec![0.0; rd * rd];
        if r > 1 {
            let q0 = initial_covariance(phi, &theta);
            for j in 0..r {
                for i in 0..r {
                    pn[i + rd * j] = q0[i + r * j];
                }
            }
        } else {
            pn[0] = if p > 0 {
                1.0 / (1.0 - phi[0] * phi[0])
            } else {
                1.0
            };
        }
        for i in r..rd {
            pn[i + rd * i] = kappa;
        }
        Self {
            phi: phi.to_vec(),
            theta,
            delta: delta.to_vec(),
            a: vec![0.0; rd],
            p: vec![0.0; rd * rd],
            pn,
        }
    }

    fn r(&self) -> usize {
        self.a.len() - self.delta.len()
    }

    /// `ARIMA_Like(y, mod, 0, giveResid)`: `(ssq, sumlog, nu)` and, if asked, the standardised
    /// residuals. Updates `a`, `p` and `pn` in place, as R's C code does.
    pub fn likelihood(&mut self, y: &[f64], give_resid: bool) -> (f64, f64, usize, Vec<f64>) {
        let rd = self.a.len();
        let d = self.delta.len();
        let r = rd - d;
        let p = self.phi.len();
        let q = self.theta.len();
        let (phi, theta, delta) = (&self.phi, &self.theta, &self.delta);
        let (a, pm, pnew) = (&mut self.a, &mut self.p, &mut self.pn);
        let mut anew = vec![0.0; rd];
        let mut m = vec![0.0; rd];
        let mut mm = if d > 0 {
            vec![0.0; rd * rd]
        } else {
            Vec::new()
        };
        let mut resid = if give_resid {
            vec![0.0; y.len()]
        } else {
            Vec::new()
        };
        let (mut ssq, mut sumlog, mut nu) = (0.0, 0.0, 0usize);
        for (l, &yl) in y.iter().enumerate() {
            for i in 0..r {
                let mut tmp = if i < r - 1 { a[i + 1] } else { 0.0 };
                if i < p {
                    tmp += phi[i] * a[0];
                }
                anew[i] = tmp;
            }
            if d > 0 {
                for i in r + 1..rd {
                    anew[i] = a[i - 1];
                }
                let mut tmp = a[0];
                for i in 0..d {
                    tmp += delta[i] * a[r + i];
                }
                anew[r] = tmp;
            }
            if l > 0 {
                if d == 0 {
                    for i in 0..r {
                        let vi = if i == 0 {
                            1.0
                        } else if i - 1 < q {
                            theta[i - 1]
                        } else {
                            0.0
                        };
                        for j in 0..r {
                            let mut tmp = if j == 0 {
                                vi
                            } else if j - 1 < q {
                                vi * theta[j - 1]
                            } else {
                                0.0
                            };
                            if i < p && j < p {
                                tmp += phi[i] * phi[j] * pm[0];
                            }
                            if i < r - 1 && j < r - 1 {
                                tmp += pm[i + 1 + r * (j + 1)];
                            }
                            if i < p && j < r - 1 {
                                tmp += phi[i] * pm[j + 1];
                            }
                            if j < p && i < r - 1 {
                                tmp += phi[j] * pm[i + 1];
                            }
                            pnew[i + r * j] = tmp;
                        }
                    }
                } else {
                    // mm = TP
                    for i in 0..r {
                        for j in 0..rd {
                            let mut tmp = 0.0;
                            if i < p {
                                tmp += phi[i] * pm[rd * j];
                            }
                            if i < r - 1 {
                                tmp += pm[i + 1 + rd * j];
                            }
                            mm[i + rd * j] = tmp;
                        }
                    }
                    for j in 0..rd {
                        let mut tmp = pm[rd * j];
                        for k in 0..d {
                            tmp += delta[k] * pm[r + k + rd * j];
                        }
                        mm[r + rd * j] = tmp;
                    }
                    for i in 1..d {
                        for j in 0..rd {
                            mm[r + i + rd * j] = pm[r + i - 1 + rd * j];
                        }
                    }
                    // Pnew = mm T'
                    for i in 0..r {
                        for j in 0..rd {
                            let mut tmp = 0.0;
                            if i < p {
                                tmp += phi[i] * mm[j];
                            }
                            if i < r - 1 {
                                tmp += mm[rd * (i + 1) + j];
                            }
                            pnew[j + rd * i] = tmp;
                        }
                    }
                    for j in 0..rd {
                        let mut tmp = mm[j];
                        for k in 0..d {
                            tmp += delta[k] * mm[rd * (r + k) + j];
                        }
                        pnew[rd * r + j] = tmp;
                    }
                    for i in 1..d {
                        for j in 0..rd {
                            pnew[rd * (r + i) + j] = mm[rd * (r + i - 1) + j];
                        }
                    }
                    for i in 0..=q {
                        let vi = if i == 0 { 1.0 } else { theta[i - 1] };
                        for j in 0..=q {
                            pnew[i + rd * j] += vi * if j == 0 { 1.0 } else { theta[j - 1] };
                        }
                    }
                }
            }
            if yl.is_nan() {
                a.copy_from_slice(&anew);
                pm.copy_from_slice(pnew);
                if give_resid {
                    resid[l] = f64::NAN;
                }
            } else {
                let mut res = yl - anew[0];
                for i in 0..d {
                    res -= delta[i] * anew[r + i];
                }
                for i in 0..rd {
                    let mut tmp = pnew[i];
                    for j in 0..d {
                        tmp += pnew[i + (r + j) * rd] * delta[j];
                    }
                    m[i] = tmp;
                }
                let mut gain = m[0];
                for j in 0..d {
                    gain += delta[j] * m[r + j];
                }
                if gain < 1e4 {
                    nu += 1;
                    ssq += res * res / gain;
                    sumlog += gain.ln();
                }
                if give_resid {
                    resid[l] = res / gain.sqrt();
                }
                for i in 0..rd {
                    a[i] = anew[i] + m[i] * res / gain;
                }
                for i in 0..rd {
                    for j in 0..rd {
                        pm[i + j * rd] = pnew[i + j * rd] - m[i] * m[j] / gain;
                    }
                }
            }
        }
        (ssq, sumlog, nu, resid)
    }

    /// `KalmanForecast(h, mod)`: the means and the variances (in units of `sigma2`) of the next
    /// `h` observations from the current state.
    pub fn forecast(&self, h: usize) -> (Vec<f64>, Vec<f64>) {
        let rd = self.a.len();
        let r = self.r();
        let d = self.delta.len();
        // T and V as makeARIMA builds them, and Z.
        let mut t = vec![0.0; rd * rd];
        for (i, v) in self.phi.iter().enumerate() {
            t[i] = *v;
        }
        for i in 1..r {
            t[(i - 1) + rd * i] = 1.0;
        }
        let mut z = vec![0.0; rd];
        z[0] = 1.0;
        for (i, v) in self.delta.iter().enumerate() {
            z[r + i] = *v;
        }
        if d > 0 {
            for j in 0..rd {
                t[r + rd * j] = z[j];
            }
            for i in 1..d {
                t[(r + i) + rd * (r + i - 1)] = 1.0;
            }
        }
        let mut rv = vec![0.0; rd];
        rv[0] = 1.0;
        for (i, v) in self.theta.iter().enumerate() {
            rv[1 + i] = *v;
        }
        let mut a = self.a.clone();
        let mut pm = self.p.clone();
        let mut anew = vec![0.0; rd];
        let mut mm = vec![0.0; rd * rd];
        let mut pnew = vec![0.0; rd * rd];
        let mut pred = Vec::with_capacity(h);
        let mut var = Vec::with_capacity(h);
        for _ in 0..h {
            let mut fc = 0.0;
            for i in 0..rd {
                let mut tmp = 0.0;
                for k in 0..rd {
                    tmp += t[i + rd * k] * a[k];
                }
                anew[i] = tmp;
                fc += tmp * z[i];
            }
            a.copy_from_slice(&anew);
            pred.push(fc);
            for i in 0..rd {
                for j in 0..rd {
                    let mut tmp = 0.0;
                    for k in 0..rd {
                        tmp += t[i + rd * k] * pm[k + rd * j];
                    }
                    mm[i + rd * j] = tmp;
                }
            }
            for i in 0..rd {
                for j in 0..rd {
                    let mut tmp = rv[i] * rv[j];
                    for k in 0..rd {
                        tmp += mm[i + rd * k] * t[j + rd * k];
                    }
                    pnew[i + rd * j] = tmp;
                }
            }
            let mut tmp = 0.0;
            for i in 0..rd {
                for j in 0..rd {
                    pm[i + j * rd] = pnew[i + j * rd];
                    tmp += z[i] * z[j] * pm[i + j * rd];
                }
            }
            var.push(tmp);
        }
        (pred, var)
    }
}

/// The initial ARMA state covariance: Gardner's (`getQ0`, as R) unless it fails to solve
/// `P = T P T' + R R'` (it loses all accuracy when the covariance is close to singular, as for
/// long seasonal lags; R then goes on with a matrix that is not even positive semi-definite), in
/// which case the series `sum T^k R R' T'^k` by doubling.
pub(crate) fn initial_covariance(phi: &[f64], theta: &[f64]) -> Vec<f64> {
    let g = q0(phi, theta);
    let r = phi.len().max(theta.len() + 1);
    let rv = |i: usize| -> f64 {
        if i == 0 {
            1.0
        } else {
            theta.get(i - 1).copied().unwrap_or(0.0)
        }
    };
    let ph = |i: usize| phi.get(i).copied().unwrap_or(0.0);
    // T P T' for the companion T (phi in the first column, ones above the diagonal).
    let at = |m: &[f64], i: usize, j: usize| if i < r && j < r { m[i + r * j] } else { 0.0 };
    let mut scale = 0.0f64;
    let mut resid = 0.0f64;
    for i in 0..r {
        for j in 0..r {
            let tpt = ph(i) * ph(j) * g[0]
                + ph(i) * at(&g, 0, j + 1)
                + ph(j) * at(&g, i + 1, 0)
                + at(&g, i + 1, j + 1);
            let v = g[i + r * j];
            scale = scale.max(v.abs());
            resid = resid.max((v - tpt - rv(i) * rv(j)).abs());
        }
    }
    let gardner_ok = resid.is_finite() && resid <= 1e-8 * scale.max(1.0);
    if gardner_ok {
        return g;
    }
    doubling(phi, theta, r).unwrap_or(g)
}

/// `P = sum_k T^k R R' T'^k` by doubling: `P <- P + A P A'`, `A <- A A`. `None` when it does not
/// converge (a non-stationary `T`).
fn doubling(phi: &[f64], theta: &[f64], r: usize) -> Option<Vec<f64>> {
    let mut a = vec![0.0; r * r];
    for (i, v) in phi.iter().enumerate() {
        a[i] = *v;
    }
    for i in 1..r {
        a[(i - 1) + r * i] = 1.0;
    }
    let mut rv = vec![0.0; r];
    rv[0] = 1.0;
    for (i, v) in theta.iter().enumerate() {
        rv[i + 1] = *v;
    }
    let mut p: Vec<f64> = (0..r * r).map(|ij| rv[ij % r] * rv[ij / r]).collect();
    let mul = |x: &[f64], y: &[f64], yt: bool| -> Vec<f64> {
        let mut out = vec![0.0; r * r];
        for j in 0..r {
            for k in 0..r {
                let ykj = if yt { y[j + r * k] } else { y[k + r * j] };
                if ykj != 0.0 {
                    for i in 0..r {
                        out[i + r * j] += x[i + r * k] * ykj;
                    }
                }
            }
        }
        out
    };
    for _ in 0..64 {
        let apa = mul(&mul(&a, &p, false), &a, true);
        let size = p.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let change = apa.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (pi, x) in p.iter_mut().zip(&apa) {
            *pi += x;
        }
        if !change.is_finite() {
            return None;
        }
        if change <= 1e-16 * size {
            return Some(p);
        }
        a = mul(&a, &a, false);
    }
    None
}

/// `inclu2` (AS 154): include one row in the QR of the equations for the initial covariance.
fn inclu2(
    xnext: &[f64],
    xrow: &mut [f64],
    mut ynext: f64,
    d: &mut [f64],
    rbar: &mut [f64],
    thetab: &mut [f64],
) {
    let np = xnext.len();
    xrow.copy_from_slice(xnext);
    let mut ithisr = 0;
    for i in 0..np {
        if xrow[i] == 0.0 {
            ithisr += np - i - 1;
        } else {
            let xi = xrow[i];
            let di = d[i];
            let dpi = di + xi * xi;
            d[i] = dpi;
            let cbar = di / dpi;
            let sbar = xi / dpi;
            for k in i + 1..np {
                let xk = xrow[k];
                let rbthis = rbar[ithisr];
                xrow[k] = xk - xi * rbthis;
                rbar[ithisr] = cbar * rbthis + sbar * xk;
                ithisr += 1;
            }
            let xk = ynext;
            ynext = xk - xi * thetab[i];
            thetab[i] = cbar * thetab[i] + sbar * xk;
            if di == 0.0 {
                return;
            }
        }
    }
}

/// `getQ0`: the covariance of the initial ARMA state (Gardner, Harvey and Phillips, 1980), `r x r`
/// column-major with `r = max(p, q + 1)`.
#[allow(clippy::similar_names)]
pub(crate) fn q0(phi: &[f64], theta: &[f64]) -> Vec<f64> {
    let p = phi.len();
    let q = theta.len();
    let r = p.max(q + 1);
    let np = r * (r + 1) / 2;
    let nrbar = np * (np.saturating_sub(1)) / 2;
    let coef = |v: &[f64], k: usize| -> f64 {
        if k == 0 {
            1.0
        } else if k - 1 < v.len() {
            v[k - 1]
        } else {
            0.0
        }
    };
    let mut vv = vec![0.0; np];
    let mut ind = 0;
    for j in 0..r {
        let vj = coef(theta, j);
        for i in j..r {
            vv[ind] = coef(theta, i) * vj;
            ind += 1;
        }
    }
    let mut pm = vec![0.0; r * r];
    if r == 1 {
        pm[0] = if p == 0 {
            1.0
        } else {
            1.0 / (1.0 - phi[0] * phi[0])
        };
        return pm;
    }
    if p > 0 {
        let mut rbar = vec![0.0; nrbar];
        let mut thetab = vec![0.0; np];
        let mut xnext = vec![0.0; np];
        let mut xrow = vec![0.0; np];
        let mut ind = 0;
        let mut ind1: isize = -1;
        let npr = np - r;
        let npr1 = npr + 1;
        let mut indj = npr;
        let mut ind2: isize = npr as isize - 1;
        for j in 0..r {
            let phij = if j < p { phi[j] } else { 0.0 };
            xnext[indj] = 0.0;
            indj += 1;
            let mut indi = npr1 + j;
            for i in j..r {
                let ynext = vv[ind];
                ind += 1;
                let phii = if i < p { phi[i] } else { 0.0 };
                if j != r - 1 {
                    xnext[indj] = -phii;
                    if i != r - 1 {
                        xnext[indi] -= phij;
                        ind1 += 1;
                        xnext[ind1 as usize] = -1.0;
                    }
                }
                xnext[npr] = -phii * phij;
                ind2 += 1;
                if ind2 as usize >= np {
                    ind2 = 0;
                }
                xnext[ind2 as usize] += 1.0;
                inclu2(&xnext, &mut xrow, ynext, &mut pm, &mut rbar, &mut thetab);
                xnext[ind2 as usize] = 0.0;
                if i != r - 1 {
                    xnext[indi] = 0.0;
                    indi += 1;
                    xnext[ind1 as usize] = 0.0;
                }
            }
        }
        // pm holds d (np values) so far: it is the work vector `P` of the C code, sized r*r >= np.
        let mut ithisr = nrbar as isize - 1;
        let mut im = np as isize - 1;
        for i in 0..np {
            let mut bi = thetab[im as usize];
            let mut jm = np as isize - 1;
            for _ in 0..i {
                bi -= rbar[ithisr as usize] * pm[jm as usize];
                ithisr -= 1;
                jm -= 1;
            }
            pm[im as usize] = bi;
            im -= 1;
        }
        // Re-order.
        let mut ind = npr;
        for x in xnext.iter_mut().take(r) {
            *x = pm[ind];
            ind += 1;
        }
        let mut ind = np as isize - 1;
        let mut ind1 = npr as isize - 1;
        for _ in 0..npr {
            pm[ind as usize] = pm[ind1 as usize];
            ind -= 1;
            ind1 -= 1;
        }
        pm[..r].copy_from_slice(&xnext[..r]);
    } else {
        let mut indn = np;
        let mut ind = np;
        for i in 0..r {
            for j in 0..=i {
                ind -= 1;
                pm[ind] = vv[ind];
                if j != 0 {
                    indn -= 1;
                    pm[ind] += pm[indn];
                }
            }
        }
    }
    // Unpack to a full matrix.
    let mut ind = np;
    for i in (1..r).rev() {
        for j in (i..r).rev() {
            ind -= 1;
            pm[r * i + j] = pm[ind];
        }
    }
    for i in 0..r - 1 {
        for j in i + 1..r {
            pm[i + r * j] = pm[j + r * i];
        }
    }
    pm
}

/// `ARIMA_CSS`: the conditional sum of squares per term, and the residuals (zero for the first
/// `ncond`). `arma` is R's `(p, q, P, Q, period, d, D)`.
pub(crate) fn css(
    y: &[f64],
    arma: &[usize; 7],
    phi: &[f64],
    theta: &[f64],
    ncond: usize,
) -> (f64, Vec<f64>) {
    let n = y.len();
    let mut w = y.to_vec();
    for _ in 0..arma[5] {
        for l in (1..n).rev() {
            w[l] -= w[l - 1];
        }
    }
    let ns = arma[4];
    for _ in 0..arma[6] {
        for l in (ns..n).rev() {
            w[l] -= w[l - ns];
        }
    }
    let mut resid = vec![0.0; n];
    let (mut ssq, mut nu) = (0.0, 0usize);
    for l in ncond..n {
        let mut tmp = w[l];
        for (j, ph) in phi.iter().enumerate() {
            tmp -= ph * w[l - j - 1];
        }
        for j in 0..(l - ncond).min(theta.len()) {
            tmp -= theta[j] * resid[l - j - 1];
        }
        resid[l] = tmp;
        if !tmp.is_nan() {
            nu += 1;
            ssq += tmp * tmp;
        }
    }
    (ssq / nu as f64, resid)
}

/// `partrans`: the PACF (via `tanh`) to AR coefficients by the Durbin-Levinson recursion.
fn partrans(raw: &[f64]) -> Vec<f64> {
    let p = raw.len();
    let mut new: Vec<f64> = raw.iter().map(|v| v.tanh()).collect();
    let mut work = new.clone();
    for j in 1..p {
        let a = new[j];
        for k in 0..j {
            work[k] -= a * new[j - k - 1];
        }
        new[..j].copy_from_slice(&work[..j]);
    }
    new
}

/// `invpartrans`: AR coefficients back to `atanh` of the PACF.
fn invpartrans(phi: &[f64]) -> Vec<f64> {
    let p = phi.len();
    let mut new = phi.to_vec();
    let mut work = new.clone();
    for j in (1..p).rev() {
        let a = new[j];
        for k in 0..j {
            work[k] = (new[k] + a * new[j - k - 1]) / (1.0 - a * a);
        }
        new[..j].copy_from_slice(&work[..j]);
    }
    new.iter().map(|v| v.atanh()).collect()
}

/// `ARIMA_transPars`: expand `par` (`ar, ma, sar, sma, ...`) into the full `phi` and `theta`,
/// transforming the AR parts from the PACF scale when `trans`.
pub(crate) fn trans_pars(par: &[f64], arma: &[usize; 7], trans: bool) -> (Vec<f64>, Vec<f64>) {
    let [mp, mq, msp, msq, ns, ..] = *arma;
    let p = mp + ns * msp;
    let q = mq + ns * msq;
    let mut params = par.to_vec();
    if trans {
        if mp > 0 {
            params[..mp].copy_from_slice(&partrans(&par[..mp]));
        }
        let v = mp + mq;
        if msp > 0 {
            params[v..v + msp].copy_from_slice(&partrans(&par[v..v + msp]));
        }
    }
    let mut phi = vec![0.0; p];
    let mut theta = vec![0.0; q];
    phi[..mp].copy_from_slice(&params[..mp]);
    theta[..mq].copy_from_slice(&params[mp..mp + mq]);
    if ns > 0 {
        for j in 0..msp {
            phi[(j + 1) * ns - 1] += params[j + mp + mq];
            for i in 0..mp {
                phi[(j + 1) * ns + i] -= params[i] * params[j + mp + mq];
            }
        }
        for j in 0..msq {
            theta[(j + 1) * ns - 1] += params[j + mp + mq + msp];
            for i in 0..mq {
                theta[(j + 1) * ns + i] += params[i + mp] * params[j + mp + mq + msp];
            }
        }
    }
    (phi, theta)
}

/// `ARIMA_undoPars`: the AR parts from the PACF scale to coefficients.
pub(crate) fn undo_pars(par: &[f64], arma: &[usize; 7]) -> Vec<f64> {
    let [mp, mq, msp, ..] = *arma;
    let mut out = par.to_vec();
    if mp > 0 {
        out[..mp].copy_from_slice(&partrans(&par[..mp]));
    }
    let v = mp + mq;
    if msp > 0 {
        out[v..v + msp].copy_from_slice(&partrans(&par[v..v + msp]));
    }
    out
}

/// `ARIMA_Invtrans`: the AR parts from coefficients to the PACF scale.
pub(crate) fn inv_trans(par: &[f64], arma: &[usize; 7]) -> Vec<f64> {
    let [mp, mq, msp, ..] = *arma;
    let mut out = par.to_vec();
    if mp > 0 {
        out[..mp].copy_from_slice(&invpartrans(&par[..mp]));
    }
    let v = mp + mq;
    if msp > 0 {
        out[v..v + msp].copy_from_slice(&invpartrans(&par[v..v + msp]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q0_is_the_stationary_covariance() {
        // AR(1): 1 / (1 - phi^2).
        let q = q0(&[0.5], &[]);
        assert!((q[0] - 1.0 / 0.75).abs() < 1e-12);
        // AR(2) with phi = (0.5, 0.2): solve P = T P T' + R R' and compare.
        let (phi, theta) = ([0.5, 0.2], [0.3]);
        let q = q0(&phi, &theta);
        let t = [phi[0], phi[1], 1.0, 0.0];
        let rv = [1.0, theta[0]];
        for i in 0..2 {
            for j in 0..2 {
                let mut v = rv[i] * rv[j];
                for k in 0..2 {
                    for l in 0..2 {
                        v += t[i + 2 * k] * q[k + 2 * l] * t[j + 2 * l];
                    }
                }
                assert!((v - q[i + 2 * j]).abs() < 1e-10, "{i},{j}");
            }
        }
    }

    #[test]
    fn falls_back_when_gardner_fails() {
        // AR(3) times a weekly seasonal AR(1), MA(1) times a seasonal MA(1): r = 55, where
        // Gardner's method returns a matrix with a negative eigenvalue.
        let arma = [3, 1, 1, 1, 52, 1, 0];
        let par = [0.4847, 0.3493, 0.0480, -0.6377, -0.1151, 0.0864];
        let (phi, theta) = trans_pars(&par, &arma, true);
        let p = initial_covariance(&phi, &theta);
        let r = 55;
        // Check P = T P T' + R R'.
        let rv = |i: usize| {
            if i == 0 {
                1.0
            } else {
                theta.get(i - 1).copied().unwrap_or(0.0)
            }
        };
        let ph = |i: usize| phi.get(i).copied().unwrap_or(0.0);
        let at = |i: usize, j: usize| if i < r && j < r { p[i + r * j] } else { 0.0 };
        for i in 0..r {
            for j in 0..r {
                let tpt = ph(i) * ph(j) * p[0]
                    + ph(i) * at(0, j + 1)
                    + ph(j) * at(i + 1, 0)
                    + at(i + 1, j + 1);
                assert!((p[i + r * j] - tpt - rv(i) * rv(j)).abs() < 1e-9, "{i},{j}");
            }
        }
        // Where Gardner's is accurate it is kept as is.
        assert_eq!(
            initial_covariance(&[0.5, 0.2], &[0.3]),
            q0(&[0.5, 0.2], &[0.3])
        );
    }

    #[test]
    fn transforms_round_trip() {
        let arma = [2, 1, 1, 0, 4, 0, 0];
        let par = [0.4, -0.3, 0.2, 0.5];
        let back = undo_pars(&inv_trans(&par, &arma), &arma);
        for (a, b) in par.iter().zip(&back) {
            assert!((a - b).abs() < 1e-12);
        }
        let (phi, theta) = trans_pars(&par, &arma, false);
        // (1 - 0.4B + 0.3B^2)(1 - 0.5B^4): phi_4 = 0.5, phi_5 = -0.2, phi_6 = 0.15.
        assert_eq!(phi.len(), 6);
        assert!((phi[3] - 0.5).abs() < 1e-15 && (phi[4] + 0.2).abs() < 1e-15);
        assert!((phi[5] - 0.15).abs() < 1e-15);
        assert_eq!(theta, vec![0.2]);
    }
}
