//! scipy's L-BFGS-B (`fmin_l_bfgs_b`) as statsmodels' `MLEModel.fit` calls it: no bounds,
//! `m = 10`, `factr = 1e7`, `pgtol = 1e-5`, `maxls = 20`, and the gradient by forward differences
//! with an absolute step of `1e-5` (`approx_grad=True, epsilon=1e-5`).
//!
//! Without bounds the generalized Cauchy point is `x - g / theta` and the subspace minimisation is
//! the L-BFGS step, here by the two-loop recursion with the pairs and `theta` L-BFGS-B keeps; the
//! line search is MINPACK-2's `dcsrch` with L-BFGS-B's constants.

use rayon::prelude::*;

use crate::Result;

const M: usize = 10;
const PGTOL: f64 = 1e-5;
const EPS_STEP: f64 = 1e-5;
const MAXLS: usize = 20;
const FTOL: f64 = 1e-3;
const GTOL: f64 = 0.9;
const XTOL: f64 = 0.1;
const STPMAX: f64 = 1e10;
/// `factr * epsmch`.
const TOL: f64 = 1e7 * f64::EPSILON;

/// The value and forward-difference gradient at `x`.
fn eval<F: Fn(&[f64]) -> Result<f64> + Sync>(f: &F, x: &[f64]) -> Result<(f64, Vec<f64>)> {
    let f0 = f(x)?;
    let g: Result<Vec<f64>> = (0..x.len())
        .into_par_iter()
        .map(|i| {
            let mut x1 = x.to_vec();
            x1[i] = x[i] + EPS_STEP;
            let dx = x1[i] - x[i];
            Ok((f(&x1)? - f0) / dx)
        })
        .collect();
    Ok((f0, g?))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// MINPACK-2's `dcstep`: a safeguarded step for the line search.
#[allow(clippy::too_many_arguments)]
fn dcstep(
    stx: &mut f64,
    fx: &mut f64,
    dx: &mut f64,
    sty: &mut f64,
    fy: &mut f64,
    dy: &mut f64,
    stp: &mut f64,
    fp: f64,
    dp: f64,
    brackt: &mut bool,
    stpmin: f64,
    stpmax: f64,
) {
    let sgnd = dp * (*dx / dx.abs());
    let stpf;
    if fp > *fx {
        let theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).sqrt();
        if *stp < *stx {
            gamma = -gamma;
        }
        let p = (gamma - *dx) + theta;
        let q = ((gamma - *dx) + gamma) + dp;
        let r = p / q;
        let stpc = *stx + r * (*stp - *stx);
        let stpq = *stx + ((*dx / ((*fx - fp) / (*stp - *stx) + *dx)) / 2.0) * (*stp - *stx);
        stpf = if (stpc - *stx).abs() < (stpq - *stx).abs() {
            stpc
        } else {
            stpc + (stpq - stpc) / 2.0
        };
        *brackt = true;
    } else if sgnd < 0.0 {
        let theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).sqrt();
        if *stp > *stx {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = ((gamma - dp) + gamma) + *dx;
        let r = p / q;
        let stpc = *stp + r * (*stx - *stp);
        let stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        stpf = if (stpc - *stp).abs() > (stpq - *stp).abs() {
            stpc
        } else {
            stpq
        };
        *brackt = true;
    } else if dp.abs() < dx.abs() {
        let theta = 3.0 * (*fx - fp) / (*stp - *stx) + *dx + dp;
        let s = theta.abs().max(dx.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dx / s) * (dp / s)).max(0.0).sqrt();
        if *stp > *stx {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = (gamma + (*dx - dp)) + gamma;
        let r = p / q;
        let stpc = if r < 0.0 && gamma != 0.0 {
            *stp + r * (*stx - *stp)
        } else if *stp > *stx {
            stpmax
        } else {
            stpmin
        };
        let stpq = *stp + (dp / (dp - *dx)) * (*stx - *stp);
        if *brackt {
            let mut s = if (stpc - *stp).abs() < (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            s = if *stp > *stx {
                (*stp + 0.66 * (*sty - *stp)).min(s)
            } else {
                (*stp + 0.66 * (*sty - *stp)).max(s)
            };
            stpf = s;
        } else {
            let s = if (stpc - *stp).abs() > (stpq - *stp).abs() {
                stpc
            } else {
                stpq
            };
            stpf = s.min(stpmax).max(stpmin);
        }
    } else if *brackt {
        let theta = 3.0 * (fp - *fy) / (*sty - *stp) + *dy + dp;
        let s = theta.abs().max(dy.abs()).max(dp.abs());
        let mut gamma = s * ((theta / s).powi(2) - (*dy / s) * (dp / s)).sqrt();
        if *stp > *sty {
            gamma = -gamma;
        }
        let p = (gamma - dp) + theta;
        let q = ((gamma - dp) + gamma) + *dy;
        let r = p / q;
        stpf = *stp + r * (*sty - *stp);
    } else if *stp > *stx {
        stpf = stpmax;
    } else {
        stpf = stpmin;
    }
    if fp > *fx {
        *sty = *stp;
        *fy = fp;
        *dy = dp;
    } else {
        if sgnd < 0.0 {
            *sty = *stx;
            *fy = *fx;
            *dy = *dx;
        }
        *stx = *stp;
        *fx = fp;
        *dx = dp;
    }
    *stp = stpf;
}

#[derive(PartialEq)]
enum Task {
    Fg,
    Done,
}

/// MINPACK-2's `dcsrch` (Moré-Thuente line search), the state between calls.
struct Dcsrch {
    brackt: bool,
    stage: u8,
    ginit: f64,
    gtest: f64,
    gx: f64,
    gy: f64,
    finit: f64,
    fx: f64,
    fy: f64,
    stx: f64,
    sty: f64,
    stmin: f64,
    stmax: f64,
    width: f64,
    width1: f64,
}

impl Dcsrch {
    /// `task = 'START'`: the value and slope at step 0.
    fn start(f: f64, g: f64, stp: f64) -> Self {
        let width = STPMAX;
        Self {
            brackt: false,
            stage: 1,
            ginit: g,
            gtest: FTOL * g,
            gx: g,
            gy: g,
            finit: f,
            fx: f,
            fy: f,
            stx: 0.0,
            sty: 0.0,
            stmin: 0.0,
            stmax: stp + 4.0 * stp,
            width,
            width1: width / 0.5,
        }
    }

    /// The value and slope at `stp`: done (converged or a warning) or the next step to try.
    fn step(&mut self, f: f64, g: f64, stp: &mut f64) -> Task {
        let ftest = self.finit + *stp * self.gtest;
        if self.stage == 1 && f <= ftest && g >= 0.0 {
            self.stage = 2;
        }
        let mut done = false;
        if self.brackt && (*stp <= self.stmin || *stp >= self.stmax) {
            done = true;
        }
        if self.brackt && self.stmax - self.stmin <= XTOL * self.stmax {
            done = true;
        }
        if *stp == STPMAX && f <= ftest && g <= self.gtest {
            done = true;
        }
        if *stp == 0.0 && (f > ftest || g >= self.gtest) {
            done = true;
        }
        if f <= ftest && g.abs() <= GTOL * (-self.ginit) {
            done = true;
        }
        if done {
            return Task::Done;
        }
        if self.stage == 1 && f <= self.fx && f > ftest {
            let fm = f - *stp * self.gtest;
            let mut fxm = self.fx - self.stx * self.gtest;
            let mut fym = self.fy - self.sty * self.gtest;
            let gm = g - self.gtest;
            let mut gxm = self.gx - self.gtest;
            let mut gym = self.gy - self.gtest;
            dcstep(
                &mut self.stx,
                &mut fxm,
                &mut gxm,
                &mut self.sty,
                &mut fym,
                &mut gym,
                stp,
                fm,
                gm,
                &mut self.brackt,
                self.stmin,
                self.stmax,
            );
            self.fx = fxm + self.stx * self.gtest;
            self.fy = fym + self.sty * self.gtest;
            self.gx = gxm + self.gtest;
            self.gy = gym + self.gtest;
        } else {
            dcstep(
                &mut self.stx,
                &mut self.fx,
                &mut self.gx,
                &mut self.sty,
                &mut self.fy,
                &mut self.gy,
                stp,
                f,
                g,
                &mut self.brackt,
                self.stmin,
                self.stmax,
            );
        }
        if self.brackt {
            if (self.sty - self.stx).abs() >= 0.66 * self.width1 {
                *stp = self.stx + 0.5 * (self.sty - self.stx);
            }
            self.width1 = self.width;
            self.width = (self.sty - self.stx).abs();
        }
        if self.brackt {
            self.stmin = self.stx.min(self.sty);
            self.stmax = self.stx.max(self.sty);
        } else {
            self.stmin = *stp + 1.1 * (*stp - self.stx);
            self.stmax = *stp + 4.0 * (*stp - self.stx);
        }
        *stp = stp.max(0.0).min(STPMAX);
        if (self.brackt && (*stp <= self.stmin || *stp >= self.stmax))
            || (self.brackt && self.stmax - self.stmin <= XTOL * self.stmax)
        {
            *stp = self.stx;
        }
        Task::Fg
    }
}

/// The L-BFGS memory: pairs `(s, y, s'y)`, newest last, and `theta = y'y / s'y`.
struct Memory {
    pairs: Vec<(Vec<f64>, Vec<f64>, f64)>,
    theta: f64,
}

impl Memory {
    fn reset(&mut self) {
        self.pairs.clear();
        self.theta = 1.0;
    }

    /// `-H g` with `H0 = I / theta`.
    fn direction(&self, g: &[f64]) -> Vec<f64> {
        let mut q: Vec<f64> = g.iter().map(|v| -v).collect();
        let mut alpha = vec![0.0; self.pairs.len()];
        for (i, (s, y, sy)) in self.pairs.iter().enumerate().rev() {
            alpha[i] = dot(s, &q) / sy;
            for (qj, yj) in q.iter_mut().zip(y) {
                *qj -= alpha[i] * yj;
            }
        }
        let mut z: Vec<f64> = q.iter().map(|v| v / self.theta).collect();
        for (i, (s, y, sy)) in self.pairs.iter().enumerate() {
            let beta = dot(y, &z) / sy;
            for (zj, sj) in z.iter_mut().zip(s) {
                *zj += sj * (alpha[i] - beta);
            }
        }
        z
    }
}

/// Minimise `f` from `x0`, at most `maxiter` iterations; the last accepted point.
pub(crate) fn minimize<F: Fn(&[f64]) -> Result<f64> + Sync>(
    f: &F,
    x0: &[f64],
    maxiter: usize,
) -> Result<Vec<f64>> {
    let mut x = x0.to_vec();
    let (mut fx, mut g) = eval(f, &x)?;
    let pg = |g: &[f64]| g.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if pg(&g) <= PGTOL {
        return Ok(x);
    }
    let mut mem = Memory {
        pairs: Vec::new(),
        theta: 1.0,
    };
    let mut iter = 0usize;
    loop {
        // The search direction: the Cauchy point x - g (theta = 1) with no memory, the L-BFGS
        // step otherwise, as the difference of the new point and x.
        let z: Vec<f64> = if mem.pairs.is_empty() {
            let dtm = 1.0 / mem.theta;
            x.iter().zip(&g).map(|(x, g)| x + dtm * -g).collect()
        } else {
            let du = mem.direction(&g);
            x.iter().zip(&du).map(|(x, d)| x + d).collect()
        };
        let mut d: Vec<f64> = z.iter().zip(&x).map(|(z, x)| z - x).collect();
        let dnorm = dot(&d, &d).sqrt();
        let mut stp = if iter == 0 {
            (1.0 / dnorm).min(STPMAX)
        } else {
            1.0
        };
        let (t, r, fold) = (x.clone(), g.clone(), fx);
        let gdold = dot(&g, &d);
        if gdold >= 0.0 {
            // Not a descent direction.
            if mem.pairs.is_empty() {
                return Ok(x);
            }
            mem.reset();
            continue;
        }
        let mut search = Dcsrch::start(fx, gdold, stp);
        let mut gd = gdold;
        let mut ifun = 0usize;
        let mut failed = false;
        loop {
            if ifun >= MAXLS {
                failed = true;
                break;
            }
            ifun += 1;
            x = if stp == 1.0 {
                z.clone()
            } else {
                d.iter().zip(&t).map(|(d, t)| stp * d + t).collect()
            };
            (fx, g) = eval(f, &x)?;
            gd = dot(&g, &d);
            if search.step(fx, gd, &mut stp) == Task::Done {
                break;
            }
        }
        if failed {
            x = t;
            g = r;
            fx = fold;
            if mem.pairs.is_empty() {
                return Ok(x);
            }
            mem.reset();
            continue;
        }
        iter += 1;
        if iter >= maxiter || pg(&g) <= PGTOL {
            return Ok(x);
        }
        let ddum = fold.abs().max(fx.abs()).max(1.0);
        if fold - fx <= TOL * ddum {
            return Ok(x);
        }
        let y: Vec<f64> = g.iter().zip(&r).map(|(a, b)| a - b).collect();
        let rr = dot(&y, &y);
        let (dr, ddum) = if stp == 1.0 {
            (gd - gdold, -gdold)
        } else {
            for v in &mut d {
                *v *= stp;
            }
            ((gd - gdold) * stp, -gdold * stp)
        };
        if dr <= f64::EPSILON * ddum {
            continue;
        }
        mem.theta = rr / dr;
        if mem.pairs.len() == M {
            mem.pairs.remove(0);
        }
        mem.pairs.push((d, y, dr));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// scipy 1.17's `fmin_l_bfgs_b(f, [-1.2, 1], approx_grad=True, epsilon=1e-5, maxiter=...)`
    /// stops at (0.9970099044310269, 0.9940237486541742) after 37 iterations, and at
    /// (-1.0213918623145783, 1.039304634101353) with `maxiter=5`.
    #[test]
    fn follows_scipy_on_rosenbrock() {
        let f = |x: &[f64]| Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2));
        let x = minimize(&f, &[-1.2, 1.0], 500).unwrap();
        assert!((x[0] - 0.997_009_904_431_026_9).abs() < 1e-9, "{x:?}");
        assert!((x[1] - 0.994_023_748_654_174_2).abs() < 1e-9, "{x:?}");
        let x = minimize(&f, &[-1.2, 1.0], 5).unwrap();
        assert!((x[0] + 1.021_391_862_314_578_3).abs() < 1e-9, "{x:?}");
        assert!((x[1] - 1.039_304_634_101_353).abs() < 1e-9, "{x:?}");
    }
}
