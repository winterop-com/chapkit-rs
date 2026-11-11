//! R's `optim(method = "BFGS")` with its default controls and numerical gradient: `vmmin` (J.C.
//! Nash's variable-metric method as in R's `src/appl/optim.c`), central differences with
//! `ndeps = 1e-3` on the scaled parameters, `reltol = sqrt(.Machine$double.eps)`, `abstol = -Inf`.

use rayon::prelude::*;

use crate::{Error, Result};

const STEPREDN: f64 = 0.2;
const ACCTOL: f64 = 0.0001;
const RELTEST: f64 = 10.0;
const NDEPS: f64 = 1e-3;
/// R's default `reltol`, `sqrt(.Machine$double.eps)`.
const RELTOL: f64 = 1.490_116_119_384_765_6e-8;

/// What `optim` returns.
pub(crate) struct Optimum {
    pub par: Vec<f64>,
    pub value: f64,
    /// `optim`'s `convergence == 0`: false when `maxit` was reached.
    pub converged: bool,
}

struct Scaled<'a, F> {
    f: &'a F,
    parscale: &'a [f64],
}

impl<F: Fn(&[f64]) -> Result<f64> + Sync> Scaled<'_, F> {
    fn value(&self, p: &[f64]) -> Result<f64> {
        let mut x = Vec::with_capacity(p.len());
        for (p, s) in p.iter().zip(self.parscale) {
            if !p.is_finite() {
                return Err(Error::estimation("non-finite value supplied by optim"));
            }
            x.push(p * s);
        }
        (self.f)(&x)
    }

    /// Central differences, one coordinate per task (the evaluations are independent, so the
    /// result is the same as R's sequential loop).
    fn gradient(&self, p: &[f64], df: &mut [f64]) -> Result<()> {
        let parts: Vec<Result<f64>> = (0..p.len())
            .into_par_iter()
            .map(|i| {
                let mut q = p.to_vec();
                q[i] = p[i] + NDEPS;
                let v1 = self.value(&q)?;
                q[i] = p[i] - NDEPS;
                let v2 = self.value(&q)?;
                let g = (v1 - v2) / (2.0 * NDEPS);
                if g.is_finite() {
                    Ok(g)
                } else {
                    Err(Error::estimation(format!(
                        "non-finite finite-difference value [{}]",
                        i + 1
                    )))
                }
            })
            .collect();
        for (d, g) in df.iter_mut().zip(parts) {
            *d = g?;
        }
        Ok(())
    }
}

/// Minimise `f` from `par` (in the caller's units; `parscale` divides them for the optimiser).
pub(crate) fn bfgs<F: Fn(&[f64]) -> Result<f64> + Sync>(
    f: &F,
    par: &[f64],
    parscale: &[f64],
    maxit: usize,
) -> Result<Optimum> {
    let n = par.len();
    let s = Scaled { f, parscale };
    let mut b: Vec<f64> = par.iter().zip(parscale).map(|(p, s)| p / s).collect();
    let unscale = |b: &[f64]| -> Vec<f64> { b.iter().zip(parscale).map(|(b, s)| b * s).collect() };
    if maxit == 0 {
        let value = s.value(&b)?;
        return Ok(Optimum {
            par: unscale(&b),
            value,
            converged: true,
        });
    }
    let mut g = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut x = vec![0.0; n];
    let mut c = vec![0.0; n];
    // Lower triangle of the inverse Hessian approximation.
    let mut bm = vec![vec![0.0; n]; n];
    let mut f = s.value(&b)?;
    if !f.is_finite() {
        return Err(Error::estimation("initial value in 'vmmin' is not finite"));
    }
    let mut fmin = f;
    let mut gradcount = 1usize;
    s.gradient(&b, &mut g)?;
    let mut iter = 1usize;
    let mut ilast = gradcount;
    loop {
        if ilast == gradcount {
            for (i, row) in bm.iter_mut().enumerate() {
                for v in row.iter_mut().take(i) {
                    *v = 0.0;
                }
                row[i] = 1.0;
            }
        }
        x.copy_from_slice(&b);
        c.copy_from_slice(&g);
        let mut gradproj = 0.0;
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..=i {
                sum -= bm[i][j] * g[j];
            }
            for j in i + 1..n {
                sum -= bm[j][i] * g[j];
            }
            t[i] = sum;
            gradproj += sum * g[i];
        }
        let mut count;
        if gradproj < 0.0 {
            let mut steplength = 1.0;
            let mut accpoint = false;
            loop {
                count = 0;
                for i in 0..n {
                    b[i] = x[i] + steplength * t[i];
                    if RELTEST + x[i] == RELTEST + b[i] {
                        count += 1;
                    }
                }
                if count < n {
                    f = s.value(&b)?;
                    accpoint = f.is_finite() && f <= fmin + gradproj * steplength * ACCTOL;
                    if !accpoint {
                        steplength *= STEPREDN;
                    }
                }
                if count == n || accpoint {
                    break;
                }
            }
            // abstol is -Inf: only the relative change stops.
            let enough = f > f64::NEG_INFINITY && (f - fmin).abs() > RELTOL * (fmin.abs() + RELTOL);
            if !enough {
                count = n;
                fmin = f;
            }
            if count < n {
                fmin = f;
                s.gradient(&b, &mut g)?;
                gradcount += 1;
                iter += 1;
                let mut d1 = 0.0;
                for i in 0..n {
                    t[i] *= steplength;
                    c[i] = g[i] - c[i];
                    d1 += t[i] * c[i];
                }
                if d1 > 0.0 {
                    let mut d2 = 0.0;
                    for i in 0..n {
                        let mut sum = 0.0;
                        for j in 0..=i {
                            sum += bm[i][j] * c[j];
                        }
                        for j in i + 1..n {
                            sum += bm[j][i] * c[j];
                        }
                        x[i] = sum;
                        d2 += sum * c[i];
                    }
                    d2 = 1.0 + d2 / d1;
                    for i in 0..n {
                        for j in 0..=i {
                            bm[i][j] += (d2 * t[i] * t[j] - x[i] * t[j] - t[i] * x[j]) / d1;
                        }
                    }
                } else {
                    ilast = gradcount;
                }
            } else if ilast < gradcount {
                count = 0;
                ilast = gradcount;
            }
        } else {
            count = 0;
            if ilast == gradcount {
                count = n;
            } else {
                ilast = gradcount;
            }
        }
        if iter >= maxit {
            break;
        }
        if gradcount - ilast > 2 * n {
            ilast = gradcount;
        }
        if count == n && ilast == gradcount {
            break;
        }
    }
    Ok(Optimum {
        par: unscale(&b),
        value: fmin,
        converged: iter < maxit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimises_a_quadratic() {
        let f = |p: &[f64]| Ok((p[0] - 3.0).powi(2) + 10.0 * (p[1] + 1.0).powi(2));
        let o = bfgs(&f, &[0.0, 0.0], &[1.0, 1.0], 100).unwrap();
        assert!(o.converged);
        assert!((o.par[0] - 3.0).abs() < 1e-4 && (o.par[1] + 1.0).abs() < 1e-4);
    }
}
