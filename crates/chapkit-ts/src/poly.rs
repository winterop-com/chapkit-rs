//! Polynomials in the lag operator: products, a stability test and roots.
//!
//! Coefficients run from the constant term up: `c[0] + c[1] z + ... + c[n] z^n`.

/// A complex number, for polynomial roots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    const ONE: Complex = Complex { re: 1.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// The modulus.
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }

    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }

    fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    fn div(self, o: Complex) -> Complex {
        let d = o.re * o.re + o.im * o.im;
        Complex::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }

    /// `1 / self`.
    pub fn recip(self) -> Complex {
        Complex::ONE.div(self)
    }
}

/// The product of two polynomials (R's `TSconv`).
pub fn conv(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// True when every root of `1 + c[0] z + c[1] z^2 + ...` lies outside the circle of radius `rho`
/// (no root has modulus `<= rho`). Exact zero high-order coefficients are dropped.
///
/// Scaled to the unit circle, this is the stationarity of the autoregression with coefficients
/// `-c[i] rho^(i+1)`, checked by running the Durbin-Levinson recursion backwards: every partial
/// autocorrelation must be inside (-1, 1).
pub fn roots_outside(c: &[f64], rho: f64) -> bool {
    let n = c.iter().rposition(|v| *v != 0.0).map_or(0, |i| i + 1);
    let mut phi: Vec<f64> = c
        .iter()
        .take(n)
        .enumerate()
        .map(|(i, v)| -v * rho.powi(i as i32 + 1))
        .collect();
    let mut work = phi.clone();
    for j in (0..n).rev() {
        let a = phi[j];
        if a.is_nan() || a.abs() >= 1.0 {
            return false;
        }
        for k in 0..j {
            work[k] = (phi[k] + a * phi[j - k - 1]) / (1.0 - a * a);
        }
        phi[..j].copy_from_slice(&work[..j]);
    }
    true
}

fn eval(c: &[f64], z: Complex) -> (Complex, Complex) {
    // Horner for p(z) and p'(z).
    let mut p = Complex::ZERO;
    let mut dp = Complex::ZERO;
    for &v in c.iter().rev() {
        dp = dp.mul(z).add(p);
        p = p.mul(z).add(Complex::new(v, 0.0));
    }
    (p, dp)
}

/// The roots of `c[0] + c[1] z + ... + c[n] z^n` (trailing zero coefficients dropped), by the
/// Aberth-Ehrlich iteration. Empty for a constant.
pub fn roots(c: &[f64]) -> Vec<Complex> {
    let n = c.iter().rposition(|v| *v != 0.0).unwrap_or(0);
    if n == 0 {
        return Vec::new();
    }
    let c = &c[..=n];
    // Zero roots at the origin first (c[0] == 0), as many as leading zeros.
    let zeros = c.iter().take_while(|v| **v == 0.0).count();
    let c = &c[zeros..];
    let m = c.len() - 1;
    let mut out = vec![Complex::ZERO; zeros];
    if m == 0 {
        return out;
    }
    if m == 1 {
        out.push(Complex::new(-c[0] / c[1], 0.0));
        return out;
    }
    let radius = (c[0] / c[m])
        .abs()
        .powf(1.0 / m as f64)
        .max(f64::MIN_POSITIVE);
    let mut z: Vec<Complex> = (0..m)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / m as f64 + 0.4;
            Complex::new(radius * angle.cos(), radius * angle.sin())
        })
        .collect();
    for _ in 0..500 {
        let mut moved = 0.0f64;
        for i in 0..m {
            let (p, dp) = eval(c, z[i]);
            if p.norm() == 0.0 {
                continue;
            }
            let ratio = p.div(dp);
            let mut sum = Complex::ZERO;
            for (j, zj) in z.iter().enumerate() {
                if j != i {
                    sum = sum.add(z[i].sub(*zj).recip());
                }
            }
            let w = ratio.div(Complex::ONE.sub(ratio.mul(sum)));
            if w.re.is_finite() && w.im.is_finite() {
                z[i] = z[i].sub(w);
                moved = moved.max(w.norm() / z[i].norm().max(1e-300));
            }
        }
        if moved < 1e-15 {
            break;
        }
    }
    out.extend(z);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_multiplies() {
        assert_eq!(conv(&[1.0, -1.0], &[1.0, -1.0]), vec![1.0, -2.0, 1.0]);
    }

    #[test]
    fn roots_of_known_polynomials() {
        // (z - 2)(z + 0.5) = z^2 - 1.5 z - 1
        let mut r: Vec<f64> = roots(&[-1.0, -1.5, 1.0]).iter().map(|z| z.re).collect();
        r.sort_by(f64::total_cmp);
        assert!((r[0] + 0.5).abs() < 1e-12 && (r[1] - 2.0).abs() < 1e-12);
        // z^2 + 1: +-i
        let r = roots(&[1.0, 0.0, 1.0]);
        assert!(
            r.iter()
                .all(|z| (z.norm() - 1.0).abs() < 1e-12 && z.re.abs() < 1e-12)
        );
        assert_eq!(roots(&[0.0, 1.0]).len(), 1);
    }

    #[test]
    fn stability() {
        // 1 - 0.5 z: root 2.
        assert!(roots_outside(&[-0.5], 1.0));
        assert!(roots_outside(&[-0.5], 1.99));
        assert!(!roots_outside(&[-0.5], 2.01));
        // 1 - 1.2 z + 0.5 z^2: complex roots of modulus sqrt(2).
        assert!(roots_outside(&[-1.2, 0.5], 1.41));
        assert!(!roots_outside(&[-1.2, 0.5], 1.42));
        assert!(roots_outside(&[0.0, 0.0], 1.0));
    }
}
