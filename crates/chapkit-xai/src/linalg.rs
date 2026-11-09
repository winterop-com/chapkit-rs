//! Weighted least squares via normal equations.

/// Weighted least squares of `y` on design rows `x`. `None` if singular.
pub(crate) fn weighted_least_squares(x: &[Vec<f64>], y: &[f64], w: &[f64]) -> Option<Vec<f64>> {
    let p = x.first()?.len();
    let mut a = vec![0.0; p * p];
    let mut b = vec![0.0; p];
    for ((row, yi), wi) in x.iter().zip(y).zip(w) {
        for r in 0..p {
            let xr = *row.get(r)?;
            b[r] += wi * xr * yi;
            for c in 0..p {
                a[r * p + c] += wi * xr * row.get(c)?;
            }
        }
    }
    solve(a, b)
}

/// Solve `a x = b` (row-major `n x n`) by Gaussian elimination with partial pivoting.
#[allow(clippy::many_single_char_names)]
pub(crate) fn solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))?;
        if a[pivot * n + col].abs() < 1e-12 {
            return None;
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        for row in col + 1..n {
            let f = a[row * n + col] / a[col * n + col];
            for k in col..n {
                a[row * n + k] -= f * a[col * n + k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|k| a[row * n + k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row * n + row];
    }
    Some(x)
}
