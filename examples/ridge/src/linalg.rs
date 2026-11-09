use chapkit::{Error, Result};

/// Mean and (population) standard deviation of the finite values; `None` if there are none.
pub(crate) fn mean_std(xs: impl IntoIterator<Item = f64>) -> Option<(f64, f64)> {
    let (mut n, mut sum, mut sq) = (0usize, 0.0, 0.0);
    for x in xs.into_iter().filter(|x| x.is_finite()) {
        n += 1;
        sum += x;
        sq += x * x;
    }
    (n > 0).then(|| {
        let m = sum / n as f64;
        (m, (sq / n as f64 - m * m).max(0.0).sqrt())
    })
}

/// Solve `a * x = b` in place by Gaussian elimination with partial pivoting.
/// `a` is row-major `n x n`.
#[allow(clippy::many_single_char_names)]
pub(crate) fn solve(mut a: Vec<f64>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i * n + col].abs().total_cmp(&a[j * n + col].abs()))
            .unwrap_or(col);
        if a[pivot * n + col].abs() < 1e-12 {
            return Err(Error::model("singular system; increase alpha"));
        }
        if pivot != col {
            for k in 0..n {
                a.swap(col * n + k, pivot * n + k);
            }
            b.swap(col, pivot);
        }
        for row in col + 1..n {
            let f = a[row * n + col] / a[col * n + col];
            if f != 0.0 {
                for k in col..n {
                    a[row * n + k] -= f * a[col * n + k];
                }
                b[row] -= f * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let s: f64 = (row + 1..n).map(|k| a[row * n + k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row * n + row];
    }
    Ok(x)
}
