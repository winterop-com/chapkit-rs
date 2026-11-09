//! A small deterministic RNG (splitmix64) with the helpers models, explainers,
//! search and data generators need. No dependencies; the same seed gives the same
//! stream on every platform.

/// Seedable pseudo-random generator (splitmix64). Not for cryptography.
#[derive(Debug, Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    /// A generator from a seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    /// A generator seeded from the clock (different stream each call).
    pub fn from_time() -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        Self::new(nanos)
    }

    /// Next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[low, high)`.
    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.unit()
    }

    /// Uniform integer in `[low, high]`.
    pub fn int_range(&mut self, low: i64, high: i64) -> i64 {
        if high <= low {
            return low;
        }
        let span = (high - low) as u64 + 1;
        low + (self.next_u64() % span) as i64
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-12);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Normal with mean `mu` and standard deviation `sigma`.
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        mu + sigma * self.normal()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_in_range() {
        let (mut a, mut b) = (Rng::new(7), Rng::new(7));
        let sa: Vec<u64> = (0..100).map(|_| a.next_u64()).collect();
        let sb: Vec<u64> = (0..100).map(|_| b.next_u64()).collect();
        assert_eq!(sa, sb);
        for _ in 0..1000 {
            let u = a.unit();
            assert!((0.0..1.0).contains(&u));
            let i = a.int_range(3, 5);
            assert!((3..=5).contains(&i));
        }
        b.next_u64();
        let mean: f64 = (0..20_000).map(|_| b.gauss(10.0, 2.0)).sum::<f64>() / 20_000.0;
        assert!((mean - 10.0).abs() < 0.1);
    }
}
