//! An online mean/variance accumulator (Welford), mergeable so summaries
//! combine across restart segments and file boundaries without revisiting
//! points — the "Moments law" the whole summary layer rests on.

/// Streaming mean and variance over finite values.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Moments {
    count: u64,
    mean: f64,
    m2: f64,
}

impl Moments {
    /// Accumulates one value. Non-finite values are the caller's to exclude.
    pub fn push(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (value - self.mean);
    }

    /// Merges another accumulator into this one (Chan et al.'s parallel
    /// formula): the result is as if every value had been pushed here.
    pub fn merge(&mut self, other: &Moments) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = *other;
            return;
        }
        let total = self.count + other.count;
        let delta = other.mean - self.mean;
        self.mean += delta * other.count as f64 / total as f64;
        self.m2 += other.m2 + delta * delta * (self.count as f64 * other.count as f64) / total as f64;
        self.count = total;
    }

    /// Number of accumulated values.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// The mean, or `None` before any value.
    pub fn mean(&self) -> Option<f64> {
        (self.count > 0).then_some(self.mean)
    }

    /// The population variance, or `None` before any value.
    pub fn variance(&self) -> Option<f64> {
        (self.count > 0).then_some((self.m2 / self.count as f64).max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn direct(values: &[f64]) -> (f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64;
        (mean, variance)
    }

    #[test]
    fn matches_direct_computation() {
        let values = [1.5, -2.0, 0.25, 8.0, 3.5, -0.75];
        let mut moments = Moments::default();
        for &value in &values {
            moments.push(value);
        }
        let (mean, variance) = direct(&values);
        assert!((moments.mean().unwrap() - mean).abs() < 1e-12);
        assert!((moments.variance().unwrap() - variance).abs() < 1e-12);
        assert_eq!(moments.count(), 6);
    }

    #[test]
    fn merge_equals_sequential() {
        let values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        for split in 0..=values.len() {
            let mut left = Moments::default();
            let mut right = Moments::default();
            for &value in &values[..split] {
                left.push(value);
            }
            for &value in &values[split..] {
                right.push(value);
            }
            left.merge(&right);
            let mut sequential = Moments::default();
            for &value in &values {
                sequential.push(value);
            }
            assert_eq!(left.count(), sequential.count());
            assert!((left.mean().unwrap() - sequential.mean().unwrap()).abs() < 1e-12);
            assert!((left.variance().unwrap() - sequential.variance().unwrap()).abs() < 1e-12);
        }
    }

    #[test]
    fn empty_is_none() {
        let moments = Moments::default();
        assert_eq!(moments.mean(), None);
        assert_eq!(moments.variance(), None);
    }
}
