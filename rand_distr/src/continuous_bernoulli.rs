// Copyright 2018 Developers of the Rand project.
// Copyright 2013 The Rust Project Developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The continuous [Bernoulli distribution `ContinuousBernoulli(l)`]
//! (https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution).

#![cfg(feature = "alloc")]
use crate::{Distribution, Open01};
use core::fmt;
use num_traits::Float;
use rand::Rng;
#[cfg(feature = "serde")]
use serde_with::serde_as;

/// The [continuous Bernoulli distribution](https://en.wikipedia.org/wiki/Continuous_Bernoulli_distribution) `ContinuousBernoulli(l)`.
///
/// The Dirichlet distribution is a family of continuous multivariate
/// probability distributions parameterized by a vector of positive
/// real numbers `α₁, α₂, ..., αₖ`, where `k` is the number of dimensions
/// of the distribution. The distribution is supported on the `k-1`-dimensional
/// simplex, which is the set of points `x = [x₁, x₂, ..., xₖ]` such that
/// `0 ≤ xᵢ ≤ 1` and `∑ xᵢ = 1`.
/// It is a multivariate generalization of the [`Beta`](crate::Beta) distribution.
/// The distribution is symmetric when all `αᵢ` are equal.
///
/// # Plot
///
/// The following plot illustrates the 2-dimensional simplices for various
/// 3-dimensional Dirichlet distributions.
///
/// ![Dirichlet distribution](https://raw.githubusercontent.com/rust-random/charts/main/charts/dirichlet.png)
///
/// # Example
///
/// ```
/// use rand::prelude::*;
/// use rand_distr::Dirichlet;
///
/// let dirichlet = Dirichlet::new([1.0, 2.0, 3.0]).unwrap();
/// let samples = dirichlet.sample(&mut rand::rng());
/// println!("{:?} is from a Dirichlet([1.0, 2.0, 3.0]) distribution", samples);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ContinuousBernoulli<F>
where
    F: Float,
    Open01: Distribution<F>,
{
    fraction_numerator: F,
    inner_denominator_log2: F,
    lambda: F,
}

/// Error type that can be the result of calling [`ContinuousBernoulli::new`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// Shape is not > 0
    ShapeTooSmall,
    /// Shape is not < 1
    ShapeTooLarge,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Error::ShapeTooSmall => "shape is too small",
            Error::ShapeTooLarge => "shape is too large",
        })
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl<F> ContinuousBernoulli<F>
where
    F: Float,
    Open01: Distribution<F>,
{
    /// Constructs a new instance representing `ContinuousBernoulli(lambda)`
    #[inline]
    pub fn new(lambda: F) -> Result<ContinuousBernoulli<F>, Error> {
        if !(lambda > F::zero()) {
            return Err(Error::ShapeTooSmall);
        }
        if !(lambda < F::one()) {
            return Err(Error::ShapeTooLarge);
        }

        let fraction_numerator = ((F::one() + F::one()) * lambda - F::one()) / (F::one() - lambda);
        let inner_denominator_log2 = (lambda / (F::one() - lambda)).log2();
        Ok(ContinuousBernoulli {
            lambda,
            fraction_numerator,
            inner_denominator_log2,
        })
    }
}

impl<F> Distribution<F> for ContinuousBernoulli<F>
where
    F: Float,
    Open01: Distribution<F>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> F {
        let y = rng.sample(Open01);
        debug_assert!(y > F::zero() && y < F::one());

        let inner_numerator = F::one() + self.fraction_numerator * y;
        inner_numerator.log2() / self.inner_denominator_log2
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn continuous_bernoulli_expected_mean(lambda: f64) -> f64 {
        assert!((0.0..=1.0).contains(&lambda));
        let first = lambda / (2.0 * lambda - 1.0);
        let second = 1.0 / (2.0 * (1.0 - 2.0 * lambda).atanh());
        first + second
    }

    #[test]
    fn continuous_bernoulli_has_expected_mean() {
        const ITERATIONS: usize = 100_000;
        for lambda in [0.2, 0.26, 0.31, 0.37, 0.4, 0.55, 0.61, 0.7] {
            let distr =
                super::ContinuousBernoulli::new(lambda).expect("Could not create distribution");
            let average = distr
                .sample_iter(crate::test::rng(1787569))
                .take(ITERATIONS)
                .fold(0.0, |acc, x| acc + x / (ITERATIONS as f64));
            let expected = continuous_bernoulli_expected_mean(lambda);
            assert_almost_eq!(average, expected, 0.01);
        }
    }

    #[test]
    fn continuous_bernoulli_can_be_compared() {
        assert_eq!(ContinuousBernoulli::new(0.5), ContinuousBernoulli::new(0.5));
    }
}
