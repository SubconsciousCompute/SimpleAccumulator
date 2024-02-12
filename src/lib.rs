//! A library for incremental statistical computation inspired by
//! [Boost.Accumulators](https://www.boost.org/doc/libs/1_84_0/doc/html/accumulators.html).
//!
//! ```rust
//!     let k = [1.0, 2.0, 3.0, 4.0];
//!
//!     // Creates an accumulator that stores maximum of last 4 elements.
//!     let mut x = simple_accumulator::SimpleAccumulator::new(&k, Some(4));
//!
//!     println!("{:?}", x);
//!     x.push(5.0);
//!
//!     println!("{:?}", x);
//! ```

use std::collections::VecDeque;
use std::ops::{AddAssign, SubAssign};

#[cfg(feature = "histogram")]
use histogram::Histogram;

use num_traits::{cast::FromPrimitive, float::Float};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use watermill::kurtosis::Kurtosis;
use watermill::maximum::Max;
use watermill::mean::Mean;
use watermill::minimum::Min;
use watermill::quantile::Quantile;
use watermill::skew::Skew;
use watermill::stats::Univariate;
use watermill::sum::Sum;
use watermill::variance::Variance;

/// Accumulator
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimpleAccumulator<
    F: Float + FromPrimitive + AddAssign + SubAssign + std::default::Default,
> {
    /// Vec to store the data
    pub data: VecDeque<F>,

    /// Running counter of elements seen
    /// same as self.len() in case of unbounded capacity
    pub total: usize,

    /// Running mean
    mean: Mean<F>,

    /// Middle element. We use a rough estimate when using `accumulate=true`
    median: Quantile<F>,

    /// Running variance
    variance: Variance<F>,

    /// Minimum element in the Accumulator
    min: Min<F>,

    /// Maximum element in the Accumulator
    max: Max<F>,

    /// Sum of all elements.
    sum: Sum<F>,

    /// Measure of bias in the population. Population follows a Poisson distribution.
    skewness: Skew<F>,

    /// Measure of the tail length of the distribution
    kurtosis: Kurtosis<F>,

    /// Can only `push` if used, for `pop` and `remove` we return `None`
    fixed_capacity: bool,

    /// Histogram
    #[serde(skip)]
    #[cfg(feature = "histogram")]
    histogram: Option<Histogram>,
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign + std::default::Default>
    SimpleAccumulator<F>
{
    /// Create a new Accumulator with some data. If `max_size` is `None`, the internal container
    /// holding data stores all data else the buffer size never grows past the given size.
    pub fn new(slice: &[F], max_size: Option<usize>) -> Self {
        let data = if let Some(capacity) = max_size {
            VecDeque::with_capacity(capacity)
        } else {
            VecDeque::new()
        };

        let mut k = SimpleAccumulator {
            data,
            fixed_capacity: max_size.is_some(),
            min: Min::new(), // Min::default() will set min to 0 and not to f64::MAX
            ..Default::default()
        };

        for &v in slice.into_iter() {
            k.push(v);
        }
        k
    }

    /// Initialize a histogram The configuration of a histogram which determines the bucketing
    /// strategy and therefore the relative error and memory utilization of a histogram.
    ///
    /// `grouping_power` - controls the number of buckets that are used to span consecutive powers
    /// of two. Lower values result in less memory usage since fewer buckets will be created.
    /// However, this will result in larger relative error as each bucket represents a wider range
    /// of values.
    ///
    /// `max_value_power` - controls the largest value which can be stored in the histogram.
    /// 2^(max_value_power) - 1 is the inclusive upper bound for the representable range of values.
    ///
    /// Reference: <https://docs.rs/histogram/latest/histogram/struct.Config.html>
    #[cfg(feature = "histogram")]
    pub fn init_histogram(&mut self, grouping_power: u8, max_value_power: u8) {
        assert!(
            grouping_power < max_value_power,
            "max_value_power must be > grouping_power"
        );
        if self.histogram.is_some() {
            tracing::info!("Histogram is already initialize. Reinitializing...");
        }
        self.histogram = match histogram::Histogram::new(grouping_power, max_value_power) {
            Ok(hist) => Some(hist),
            Err(e) => {
                tracing::warn!("Failed to initialize histogram: {e}");
                None
            }
        }
    }

    /// Get the length of underlying container storing data.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the capacity of underlying container storing data.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Can be made of any type `&[T]` but will be converted to `Vec<f64>`, panics on values that
    /// cannot be converted.
    ///
    /// Panics if the provided `slice` has greater number of elements than provided `capacity`
    ///
    ///     use simple_accumulator::SimpleAccumulator;
    ///     const CAPACITY: usize = 3;
    ///     let mut acc = SimpleAccumulator::with_fixed_capacity(&[], CAPACITY);
    ///
    ///     let data = vec![0.0, 1.1, 2.2, 3.3, 4.4];
    ///     for &v in data.iter() {
    ///         acc.push(v);
    ///     }
    ///     println!("{acc:?}");
    ///     assert_eq!(acc.len(), CAPACITY);
    ///     assert_eq!(acc.data, vec![2.2, 3.3, 4.4]);
    ///
    ///     acc.push(5.5);
    ///     assert_eq!(acc.len(), CAPACITY);
    ///     assert_eq!(acc.data, vec![3.3, 4.4, 5.5]);
    ///
    ///     acc.push(6.6);
    ///     assert_eq!(acc.len(), CAPACITY);
    ///     assert_eq!(acc.data, vec![4.4, 5.5, 6.6]);
    pub fn with_fixed_capacity(slice: &[F], capacity: usize) -> SimpleAccumulator<F> {
        assert!(
            slice.len() <= capacity,
            "Capacity less than length of given slice"
        );
        Self::new(slice, Some(capacity))
    }
}

impl<F: Float + FromPrimitive + AddAssign + SubAssign + std::default::Default>
    SimpleAccumulator<F>
{
    /// Function similar to `push` in vector `Vec`. When `fixed_capacity` is 'true' and ring buffer is full,
    /// the function rewrites the oldest element with the latest, following FIFO order.
    /// 3 different cases arise when:
    /// 1. capacity is not fixed,
    /// 2. capacity is fixed but the buffer is not full and
    /// 3. the buffer has fixed capacity and is full.
    /// In the first two cases, the native push function for vectors is used to add the new element, all stats are updated
    /// online and the number of data points incremented.
    /// In the third case, we replace the oldest element by the new element (FIFO order). All stats are updated online
    /// and the number of elements remains equal to the buffer capacity.
    ///
    pub fn push(&mut self, y: F) {
        // Running stats, Number of elements seen is incremented irrespective of buffer properties
        // Calculation is online following Knuth's algorithm
        self.total += 1;

        self.min.update(y);
        self.max.update(y);
        self.sum.update(y);
        self.mean.update(y);
        self.median.update(y);
        self.variance.update(y);
        self.skewness.update(y);
        self.kurtosis.update(y);

        // we just change the already held value and keep on rewriting it
        if self.fixed_capacity {
            if self.data.len() == self.data.capacity() {
                self.data.pop_front();
            }
        }
        self.data.push_back(y);

        #[cfg(feature = "histogram")]
        if let Some(histogram) = self.histogram.as_mut() {
            if let Some(v) = y.to_u64() {
                if let Err(e) = histogram.increment(v) {
                    debug_assert!(false, "Failed to increment the histogram: {e}");
                }
            }
        }
    }

    /// Return reference to the inner Histogram
    #[cfg(feature = "histogram")]
    pub fn histogram(&self) -> Option<&histogram::Histogram> {
        self.histogram.as_ref()
    }

    /// Function similar to `append` in `Vec`, rewrites in FIFO order if `fixed_capacity` is 'true'.
    /// Similar to push, this function deals with 3 cases:
    /// 1. capacity is not fixed,
    /// 2. capacity is fixed but the buffer has space to accommodate the input
    /// 3. capacity is fixed and some elements at the end of the input vector has to replace
    /// oldest elements in the buffer.
    /// In the first two cases, the native append function is used, and number of data points updated.
    /// For the third case: Assuming the input vector is longer than the ring buffer size, this function
    /// skips writing the elements up to the `vector length - buffer length -1` position in
    /// the input vector. Starting from the position `last write + 1` in the
    /// buffer, the function fills it with the remaining elements of the vector.
    /// Stats requiring complex calculations are computed in the offline method.
    /// Mean and variance are computed online.
    ///
    pub fn append(&mut self, value: &[F]) {
        for t in value {
            self.push(*t)
        }
    }

    #[inline]
    pub fn mean(&self) -> F {
        self.mean.get()
    }

    #[inline]
    pub fn variance(&self) -> F {
        self.variance.get()
    }

    #[inline]
    pub fn median(&self) -> F {
        self.median.get()
    }

    #[inline]
    pub fn min(&self) -> F {
        self.min.get()
    }

    #[inline]
    pub fn max(&self) -> F {
        self.max.get()
    }

    #[inline]
    pub fn sum(&self) -> F {
        self.sum.get()
    }
}

#[cfg(test)]
mod tests {
    use num_traits::ToPrimitive;

    #[test]
    fn test_f32_to_u64() {
        let a = 40.9f32;
        let ai = a.to_u64().unwrap();
        println!("{a} {ai}");
        assert_eq!(a.floor() as u64, ai);

        for _i in 0..10000 {
            let a = rand::random::<f64>() * 100.0;
            let ai = a.to_u64().unwrap();
            assert_eq!(a.floor() as u64, ai, "floor or {a} is not equal to {ai}");
        }
    }
}
