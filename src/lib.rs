//! Store and update stats related to our data array without iterating again and again
//!
//! ```rust
//!     let k = [1, 2, 3, 4];
//!
//!     let mut x = simple_accumulator::SimpleAccumulator::new(&k, true);
//!
//!     println!("{:#?}", x);
//!     x.push(5);
//!
//!     println!("{:#?}", x);
//!
//!     x.pop();
//!     println!("{:#?}", x);
//!
//!     x.remove(2);
//!     println!("{:#?}", x);
//! ```
//!
//! Set field `accumulate` to `false` to not update the value, you will need to run `calculate_all` to
//! get the updated field values
//!
//! If `with_fixed_capacity` is used then we rewrite the current buffer in FIFO order

#![allow(suspicious_double_ref_op)]
//pub use self::SimpleAccumulator;

use num::ToPrimitive;
use std::cmp::Ordering;
//use float_eq::AssertFloatEq;
// use std::collections::HashMap;

/// Our main data struct
#[derive(Clone, Default, Debug, PartialEq)]
pub struct SimpleAccumulator {
    /// Vec to store the data
    pub vec: Vec<f64>,
    /// Vec to privately store mean and three moment differences
    stats: Vec<f64>,
    /// Running mean
    pub mean: f64,
    /// Running variance
    pub variance: f64,
    /// Running counter of elements seen
    /// same as self.len in case of unbounded capacity
    pub total: usize,
    /// Average/mean of the accumulator data
    /// Same as running mean when capacity is not fixed
    pub(crate) buffer_mean: f64,
    /// Variance of the accumulator data, uses `N` not `N-1`
    pub(crate) buffer_variance: f64,
    /*
    /// (Standard deviation)^2 = variance
    pub standard_deviation: f64,
    */
    /// Minimum element in the Accumulator
    pub min: f64,
    /// 2nd lowest value - To help calculate approx min
    min_: f64,
    /// Maximum element in the Accumulator
    pub max: f64,
    /// 2nd highest value - To help calculate approx max
    max_: f64,
    /// Middle element. We use a rough estimate when using `accumulate=true`
    pub median: f64,
    // mode: f64,
    /// Current `length` or number of elements currently stored
    pub len: usize,
    /// Capacity available before it has to reallocate more, doesn't reallocate more if `with_fixed_capacity`
    /// is used - instead rewrites previous places in FIFO order
    pub capacity: usize,
    /// Can only `push` if used, for `pop` and `remove` we return `None`
    pub fixed_capacity: bool,
    /// Gives an idea about last filled position, doesn't get updated if `accumulate=true`
    pub last_write_position: usize,
    /// Flag to set whether the fields update or not after a change(push, remove, pop)
    pub accumulate: bool,
    /// Measure of bias in the population. Population follows a Poisson distribution.
    pub skewness: f64,
    // Measure of the tail length of the distribution
    pub kurtosis: f64,
    // Measure of two peaks existing in the distribution
    pub bimodality: f64,
}

impl SimpleAccumulator {
    /// Input to this function can be of generic type `&[T]` but will be converted to `Vec<f64>`. Panic on values that
    /// cannot be converted.
    /// The function initialises the individual variables inside `struct SimpleAccumulator`.
    /// Calls the function `calculate_all` to computes the values of all statistical measures and variables.
    ///
    pub fn new<T: ToPrimitive>(slice: &[T], flag: bool) -> Self {
        let vec: Vec<f64> = slice
            .iter()
            .map(|x| T::to_f64(x).expect("Not a number"))
            .collect();

        let stats: Vec<f64> = vec![0.0; 4];

        let mut k = SimpleAccumulator {
            vec,
            stats,
            mean: 0.0,
            variance: 0.0,
            total: 0,
            buffer_mean: 0.0,
            buffer_variance: 0.0,
            min: 0.0,
            min_: f64::INFINITY,
            max: 0.0,
            max_: f64::NEG_INFINITY,
            median: 0.0,
            // mode: 0.0,
            len: 0,
            capacity: 0,
            fixed_capacity: false,
            last_write_position: 0,
            accumulate: flag,
            skewness: 0.0,
            kurtosis: 0.0,
            bimodality: 0.0,
        };

        if !k.vec.is_empty() {
            k.len = k.vec.len();
            k.capacity = k.vec.capacity();
            // Mean and variance is computed to initialise the running values
            // even when accumulate flag is off
            if flag {
                k.calculate_all();
            } else {
                k.calculate_mean();
                k.calculate_variance();
            }

            k.mean = k.buffer_mean;
            k.variance = k.buffer_variance;
            k.total = k.len;
            // k.calculate_mode();
        }
        k
    }

    /// Can be made of any type `&[T]` but will be converted to `Vec<f64>`, panics on values that
    /// cannot be converted.
    ///
    /// Panics if the provided `slice` has greater number of elements than provided `capacity`
    ///
    ///     use simple_accumulator::SimpleAccumulator;
    ///     const CAPACITY: usize = 3;
    ///     let mut acc = SimpleAccumulator::with_fixed_capacity::<f64>(&[], CAPACITY, true);
    ///
    ///     let data = vec![0.0, 1.1, 2.2, 3.3, 4.4];
    ///     for &v in &data {
    ///         acc.push(v);
    ///     }
    ///     println!("{acc:?}");
    ///     assert_eq!(acc.vec.len(), CAPACITY);
    ///     assert_eq!(acc.vec, vec![3.3, 4.4, 2.2]);
    ///
    ///     acc.push(5.5);
    ///     assert_eq!(acc.vec.len(), CAPACITY);
    ///     assert_eq!(acc.vec, vec![3.3, 4.4, 5.5]);
    ///
    ///     acc.push(6.6);
    ///     assert_eq!(acc.vec.len(), CAPACITY);
    ///     assert_eq!(acc.vec, vec![6.6, 4.4, 5.5]);
    pub fn with_fixed_capacity<T: ToPrimitive>(slice: &[T], capacity: usize, flag: bool) -> Self {
        assert!(
            slice.len() <= capacity,
            "Capacity less than length of given slice"
        );

        let mut vec: Vec<f64> = slice.iter().map(|x| T::to_f64(x).unwrap()).collect();

        let stats: Vec<f64> = vec![0.0; 4];

        vec.reserve_exact(capacity);

        if slice.is_empty() {
            vec = Vec::with_capacity(capacity);
        }

        let mut k = SimpleAccumulator {
            vec,
            stats,
            mean: 0.0,
            variance: 0.0,
            total: 0,
            buffer_mean: 0.0,
            buffer_variance: 0.0,
            min: 0.0,
            min_: f64::INFINITY,
            max: 0.0,
            max_: f64::NEG_INFINITY,
            median: 0.0,
            // mode: 0.0,
            len: 0,
            capacity,
            fixed_capacity: true,
            last_write_position: 0,
            accumulate: flag,
            skewness: 0.0,
            kurtosis: 0.0,
            bimodality: 0.0,
        };

        if !k.vec.is_empty() {
            k.last_write_position = k.vec.len() - 1;
            k.len = k.vec.len();
            // Mean and variance is computed to initialise the running values
            // even when accumulate flag is off
            if flag {
                k.calculate_all();
            } else {
                k.calculate_mean();
                k.calculate_variance();
            }
            // Initially running values are same as buffer values
            k.mean = k.buffer_mean;
            k.variance = k.buffer_variance;
            k.total = k.len;
            // k.calculate_mode();
        }
        k
    }

    pub fn calculate_all(&mut self) {
        if self.len == 0 {
            return;
        }
        self.calculate_mean();
        self.calculate_variance();
        // self.calculate_standard_deviation();
        self.calculate_min();
        self.calculate_max();
        self.calculate_median();

        self.calculate_skewness();
        self.calculate_kurtosis();
        self.calculate_bimodality();
    }

    /*================================================ STATS CALCULATION FUNCTIONS ============================================================= */
    /// Calculate skewness and return it
    /// Offline version
    pub fn calculate_skewness(&mut self) -> f64 {
        let n = self.len as f64;
        let std_dev = self.buffer_variance.sqrt();

        self.skewness = self
            .vec
            .iter()
            .map(|&value| {
                let diff = (value - self.buffer_mean) / std_dev;
                diff.powi(3)
            })
            .sum::<f64>()
            / n;
        self.stats[2] = n * std_dev.powi(3) * self.skewness;

        self.skewness
    }
    ///Online version
    pub fn calculate_skewness_online(&mut self) -> f64 {
        let n = self.len as f64;
        self.skewness = ((n.sqrt()) * self.stats[2]) / (self.stats[1].powf(1.5));

        self.skewness
    }

    /// Calculate kurtosis and return it
    /// Offline version:
    pub fn calculate_kurtosis(&mut self) -> f64 {
        let n = self.len as f64;
        let std_dev4 = self.buffer_variance * self.buffer_variance;

        self.stats[3] = self
            .vec
            .iter()
            .map(|&value| {
                let diff = value - self.buffer_mean;
                diff.powi(4)
            })
            .sum::<f64>();
        self.kurtosis = self.stats[3] / (n * std_dev4);

        self.kurtosis
    }
    /// Online kurtosis
    pub fn calculate_kurtosis_online(&mut self) -> f64 {
        let n = self.len as f64;
        self.kurtosis = (n * self.stats[3]) / self.stats[1] * self.stats[1];

        self.kurtosis
    }

    /// Calculate bimodality coefficient and return it
    pub fn calculate_bimodality(&mut self) -> f64 {
        self.bimodality = self.skewness * self.skewness / self.kurtosis;

        self.bimodality
    }

    /// Calculate mean and return it
    /// Offline version
    pub fn calculate_mean(&mut self) -> f64 {
        self.buffer_mean = self.vec.iter().sum::<f64>() / self.len as f64;
        //self.buffer_mean = (self.buffer_mean * 100.0).round() / 100.0;
        self.stats[0] = self.buffer_mean;

        self.buffer_mean
    }
    ///Online version
    pub fn calculate_mean_online(&mut self) -> f64 {
        self.buffer_mean = self.stats[0];

        self.buffer_mean
    }

    /// Calculate variance and return it
    /// Offline version
    pub fn calculate_variance(&mut self) -> f64 {
        self.stats[1] = self
            .vec
            .iter()
            .map(|&value| {
                let diff = self.buffer_mean - value;
                diff * diff
            })
            .sum::<f64>();
        self.buffer_variance = self.stats[1] / self.len as f64;

        self.buffer_variance
    }
    ///Online version
    pub fn calculate_variance_online(&mut self) -> f64 {
        let n = self.len as f64;
        self.buffer_variance = self.stats[1] / (n - 1.0);

        self.buffer_variance
    }

    /// Calculate standard deviation from population variance
    pub fn calculate_standard_deviation(&mut self) -> f64 {
        self.buffer_variance.sqrt()
    }

    /// Calculate minimum(s) of values and return min
    pub fn calculate_min(&mut self) -> f64 {
        self.min = self.vec[0];
        for k in &self.vec[1..] {
            if k < &self.min {
                self.min_ = self.min;
                self.min = *k;
            } else if k < &self.min_ {
                self.min_ = *k;
            }
        }
        self.min
    }

    /// Calculate maximum(s) of values and return max
    pub fn calculate_max(&mut self) -> f64 {
        self.max = self.vec[0];
        for k in &self.vec[1..] {
            if k > &self.max {
                self.max_ = self.max;
                self.max = *k;
            } else if k > &self.max_ {
                self.max_ = *k;
            }
        }
        self.max
    }

    /// We calculate the median using the quickselect algorithm, which avoids a full sort by sorting
    /// only partitions of the data set known to possibly contain the median. This uses cmp and
    /// Ordering to succinctly decide the next `median_partition` to examine, and `split_at` to choose an
    /// arbitrary pivot for the next `median_partition` at each step
    pub fn calculate_median(&mut self) -> f64 {
        self.median = match self.len {
            even if even % 2 == 0 => {
                let fst_med = median_select(&self.vec, (even / 2) - 1);
                let snd_med = median_select(&self.vec, even / 2);

                match (fst_med, snd_med) {
                    (Some(fst), Some(snd)) => Some((fst + snd) / 2.0),
                    _ => None,
                }
            }
            odd => median_select(&self.vec, odd / 2),
        }
        .unwrap();

        self.median
    }

    /// rough estimate of median, use `calculate_median` for exact median
    pub fn calculate_approx_median(&mut self) {
        self.median = (self.max + self.min + 2.0 * self.buffer_mean) / 4.0;
    }

    // Need a better way to find mode
    /*
    fn calculate_mode(&mut self){
        let frequencies = self.vec.iter().fold(HashMap::new(), |mut freqs, value| {
            *freqs.entry(value).or_insert(0) += 1;
            freqs
        });

        let mode = frequencies
            .into_iter()
            .max_by_key(|&(_, count)| count)
            .map(|(value, _)| *value).unwrap();

        self.mode = mode;
    }
    */
}

/*========================================== Helper Functions for Median calculation ================================================================ */
/// Helper for median
fn median_partition(data: &Vec<f64>) -> Option<(Vec<f64>, f64, Vec<f64>)> {
    match data.len() {
        0 => None,
        _ => {
            let (pivot_slice, tail) = data.split_at(1);
            let pivot = pivot_slice[0];
            let (left, right) = tail.iter().fold((vec![], vec![]), |mut splits, next| {
                {
                    let (ref mut left, ref mut right) = &mut splits;
                    if next < &pivot {
                        left.push(*next);
                    } else {
                        right.push(*next);
                    }
                }
                splits
            });

            Some((left, pivot, right))
        }
    }
}

/// Helper for median
fn median_select(data: &Vec<f64>, k: usize) -> Option<f64> {
    let part = median_partition(data);

    match part {
        None => None,
        Some((left, pivot, right)) => {
            let pivot_idx = left.len();

            match pivot_idx.cmp(&k) {
                Ordering::Equal => Some(pivot),
                Ordering::Greater => median_select(&left, k),
                Ordering::Less => median_select(&right, k - (pivot_idx + 1)),
            }
        }
    }
}

/*============================================ MAIN OPERATIONS : push, append, remove, pop ============================================= */
impl SimpleAccumulator {
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
    pub fn push<T: ToPrimitive>(&mut self, value: T) {
        let y = T::to_f64(&value).unwrap();

        // Running stats, Number of elements seen is incremented irrespective of buffer properties
        // Calculation is online following Knuth's algorithm
        self.total += 1;
        let delta = y - self.mean;
        let delta_n = delta / self.total as f64;
        self.mean += delta_n;
        let term1 = delta * delta_n * (self.total as f64 - 1.0);
        let stats1 = self.variance * (self.total as f64 - 2.0) + term1;
        self.variance = stats1 / (self.total as f64 - 1.0);

        // we just change the already held value and keep on rewriting it
        if self.fixed_capacity {
            if self.len == 0 {
                self.last_write_position = 0;
            } else {
                self.last_write_position = (self.last_write_position + 1) % self.capacity;
            }

            // Using vector push when ring buffer is not full
            if self.len < self.capacity {
                self.vec.push(y);
                //Update number of elements
                self.len += 1;
            }
            // Replacing first element by new element when ring buffer is full
            else {
                self.vec[self.last_write_position] = y;
            }
        }
        // Using vector push in case fixed_capacity flag is 'false'
        else {
            self.vec.push(y);
            //Update number of elements
            self.len += 1;
            self.capacity = self.vec.capacity();
        }

        if self.accumulate {
            // Update stats for the buffer
            if self.fixed_capacity {
                self.calculate_all();
            } else {
                self.update_fields_increase(T::to_f64(&value).unwrap());
            }
        }
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
    pub fn append<T: ToPrimitive>(&mut self, value: &Vec<T>) {
        // let mut value: Vec<f64> = value.iter().map(|x| T::to_f64(&x).unwrap()).collect();
        let mut sum = 0.0;
        let mut old_sq_sum = 0.0;
        let mut old_cube_sum = 0.0;
        let mut old_fourth_power_sum = 0.0;

        let mut temp_values: Vec<f64> = Vec::with_capacity(value.len());

        for t in value {
            let k = T::to_f64(t).unwrap();
            temp_values.push(k);
            // to find mean
            sum += k;
            // to find variance
            old_sq_sum += k * k;
            // to find skewness
            old_cube_sum += k.powi(3);
            //to find kurtosis
            old_fourth_power_sum += k.powi(4);

            // updating min-max values
            if k > self.max {
                self.max_ = self.max;
                self.max = k;
            } else if k > self.max_ {
                self.max_ = k;
            } else if k < self.min {
                self.min_ = self.min;
                self.min = k;
            } else if k < self.min_ {
                self.min_ = k;
            }
        }

        let old_old_mean = self.buffer_mean;
        let old_variance = self.buffer_variance;
        let new_len = (self.len + temp_values.len()) as f64;

        // Computing running stats online
        let new_total = (self.total + temp_values.len()) as f64;
        let ra = self.total as f64 * (self.variance + self.mean * self.mean);
        self.mean = (self.mean * self.total as f64 + sum) / new_total;

        let rb = (-1.0) * (self.mean * self.mean);
        self.variance = (ra + old_sq_sum) / new_total + rb;
        self.total = new_total as usize;

        if self.fixed_capacity {
            // Using vector append() when ring buffer is not full
            if temp_values.len() <= self.capacity - self.len {
                self.len += temp_values.len();
                self.vec.append(&mut temp_values);
            }
            // Deleting at most temp_values.len() number of oldest values and replacing with the
            // new ones obeying FIFO, since the buffer does not have space for vector append()
            else {
                let temp_len = temp_values.len();
                let mut start_fill_index = 0;
                if temp_len > self.capacity {
                    start_fill_index = temp_len - self.capacity;
                }

                // Pushing the values in temp while deleting oldest elements in buffer
                // If temp is really long only the last 'capacity' number of elements
                // will be filling the buffer

                if !self.vec.is_empty() {
                    for i in temp_values.iter().skip(start_fill_index) {
                        self.vec[self.last_write_position] = *i;
                        self.last_write_position = (self.last_write_position + 1) % self.capacity;
                    }
                } else {
                    for i in temp_values.iter().skip(start_fill_index) {
                        self.vec.push(*i);
                    }
                }

                self.len = self.vec.len();
            }
        }
        // Using vector append when fixed_capacity is 'false'
        else {
            self.len += temp_values.len();
            self.vec.append(&mut temp_values);
            self.capacity = self.vec.capacity();
        }

        // Computing buffer stats online when capacity is not fixed
        if self.accumulate {
            if !self.fixed_capacity {
                self.buffer_mean = (self.buffer_mean * self.len as f64 + sum) / new_len;
                let a = self.len as f64 * (self.buffer_variance + old_old_mean * old_old_mean);
                let b = (-1.0) * (self.buffer_mean * self.buffer_mean);
                self.buffer_variance = (a + old_sq_sum) / new_len + b;

                let old_old_cube_sum =
                    self.stats[2] + old_old_mean.powi(3) + 3.0 * old_old_mean * old_variance;
                let new_cube_sum = old_old_cube_sum + old_cube_sum;
                self.skewness = (new_cube_sum
                    - self.buffer_mean.powi(3)
                    - 3.0 * self.buffer_mean * self.buffer_variance)
                    / (new_len * self.buffer_variance.powf(1.5));

                let old_old_fourth_power_sum = self.stats[3]
                    + old_old_mean.powi(4)
                    + 6.0 * old_old_mean.powi(2) * old_variance
                    + 4.0 * self.stats[2] * old_old_mean;
                let new_fourth_power_sum = old_old_fourth_power_sum + old_fourth_power_sum;
                self.kurtosis = new_fourth_power_sum / (new_len * self.buffer_variance.powi(2));
                self.calculate_approx_median();
            } else {
                self.calculate_all();
            }
            self.calculate_bimodality();
            // Updating stats vector
            self.stats[0] = self.buffer_mean;
            self.stats[1] = self.buffer_variance * self.len as f64;
            self.stats[2] = self.skewness * self.buffer_variance.powf(1.5) * self.len as f64;
            self.stats[3] = self.kurtosis * self.buffer_variance.powi(2) * self.len as f64;
        }
    }

    /// This function removes the element from the accumulator at the specified,
    /// valid **index** using vector remove() function. If the index is out of bounds,
    /// returns None. Decrements the number of elements and updates stats online.
    /// Function unavailable for fixed capacity,
    /// returns `None` when `fixed_capacity: true`
    pub fn remove(&mut self, index: usize) -> Option<f64> {
        if self.fixed_capacity {
            None
        }
        // When capacity is not fixed
        else {
            // Check if index to be removed is out of bounds
            if self.len - 1 < index {
                return None;
            }

            let k = self.vec.remove(index);

            if self.accumulate {
                self.update_fields_decrease(k);
                self.min_max_update_when_removed(k);
            }
            // Update number of elements
            self.len -= 1;
            Some(k)
        }
    }

    /// The function removes and returns the first element with the vector pop()
    /// function if the accumulator is non-empty, else returns None.
    /// Function unavailable for fixed capacity,
    /// returns `None` when `fixed_capacity: true`
    ///
    pub fn pop(&mut self) -> Option<f64> {
        if self.fixed_capacity {
            None
        }
        // When capacity is not fixed and accumulator is non-empty
        else if self.len > 0 {
            let k = self.vec.pop().unwrap();

            if self.accumulate {
                self.update_fields_decrease(k);
                self.min_max_update_when_removed(k);
            }
            // Update number of elements
            self.len -= 1;
            Some(k)
        }
        // Nothing to pop
        else {
            None
        }
    }

    /*================================== Helper Fns for push / append / pop / remove============================================================================= */

    /// Function to update fields based on an increase in data points.
    /// When the accumulate flag is set, this function re-calculates min, max and other
    /// statistics after a push to the accumulator.
    fn update_fields_increase(&mut self, value: f64) {
        let n = self.len as f64;
        let delta = value - self.stats[0];
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1.0);

        self.stats[0] += delta_n;
        self.stats[3] += term1 * delta_n2 * (3.0 + n * n - 3.0 * n)
            + 6.0 * delta_n2 * self.stats[1]
            - 4.0 * delta_n * self.stats[2];
        self.stats[2] += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.stats[1];
        self.stats[1] += term1;

        // Calculating stats online from the updated stats vector
        self.calculate_mean_online();
        self.calculate_variance_online();
        self.calculate_skewness_online();
        self.calculate_kurtosis_online();
        self.calculate_bimodality();

        // we can handle these here unlike when we remove elements
        if self.min >= value {
            self.min_ = self.min;
            self.min = value;
        } else {
            self.max_ = self.max;
            self.max = value;
        }
        self.calculate_approx_median();
    }

    /// Function to re-calculate all stats online after a pop or remove operation but
    /// does not re-compute max, min, median which is left to the inline function
    /// `min_max_update_when_removed`.
    ///
    fn update_fields_decrease(&mut self, value: f64) {
        let n = self.len as f64;
        let delta = value - self.buffer_mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n + 1.0);

        self.stats[0] -= delta_n;
        self.stats[3] -= term1 * delta_n2 * (3.0 + n * n - 3.0 * n)
            + 6.0 * delta_n2 * self.stats[1]
            - 4.0 * delta_n * self.stats[2];
        self.stats[2] -= term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.stats[1];
        self.stats[1] -= term1;

        // Calculating stats online from the updated stats vector
        self.calculate_mean_online();
        self.calculate_variance_online();
        self.calculate_skewness_online();
        self.calculate_kurtosis_online();
        self.calculate_bimodality();
    }

    #[inline]
    /// SimpleAccumulator needs to calculate min, max, approx median after the value is removed.
    /// This function re-calculates min, min_ or max, max_ according to the data removed, and
    /// calls `calculate_approx_median()` for median computation.
    fn min_max_update_when_removed(&mut self, value: f64) {
        if self.min == value {
            self.min = self.min_;
            self.min_ = ((self.len as f64 * self.min) + self.buffer_mean) / (self.len as f64 + 1.0);
        }

        if self.max == value {
            self.max = self.max_;
            self.max_ = ((self.len as f64 * self.max) + self.buffer_mean) / (self.len as f64 + 1.0);
        }
        self.calculate_approx_median();
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::SimpleAccumulator;

    #[test]
    fn new_no_fixed_capacity() {
        let k = [1, 2, 3, 4];

        let x = SimpleAccumulator::new(&k, true);
        let y = SimpleAccumulator::new(&[101.5, 33.25, 56.75, 61.5, 10.0], true);
        // Integer arithmetic
        assert_eq!(
            x,
            SimpleAccumulator {
                vec: Vec::from([1.0, 2.0, 3.0, 4.0,]),
                stats: Vec::from([2.5, 5.0, 0.0, 10.25]),
                mean: 2.5,
                variance: 1.25,
                total: 4,
                buffer_mean: 2.5,
                buffer_variance: 1.25,
                min: 1.0,
                min_: 2.0,
                max: 4.0,
                max_: 3.0,
                median: 2.5,
                len: 4,
                capacity: 4,
                fixed_capacity: false,
                last_write_position: 0,
                accumulate: true,
                skewness: 0.0,
                kurtosis: 1.64,
                bimodality: 0.0,
            }
        );

        // Floating point arithmetic
        assert_float_eq!(y.buffer_mean, 52.6, abs <= 0.01);
        assert_float_eq!(y.buffer_variance, 935.365, abs <= 0.01);
        assert_float_eq!(y.median, 56.75, abs <= 0.01);
        assert_float_eq!(y.skewness, 0.23, abs <= 0.01);
        assert_float_eq!(y.kurtosis, 2.09, abs <= 0.01);
        assert_float_eq!(y.bimodality, 0.03, abs <= 0.01);
    }
}

#[cfg(examples)]
mod examples {
    use super::SimpleAccumulator;
    use plotly::common::Mode;
    use plotly::{Plot, Scatter};
    use rand::Rng;

    fn online_offline_means_converge() {
        let mut acc = SimpleAccumulator::new::<f64>(&[], true);
        let mut error_mean: Vec<f64> = Vec::new();
        let mut len_per_error_mean: Vec<f64> = Vec::new();
        let base: i32 = 10;
        let multiplier = base.pow(5) as f64;

        println!("Waiting to plot the error data...");
        for _i in 0..1000 {
            for _j in 0..1000 {
                let data = rand::thread_rng().gen::<f64>();
                acc.push(data);
            }
            let mean = acc.buffer_mean;
            let offline_mean = acc.calculate_mean();
            let error_diff = (offline_mean - mean) / acc.len as f64;
            error_mean.push(error_diff * multiplier);
            len_per_error_mean.push(acc.len as f64);
        }

        // Plot the error data
        let trace = Scatter::new(len_per_error_mean, error_mean)
            .name("trace")
            .mode(Mode::LinesMarkers);
        let mut plot = Plot::new();
        plot.add_trace(trace);

        plot.show();
        println!("{}", plot.to_inline_html(Some("error_mean_scatter_plot")));
    }
}
