//! Store and update stats related to our data array without iterating again and again
//!
//! ```rust
//!     let k = [1, 2, 3, 4];
//!
//!     let mut x = SimpleAccumulator::new(&k, true);
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
//! }
//! ```
//!
//! Set field `accumulate` to `false` to not update the value, you will need to run `calculate_all` to
//! get the updated field values
#![allow(clippy::clone_double_ref)]
use num::ToPrimitive;
use std::cmp::Ordering;
// use std::collections::HashMap;

/// Our main data struct
#[derive(Clone, Debug)]
pub struct SimpleAccumulator {
    pub vec: Vec<f64>,
    pub mean: f64,
    /// Population variance uses `N` NOT `N-1`
    pub population_variance: f64,
    pub standard_deviation: f64,
    pub min: f64,
    pub max: f64,
    /// We use a rough estimate when using `accumulate=true`
    pub median: f64,
    // mode: f64,
    pub len: usize,
    capacity: usize,
    fixed_capacity: bool,
    current_write_position: usize,
    /// Flag to set whether the fields update or not after a change
    pub accumulate: bool,
}

impl SimpleAccumulator {
    /// Can be made of any type `&[T]` but will be converted to `Vec<f64>`, panics on values that
    /// cannot be converted
    pub fn new<T: ToPrimitive>(slice: &[T], flag: bool) -> Self {
        let vec: Vec<f64> = slice
            .clone()
            .iter()
            .map(|x| T::to_f64(x).expect("Not a number"))
            .collect();

        let mut k = SimpleAccumulator {
            vec,
            mean: 0.0,
            population_variance: 0.0,
            standard_deviation: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            // mode: 0.0,
            len: 0,
            capacity: 0,
            fixed_capacity: false,
            current_write_position: 0,
            accumulate: flag,
        };

        if !k.vec.is_empty() && flag {
            k.len = k.vec.len();
            k.capacity = k.vec.capacity();
            k.calculate_all();
            // k.calculate_mode();
        }
        k
    }

    /// Can be made of any type `&[T]` but will be converted to `Vec<f64>`, panics on values that
    /// cannot be converted.
    ///
    /// Returns `err` if the provided `slice` has greater number of elements than provided `capacity`
    pub fn with_fixed_capacity<T: ToPrimitive>(slice: &[T], capacity: usize, flag: bool) -> Self {
        if slice.len() > capacity {
            panic!("Capacity less than length of given slice");
        }

        let mut vec: Vec<f64> = Vec::with_capacity(capacity);

        vec = slice
            .clone()
            .iter()
            .map(|x| T::to_f64(x).unwrap())
            .collect();

        let mut k = SimpleAccumulator {
            vec,
            mean: 0.0,
            population_variance: 0.0,
            standard_deviation: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            // mode: 0.0,
            len: 0,
            capacity,
            fixed_capacity: true,
            current_write_position: 0,
            accumulate: flag,
        };

        if !k.vec.is_empty() && flag {
            k.current_write_position = k.vec.len() - 1;
            k.capacity = k.vec.capacity();
            k.len = k.vec.len();
            k.calculate_all();
            // k.calculate_mode();
        }
        k
    }

    pub fn calculate_all(&mut self) {
        self.calculate_mean();
        self.calculate_population_variance();
        self.calculate_standard_deviation();
        self.calculate_min();
        self.calculate_max();
        self.calculate_median();
    }

    /// Calculate mean
    pub fn calculate_mean(&mut self) {
        self.mean = self.vec.iter().sum::<f64>() / self.len as f64;
    }

    /// Calculate population variance
    pub fn calculate_population_variance(&mut self) {
        self.population_variance = self
            .vec
            .iter()
            .map(|&value| {
                let diff = self.mean - value;
                diff * diff
            })
            .sum::<f64>()
            / self.len as f64;
    }

    /// Calculate standard deviation from population variance
    pub fn calculate_standard_deviation(&mut self) {
        self.standard_deviation = self.population_variance.sqrt();
    }

    /// Calculate minimum of values
    pub fn calculate_min(&mut self) {
        self.min = self.vec.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    }

    /// Calculate maximum of values
    pub fn calculate_max(&mut self) {
        self.max = self.vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    }

    /// We calculate the median using the quickselect algorithm, which avoids a full sort by sorting
    /// only partitions of the data set known to possibly contain the median. This uses cmp and
    /// Ordering to succinctly decide the next median_partition to examine, and split_at to choose an
    /// arbitrary pivot for the next median_partition at each step
    pub fn calculate_median(&mut self) {
        self.median = match self.len {
            even if even % 2 == 0 => {
                let fst_med = median_select(&self.vec, (even / 2) - 1);
                let snd_med = median_select(&self.vec, even / 2);

                match (fst_med, snd_med) {
                    (Some(fst), Some(snd)) => Some((fst + snd) as f64 / 2.0),
                    _ => None,
                }
            }
            odd => median_select(&self.vec, odd / 2).map(|x| x as f64),
        }
        .unwrap();
    }

    /// rough estimate of median, use `calaculate_median` for exact median
    pub fn calculate_approx_median(&mut self) {
        self.median = (self.max + self.min + 2.0 * self.mean) / 4.0;
    }

    #[inline]
    fn fields_update_when_removed(&mut self, value: f64) {
        if self.min == value {
            self.calculate_min();
        }

        if self.max == value {
            self.calculate_max();
        }
        self.calculate_approx_median();
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

impl SimpleAccumulator {
    /// Same as `push` in `Vec`
    pub fn push<T: ToPrimitive>(&mut self, value: T) {
        // we just change the already held value and keep on rewriting it
        if self.fixed_capacity {
            self.current_write_position = (self.current_write_position + 1) % self.capacity;
            if self.len < self.capacity {
                self.vec.push(T::to_f64(&value).unwrap());
                self.len += 1;
            } else {
                self.vec[self.current_write_position] = T::to_f64(&value).unwrap();
            }
        // normal push
        } else {
            self.vec.push(T::to_f64(&value).unwrap());
            self.len += 1;
            self.capacity = self.vec.capacity();
        }

        if self.accumulate {
            self.update_fields_increase(T::to_f64(&value).unwrap());
        }
    }

    /// Same as `remove` in `Vec` but returns `None` if index is out of bounds
    pub fn remove(&mut self, index: usize) -> Option<f64> {
        if self.len - 1 < index {
            return None;
        }
        let k = if self.fixed_capacity {
            // removes the element at the given index and replaces it with last
            self.vec.swap_remove(index)
        } else {
            self.vec.remove(index)
        };
        if self.accumulate {
            self.update_fields_decrease(k);
            self.fields_update_when_removed(k);
        }
        self.len -= 1;
        Some(k)
    }

    /// Same as `pop` in `Vec`
    pub fn pop(&mut self) -> Option<f64> {
        if self.len > 0 {
            let k = self.vec.pop().unwrap();

            if self.accumulate {
                self.update_fields_decrease(k);
                self.fields_update_when_removed(k);
            }
            self.len -= 1;
            Some(k)
        } else {
            None
        }
    }

    /// Update fields based on an increase, no iteration
    fn update_fields_increase(&mut self, value: f64) {
        // mean
        self.mean = ((self.mean * (self.len - 1) as f64) + value) / (self.len as f64);
        // population variance
        let iv = value - self.mean;
        self.population_variance =
            ((self.population_variance * (self.len - 1) as f64) + iv * iv) / (self.len as f64);

        // standard deviation
        self.standard_deviation = self.population_variance.sqrt();

        // we can handle these here unlike when we remove elements
        if self.min >= value {
            self.min = value;
        } else {
            self.max = value;
        }
        self.calculate_approx_median();
    }

    /// Update fields based on a decrease, no iteration
    fn update_fields_decrease(&mut self, value: f64) {
        let iv = value - self.mean;
        // mean
        self.mean = ((self.mean * self.len as f64) - value) / (self.len as f64 - 1.0);
        // population variance
        self.population_variance =
            ((self.population_variance * self.len as f64) - iv * iv) / (self.len as f64 - 1.0);
        // standard deviation
        self.standard_deviation = self.population_variance.sqrt();
    }
}
