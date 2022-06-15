//! Store and update stats related to our data array without iterating again and again

use num::ToPrimitive;
use std::cmp::Ordering;
// use std::collections::HashMap;

/// Our main data struct
#[derive(Clone, Debug)]
pub struct CustomVector {
    pub vec: Vec<f64>,
    pub mean: f64,
    /// Population variance uses `N` NOT `N-1`
    pub population_variance: f64,
    pub standard_deviation: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    // mode: f64,
    pub len: usize,
}

impl CustomVector {
    /// Can be made of any type `&[T]` but will be converted to `Vec<f64>`, panics on values that
    /// cannot be converted
    pub fn new<T: ToPrimitive>(slice: &[T]) -> Self {
        let vec: Vec<f64> = slice
            .clone()
            .iter()
            .map(|x| T::to_f64(x).unwrap())
            .collect();

        let mut k = CustomVector {
            vec,
            mean: 0.0,
            population_variance: 0.0,
            standard_deviation: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            // mode: 0.0,
            len: 0,
        };

        if !k.vec.is_empty() {
            k.len = k.vec.len();
            k.calculate_mean();
            k.calculate_population_variance();
            k.calculate_standard_deviation();
            k.calculate_min();
            k.calculate_max();
            k.calculate_median();
            // k.calculate_mode();
        }
        k
    }

    fn calculate_mean(&mut self) {
        self.mean = self.vec.iter().sum::<f64>() / self.len as f64;
    }

    fn calculate_population_variance(&mut self) {
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

    fn calculate_standard_deviation(&mut self) {
        self.standard_deviation = self.population_variance.sqrt();
    }

    fn calculate_min(&mut self) {
        self.min = self.vec.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    }

    fn calculate_max(&mut self) {
        self.max = self.vec.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    }

    /// We calculate the median using the quickselect algorithm, which avoids a full sort by sorting
    /// only partitions of the data set known to possibly contain the median. This uses cmp and
    /// Ordering to succinctly decide the next median_partition to examine, and split_at to choose an
    /// arbitrary pivot for the next median_partition at each step
    fn calculate_median(&mut self) {
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

impl CustomVector {
    /// Same as `push` in `Vec`
    pub fn push<T: ToPrimitive>(&mut self, value: T) {
        self.update_fields_increase(T::to_f64(&value).unwrap());
        self.vec.push(T::to_f64(&value).unwrap());
        // TODO: find a better way for this
        self.calculate_median();
    }

    /// Same as `remove` in `Vec`
    pub fn remove(&mut self, index: usize) -> f64 {
        self.update_fields_decrease(self.vec[index]);
        let k = self.vec.remove(index);
        if self.min == k {
            self.calculate_min();
        }

        if self.max == k {
            self.calculate_max();
        }
        // TODO: find a better way for this
        self.calculate_median();
        k
    }

    /// Same as `pop` in `Vec`
    pub fn pop(&mut self) -> Option<f64> {
        if self.len == 0 {
            None
        } else {
            Some(self.remove(self.len - 1))
        }
    }

    /// Update fields based on an increase, no iteration
    fn update_fields_increase(&mut self, value: f64) {
        // mean
        self.mean = ((self.mean * self.len as f64) + value) / (self.len as f64 + 1.0);
        // population variance
        let iv = value - self.mean;
        self.population_variance =
            ((self.population_variance * self.len as f64) + iv * iv) / (self.len as f64 + 1.0);

        self.standard_deviation = self.population_variance.sqrt();

        if self.min >= value {
            self.min = value;
        } else {
            self.max = value;
        }
        self.len += 1;
    }

    /// Update fields based on a decrease, no iteration
    fn update_fields_decrease(&mut self, value: f64) {
        let iv = value - self.mean;
        // mean
        self.mean = ((self.mean * self.len as f64) - value) / (self.len as f64 - 1.0);
        // population variance
        self.population_variance =
            ((self.population_variance * self.len as f64) - iv * iv) / (self.len as f64 - 1.0);

        self.standard_deviation = self.population_variance.sqrt();

        self.len -= 1;
    }
}
