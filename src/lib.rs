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
#![allow(clippy::clone_double_ref)]
pub mod check;
use num::ToPrimitive;
use std::cmp::Ordering;
// use std::collections::HashMap;

/// Our main data struct
#[derive(Clone, Debug, PartialEq)]
pub struct SimpleAccumulator {
    /// Vec to store the data
    pub vec: Vec<f64>,
    /// Vec to privately store four moments, mean, 2nd moment, 3rd and 4th moment
    stats: Vec<f64>,
    /// Average/mean of the data
    pub mean: f64,
    /// Population variance, uses `N` not `N-1`
    pub population_variance: f64,
    /*
    /// (Standard deviation)^2 = population_variance
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
    /// Can be made of any type `&[T]` but will be converted to `Vec<f64>`, panics on values that
    /// cannot be converted
    pub fn new<T: ToPrimitive>(slice: &[T], flag: bool) -> Self {
        let vec: Vec<f64> = slice
            .clone()
            .iter()
            .map(|x| T::to_f64(x).expect("Not a number"))
            .collect();

        let stats: Vec<f64> = vec![0.0;4];

        let mut k = SimpleAccumulator {
            vec,
            stats,
            mean: 0.0,
            population_variance: 0.0,
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
            bimodality:0.0,
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

        let mut vec: Vec<f64> = slice
            .clone()
            .iter()
            .map(|x| T::to_f64(x).unwrap())
            .collect();

        let stats: Vec<f64> = vec![0.0;4];

        vec.reserve_exact(capacity);

        let mut k = SimpleAccumulator {
            vec,
            stats,
            mean: 0.0,
            population_variance: 0.0,
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
            bimodality:0.0,
        };

        if !k.vec.is_empty() && flag {
            k.last_write_position = k.vec.len() - 1;
            k.len = k.vec.len();
            k.calculate_all();
            // k.calculate_mode();
        }
        k
    }

    pub fn calculate_all(&mut self) {
        self.calculate_mean();
        self.calculate_population_variance();
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
        let std_dev = self.population_variance.sqrt();
        
        self.skewness = self
        .vec
        .iter()
        .map(|&value| {
            let diff = (value - self.mean)/std_dev;
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
        self.skewness = ((n.sqrt())*self.stats[2])/(self.stats[1].powf(1.5));
        self.skewness
    }

    /// Calculate kurtosis and return it
    /// Offline version:
    pub fn calculate_kurtosis(&mut self) -> f64 {
        let n = self.len as f64;
        let std_dev4 = self.population_variance * self.population_variance;
        
        self.stats[3] = self
        .vec
        .iter()
        .map(|&value| {
            let diff = value - self.mean;
            diff.powi(4)
        })
        .sum::<f64>();
        self.kurtosis = self.stats[3]/ (n*std_dev4);

        self.kurtosis
    }
    /// Online kurtosis
    pub fn calculate_kurtosis_online(&mut self) -> f64 {
        let n = self.len as f64;
        self.kurtosis = (n*self.stats[3])/self.stats[1]*self.stats[1];
        self.kurtosis
    }

    /// Calculate bimodality and return it
    pub fn calculate_bimodality(&mut self) -> f64 {
        self.bimodality = self.skewness * self.skewness/self.kurtosis;
        self.bimodality
    }

    /// Calculate mean and return it
    /// Offline version
    pub fn calculate_mean(&mut self) -> f64 {
        self.mean = self.vec.iter().sum::<f64>() / self.len as f64;
        self.stats[0] = self.mean;
        self.mean
    }
    ///Online version
    pub fn calculate_mean_online(&mut self) -> f64 {
        self.mean = self.stats[0];
        self.mean
    }

    /// Calculate population variance and return it
    /// Offline version
    pub fn calculate_population_variance(&mut self) -> f64 {
        self.stats[1] = self
            .vec
            .iter()
            .map(|&value| {
                let diff = self.mean - value;
                diff * diff
            })
            .sum::<f64>();
        self.population_variance = self.stats[1]/ self.len as f64;

        self.population_variance
    }
    ///Online version
    pub fn calculate_population_variance_online(&mut self) -> f64 {
        let n = self.len as f64;
        self.population_variance = self.stats[1]/(n - 1.0);
        self.population_variance
    }

    /// Calculate standard deviation from population variance
    pub fn calculate_standard_deviation(&mut self) -> f64 {
        self.population_variance.sqrt()
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
        self.median = (self.max + self.min + 2.0 * self.mean) / 4.0;
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
    /// Similar to `push` in vector `Vec`. When `fixed_capacity` flag is 'true' and ring buffer is full
    /// rewrites the oldest element with the latest, following FIFO order. 
    pub fn push<T: ToPrimitive>(&mut self, value: T) {
        let y = T::to_f64(&value).unwrap();
        let n = self.len as f64;

        // we just change the already held value and keep on rewriting it
        if self.fixed_capacity {
            if self.len == 0 {
                self.last_write_position = 0;
            }
            else {
                self.last_write_position = (self.last_write_position + 1) % self.capacity;
            }

            // Using vector push when ring buffer is not full
            if self.len < self.capacity {
                self.vec.push(y);
                //Update number of elements
                self.len += 1;
                if self.accumulate {
                    // Update stats vector
                    self.update_fields_increase(y);
                }
            } 
            // Replacing first element by new element when ring buffer is full
            else {
                let k = self.vec[self.last_write_position];
                self.vec[self.last_write_position] = y;
                //Old mean
                //let old_mean = self.mean;
                
                //-------------------New: Knuth's method from John D. Cook's website ----------------------------------------------------
                let delta = y - k;
                let delta_n = delta/n;
                let delta_n2 = delta_n * delta_n ;
                let term1 = delta * delta_n * (n - 1.0);
                //First moment, mean
                self.stats[0] += delta_n;
                //Fourth moment difference, needed to compute kurtosis
                self.stats[3] += term1*delta_n2*(3.0 + n*n - 3.0*n) + 6.0*delta_n2*self.stats[1] - 4.0*delta_n*self.stats[2];
                //Third moment difference, needed to compute skewness
                self.stats[2] += term1*delta_n*(n - 2.0) - 3.0*delta_n*self.stats[1];
                //Second moment difference, needed to compute variance, standard deviation
                self.stats[1] += term1;
                
                //-------------------------------------------------------------------------------------------------------------------------
                // Calculating stats online from the updated stats vector
                self.calculate_mean_online();
                self.calculate_population_variance_online();
                self.calculate_skewness_online();
                self.calculate_kurtosis_online();
                self.calculate_bimodality();


                //Old method:
                //self.mean = (self.len as f64 * self.mean - k + y) / self.len as f64;

                //Old method:
                // variance
                /*
                self.population_variance = (((self.len as f64)
                    * (self.population_variance
                        + (self.mean - old_mean) * (self.mean - old_mean)))
                    + ((y - k) * (y + k - 2.0 * self.mean)))
                    / self.len as f64;
                */
                if y > self.max {
                    self.max_ = self.max;
                    self.max = y;
                } else if y > self.max_ {
                    self.max_ = y;
                } else if y < self.min {
                    self.min_ = self.min;
                    self.min = y;
                } else if y < self.min_ {
                    self.min_ = y;
                }
                self.calculate_approx_median();
            }
        } 
        // Using vector push in case fixed_capacity flag is 'false'
        else {
            self.vec.push(y);
            //Update number of elements
            self.len += 1;
            self.capacity = self.vec.capacity();

            if self.accumulate {
                // Update stats vector
                self.update_fields_increase(T::to_f64(&value).unwrap());
            }
        }
    }

    /// Similar behaviour to `append` in `Vec`, rewrites in FIFO order if `fixed_capacity` is 'true'
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
        
        let old_old_mean = self.mean;
        let old_variance = self.population_variance;
        let mut new_cube_sum = 0.0;
        let mut new_fourth_power_sum = 0.0;
        let new_len = (self.len + temp_values.len()) as f64;

        if self.accumulate {
            self.mean = (self.mean * self.len as f64 + sum) / new_len;
            let a = self.len as f64 * (self.population_variance + old_old_mean * old_old_mean);
            let b = (-1.0) * (self.mean * self.mean);
            self.population_variance = (a + old_sq_sum) / new_len + b;

            let old_old_cube_sum = self.stats[2] + old_old_mean.powi(3) + 3.0*old_old_mean*old_variance;
            new_cube_sum = old_old_cube_sum + old_cube_sum;
            self.skewness = (new_cube_sum - self.mean.powi(3) - 3.0*self.mean*self.population_variance)/(new_len*self.population_variance.powf(1.5));

            let old_old_fourth_power_sum = self.stats[3] + old_old_mean.powi(4) + 6.0 * old_old_mean.powi(2) *old_variance + 4.0*self.stats[2]*old_old_mean;
            new_fourth_power_sum = old_old_fourth_power_sum + old_fourth_power_sum;
            self.kurtosis = new_fourth_power_sum/(new_len * self.population_variance.powi(2));
        }
        

        if self.fixed_capacity {
            // Using vector append() when ring buffer is not full
            if temp_values.len() <= self.capacity - self.len {
                self.len += temp_values.len();
                self.vec.append(&mut temp_values);
                //self.calculate_approx_median();
            }
            // Deleting at most temp_values.len() number of oldest values and replacing with the
            // new ones obeying FIFO, since the buffer does not have space for vector append() 
            else {
                let mut start_fill_index = 0;
                if let Some(i) = temp_values.len().checked_sub(self.len) {
                    start_fill_index = i;
                }
                let mut sum = 0.0;
                let mut sq_sum = 0.0;
                let mut cube_sum = 0.0;
                let mut fourth_power_sum = 0.0;

                for i in temp_values.iter().skip(start_fill_index) {
                    self.last_write_position = (self.last_write_position + 1) % self.capacity;
                    let num = self.vec[self.last_write_position];
                    sum += num;
                    sq_sum += num * num;
                    cube_sum += num.powi(3);
                    fourth_power_sum += num.powi(4);
                    self.vec[self.last_write_position] = *i;
                }
                
                self.mean = ((self.mean * new_len) - sum) / self.len as f64;

                self.population_variance = (self.len as f64
                    * (old_variance + old_old_mean * old_old_mean)
                    + (-1.0) * (sq_sum + self.len as f64 * self.mean * self.mean)
                    + old_sq_sum)
                    / self.len as f64;   

                self.skewness = (new_cube_sum - cube_sum - self.mean.powi(3) - 3.0*self.mean*self.population_variance)
                /((self.len as f64)*self.population_variance.powf(1.5));     

                self.kurtosis =  (new_fourth_power_sum - fourth_power_sum)/((self.len as f64) * self.population_variance.powi(2));
            }
        } 
        // Using vector append when fixed_capacity is 'false'
        else {
            self.len += temp_values.len();
            self.vec.append(&mut temp_values);
            self.capacity = self.vec.capacity();
            //self.calculate_approx_median();
        }

        self.stats[0] = self.mean;
        self.stats[1] = self.population_variance * self.len as f64;
        self.stats[2] = self.skewness * self.population_variance.powf(1.5) * self.len as f64;
        self.stats[3] = self.kurtosis * self.population_variance.powi(2) * self.len as f64;

        if self.accumulate {
            self.calculate_approx_median();
            self.calculate_bimodality();
        }
    }

    /// Same as `remove` in `Vec` but returns `None` if index is out of bounds
    /// Function unavailable for fixed capacity.
    /// Always returns `None` when `fixed_capacity: true`
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

    /// Same as `pop` in `Vec`
    /// Function unavailable for fixed capacity.
    /// Always returns `None` when `fixed_capacity: true`
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

    /// Update fields based on an increase, no iteration
    fn update_fields_increase(&mut self, value: f64) {
        /*let old_mean = self.mean;
        // mean
        self.mean = ((self.mean * (self.len - 1) as f64) + value) / (self.len as f64);
        // population variance
        let iv = self.mean - value;
        let ib = (self.len as f64 - 1.0)
            * (self.population_variance - (2.0 * self.mean * old_mean)
                + (old_mean * old_mean)
                + (self.mean * self.mean));
        self.population_variance = (ib + (iv * iv)) / (self.len as f64);*/

        let n = self.len as f64;
        let delta = self.vec[self.len-1] - self.stats[0];
        let delta_n = delta/n ;
        let delta_n2 = delta_n * delta_n ;
        let term1 = delta * delta_n * (n - 1.0);
    
        self.stats[0] += delta_n;
        self.stats[3] += term1*delta_n2*(3.0 + n*n - 3.0*n) + 6.0*delta_n2*self.stats[1] - 4.0*delta_n*self.stats[2];
        self.stats[2] += term1*delta_n*(n - 2.0) - 3.0*delta_n*self.stats[1];
        self.stats[1] += term1;

        // Calculating stats online from the updated stats vector
        self.calculate_mean_online();
        self.calculate_population_variance_online();
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

    /// Update fields based on a decrease, no iteration
    fn update_fields_decrease(&mut self, value: f64) {
        /*
        let old_mean = self.mean;
        // mean
        self.mean = ((self.mean * self.len as f64) - value) / (self.len as f64 - 1.0);
        // population variance
        let iv = self.mean - value;
        let ib = (self.len as f64)
            * (self.population_variance - (2.0 * self.mean * old_mean)
                + (old_mean * old_mean)
                + (self.mean * self.mean));
        self.population_variance = (ib - (iv * iv)) / (self.len as f64 - 1.0);
         */

        let n = self.len as f64;
        let delta = value - self.mean;
        let delta_n = delta/n ;
        let delta_n2 = delta_n * delta_n ;
        let term1 = delta * delta_n * (n + 1.0);

        self.stats[0] -= delta_n;
        self.stats[3] -= term1*delta_n2*(3.0 + n*n - 3.0*n) + 6.0*delta_n2*self.stats[1] - 4.0*delta_n*self.stats[2];
        self.stats[2] -= term1*delta_n*(n - 2.0) - 3.0*delta_n*self.stats[1];
        self.stats[1] -= term1;

         // Calculating stats online from the updated stats vector
         self.calculate_mean_online();
         self.calculate_population_variance_online();
         self.calculate_skewness_online();
         self.calculate_kurtosis_online();
         self.calculate_bimodality();
    }

    #[inline]
    /// Need to calculate min-max after the value is removed, approx min-max
    fn min_max_update_when_removed(&mut self, value: f64) {
        if self.min == value {
            self.min = self.min_;
            self.min_ = ((self.len as f64 * self.min) + self.mean) / (self.len as f64 + 1.0);
        }

        if self.max == value {
            self.max = self.max_;
            self.max_ = ((self.len as f64 * self.max) + self.mean) / (self.len as f64 + 1.0);
        }
        self.calculate_approx_median();
    }
}

#[cfg(test)]
mod tests {
    use super::SimpleAccumulator;

    #[test]
    fn new_no_fixed_capacity() {
        let k = [1, 2, 3, 4];

        let x = SimpleAccumulator::new(&k, true);
        let y = SimpleAccumulator::new(&[101.5, 33.25, 56.75, 61.5, 10.0], true);
        
        assert_eq!(
            x,
            SimpleAccumulator {
                vec: Vec::from([1.0, 2.0, 3.0, 4.0,]),
                stats: Vec::from([2.5, 5.0, 0.0, 10.25]),
                mean: 2.5,
                population_variance: 1.25,
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
                skewness:0.0,
                kurtosis:1.64,
                bimodality: 0.0,
            }
        );

        assert_eq!(
            y,
            SimpleAccumulator {
                vec: Vec::from([101.5, 33.25, 56.75, 61.5, 10.0,]),
                stats: Vec::from([52.6, 4676.825000000001, 33152.759999999966,  9158002.1688125]),
                mean: 52.6,
                population_variance: 935.3650000000001,
                min: 10.0,
                min_: 33.25,
                max: 101.5,
                max_: 61.5,
                median: 56.75,
                len: 5,
                capacity: 5,
                fixed_capacity: false,
                last_write_position: 0,
                accumulate: true,
                skewness:0.23178109621597587,
                kurtosis:2.0934785107967406,
                bimodality: 0.025661823747421056,
            }
        );
    }
}
