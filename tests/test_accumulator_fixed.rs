//! Tests

use std::collections::HashSet;

use float_eq::assert_float_eq;
use rand::distributions::Uniform;
use rand::Rng;

use simple_accumulator::SimpleAccumulator;

#[test]
fn test_sanity_push_in_fixed_capacity() {
    const CAPACITY: usize = 3;
    println!("Creating...");
    let mut acc = SimpleAccumulator::with_fixed_capacity(&[], CAPACITY);
    println!("{acc:?}");

    let data = vec![0.0, 1.1, 2.2, 3.3, 4.4];
    for &v in &data {
        acc.push(v);
    }
    println!("{acc:?}");
    assert_eq!(acc.len(), CAPACITY);
    assert_eq!(acc.data, vec![2.2, 3.3, 4.4]);

    acc.push(5.5);
    assert_eq!(acc.len(), CAPACITY);
    assert_eq!(acc.data, vec![3.3, 4.4, 5.5]);

    acc.push(6.6);
    assert_eq!(acc.len(), CAPACITY);
    assert_eq!(acc.data, vec![4.4, 5.5, 6.6]);
}

#[test]
fn test_only_n_recent_values() {
    // We test that values in the fixed capacity Accumulator must be same as the CAPACITY most
    // recent values pushed to it.
    let mut rng = rand::thread_rng();

    // fill an array of size 1000 with random numbers.
    let data: Vec<i32> = (&mut rng)
        .sample_iter(Uniform::new(0, 100))
        .take(1000)
        .collect();

    // Create a SimpleAccumulator for size 10
    const CAPACITY: usize = 10;
    let mut acc = SimpleAccumulator::with_fixed_capacity(&[], CAPACITY);

    // and push values into it.
    for &v in data.iter() {
        acc.push(v as f64);
    }

    assert_eq!(acc.len(), CAPACITY);

    // The values in the accumulator should be the same as lest recent (CAPACITY )
    // values pushed to it. They may not be in the same order.
    let a: HashSet<i32> = acc.data.iter().map(|&x| x as i32).collect();
    let b: HashSet<i32> = data.into_iter().rev().take(CAPACITY).collect();
    println!("{a:?}\n{b:?}");
    assert!(a.len() <= CAPACITY); // duplicates
    assert_eq!(a.intersection(&b).count(), a.len());
    assert_eq!(a.difference(&b).count(), 0); // both set must be equal.
}

#[test]
fn test_sanity() {
    let mut acc = SimpleAccumulator::with_fixed_capacity(&vec![], 3);
    acc.push(1.0);
    acc.push(2.0);
    assert_float_eq!(acc.mean(), 1.5, abs_all <= 1e-4);
    acc.push(3.0);
    assert_float_eq!(acc.mean(), 2.0, abs_all <= 1e-4);
    acc.push(2.0);
    assert_float_eq!(acc.mean(), 2.0, abs_all <= 1e-4);
    acc.push(2.0);
    assert_float_eq!(acc.mean(), 2.0, abs_all <= 1e-4);
    acc.push(10.0);
    assert_float_eq!(acc.mean(), 20.0 / 6.0, abs_all <= 1e-4);
}

/// Compare with third-party implementation.
#[test]
fn test_sanity_with_random_values() {
    use watermill::quantile::Quantile;
    use watermill::stats::Univariate;
    use watermill::variance::Variance;

    let mut acc = SimpleAccumulator::with_fixed_capacity(&vec![], 10);
    let mut running_median: Quantile<f64> = Quantile::new(0.5f64).unwrap();
    let mut running_var: Variance<f64> = Variance::default();

    let total = 1000;
    for _i in 0..total {
        let a = 100.0 * (rand::random::<f64>() - 0.5);
        acc.push(a);
        running_median.update(a);
        running_var.update(a);
        println!("{a}");
    }
    println!("{acc:#?}");
    assert_eq!(acc.total, total);
    assert_float_eq!(acc.variance(), running_var.get(), ulps_all <= 4);
    assert_float_eq!(acc.median(), running_median.get(), ulps_all <= 4);
}
