//! Tests

use std::collections::HashSet;

use rand::distributions::Uniform;
use rand::Rng;

use simple_accumulator::SimpleAccumulator;

#[test]
fn test_sanity_push_in_fixed_capacity() {
    const CAPACITY: usize = 3;
    let mut acc = SimpleAccumulator::with_fixed_capacity::<f64>(&[], CAPACITY, true);

    let data = vec![0.0, 1.1, 2.2, 3.3, 4.4];
    for &v in &data {
        acc.push(v);
    }
    println!("{:?}", acc);
    assert_eq!(acc.vec.len(), CAPACITY);
    assert_eq!(acc.vec, vec![3.3, 4.4, 2.2]);

    acc.push(5.5);
    assert_eq!(acc.vec.len(), CAPACITY);
    assert_eq!(acc.vec, vec![3.3, 4.4, 5.5]);

    acc.push(6.6);
    assert_eq!(acc.vec.len(), CAPACITY);
    assert_eq!(acc.vec, vec![6.6, 4.4, 5.5]);
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
    let mut acc = SimpleAccumulator::with_fixed_capacity::<f64>(&[], CAPACITY, true);

    // and push values into it.
    for &v in &data {
        acc.push(v);
    }

    assert_eq!(acc.vec.len(), CAPACITY);

    // The values in the accumulator should be the same as lest recent (CAPACITY )
    // values pushed to it. They may not be in the same order.
    let a: HashSet<i32> = acc.vec.iter().map(|&x| x as i32).collect();
    let b: HashSet<i32> = data.into_iter().rev().take(CAPACITY).collect();
    println!("{:?}\n{:?}", a, b);
    assert!(a.len() <= CAPACITY); // duplicates
    assert_eq!(a.intersection(&b).count(), a.len());
    assert_eq!(a.difference(&b).count(), 0); // both set must be equal.
}
