//! Tests

use std::collections::HashSet;

use rand::distributions::Uniform;
use rand::Rng;

use simple_accumulator::SimpleAccumulator;

#[test]
fn test_append_in_fixed_capacity() {
    const CAPACITY: usize = 3;
    let mut acc = SimpleAccumulator::with_fixed_capacity::<f64>(&[], CAPACITY, true);

    let data = vec![0.0, 1.1, 2.2, 3.3, 4.4];
    let copy = data.clone();
    acc.append(&copy);

    println!("{acc:?}");
    assert_eq!(acc.len(), CAPACITY);
    assert_eq!(acc.data, vec![2.2, 3.3, 4.4]);

    acc.append(&vec![7, 8]);
    assert_eq!(acc.len(), CAPACITY);
    assert_eq!(acc.data, vec![7.0, 8.0, 4.4]);

    acc.append(&vec![1, 1, 2, 2]);
    assert_eq!(acc.len(), CAPACITY);
    assert_eq!(acc.data, vec![2.0, 2.0, 1.0]);
}

#[test]
fn test_append_in_fixed_capacity_random() {
    // We test that values in the fixed capacity Accumulator must be same as the CAPACITY most
    // recent values pushed to it.
    let mut rng = rand::thread_rng();

    // fill an array of size 1000 with random numbers.
    let data: Vec<i32> = (&mut rng)
        .sample_iter(Uniform::new(0, 100))
        .take(1000)
        .collect();
    let copy = data.clone();

    // Create a SimpleAccumulator for size 10
    const CAPACITY: usize = 10;
    let mut acc = SimpleAccumulator::with_fixed_capacity::<f64>(&[], CAPACITY, true);

    // and push values into it.
    acc.append(&copy);

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
