//! Tests

use simple_accumulator::SimpleAccumulator;

#[test]
fn test_variance_fixed_capacity() {
    let mut acc = SimpleAccumulator::with_fixed_capacity(&[], 10);
    let data = vec![31.0, 22.0, 24.0, 22.0, 15.0, 28.0, 20.0, 34.0, 9.0, 17.0];
    for &v in &data {
        acc.push(v);
    }

    println!("{acc:?}");
    assert_ne!(acc.variance(), f64::NAN);
}

#[test]
fn test_variance() {
    let mut acc = SimpleAccumulator::default();
    let data = vec![
        32.0, 24.0, 17.0, 32.0, 19.0, 29.0, 23.0, 23.0, 16.0, 31.0, 23.0, 11.0, 24.0, 35.0, 12.0,
        37.0, 24.0, 13.0, 32.0, 17.0, 25.0, 18.0, 34.0, 16.0, 32.0, 23.0, 25.0, 15.0, 24.0, 22.0,
        35.0, 13.0, 20.0, 37.0, 17.0, 29.0, 24.0, 23.0, 13.0, 22.0, 38.0, 15.0, 22.0, 31.0, 16.0,
        24.0, 34.0, 21.0, 12.0, 24.0, 22.0, 24.0, 36.0, 23.0, 24.0, 17.0, 24.0, 29.0, 17.0, 30.0,
        24.0, 15.0, 32.0, 24.0, 22.0, 24.0, 23.0, 19.0, 19.0, 24.0, 36.0, 22.0, 25.0, 18.0, 16.0,
        37.0, 10.0, 26.0, 33.0, 24.0, 18.0, 30.0, 16.0, 32.0, 23.0, 23.0, 18.0, 19.0, 33.0, 17.0,
        25.0, 22.0, 17.0, 29.0, 25.0, 15.0, 25.0, 33.0, 18.0,
    ];

    for &v in &data {
        acc.push(v);
    }

    assert_ne!(acc.variance(), f64::NAN);
    println!("{}", acc.variance());
}
