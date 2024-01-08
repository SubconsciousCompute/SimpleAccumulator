//! Runs the following tests.
//!
//! 1. Check if histogram computed is 'correct'.

#![cfg(feature = "histogram")]

use simple_accumulator::SimpleAccumulator;
use std::ops::Shr;
use tracing_test::traced_test;

#[traced_test]
#[test]
fn test_hist_correctness() {
    // create an accumulator with fixed capacity.
    let mut acc = SimpleAccumulator::new(&[0.0], Some(10));
    const N: i64 = 100_000;

    acc.init_histogram(7, 8 /* 2^8 is max value */);
    let mut rng = rand::thread_rng();
    for _i in 0..N {
        let a = rand::random::<u8>();
        acc.push(a as f64);
    }

    let expected_mean = N / (u8::MAX as i64);
    let hist = acc.histogram().unwrap().as_slice();
    let mean = hist.iter().map(|x| *x as i64).sum::<i64>() / (hist.len() as i64);
    println!("computed hist = {hist:?}");
    println!("expected mean = {expected_mean}, mean={mean}");

    let var = hist.iter().map(|x| (*x as i64 - mean).pow(2)).sum::<i64>() / (hist.len() as i64);
    let std = (var as f64).powf(0.5);
    println!("variance={var} std={std}");

    assert!(
        (mean - expected_mean).abs() < 5,
        "{mean} is far away from expecte mean {expected_mean}"
    );
    assert!(std < 24.0, "Standard deviation is too high {std}");
}
