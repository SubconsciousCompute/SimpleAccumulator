//! Runs the following tests.
//!
//! - Check if histogram computed is 'correct'.

#![cfg(feature = "histogram")]

use histogram_sampler::Sampler;
use rand::distributions::Distribution;
use simple_accumulator::SimpleAccumulator;
use tracing_test::traced_test;

#[traced_test]
#[test]
fn test_hist_correctness() {
    // create an accumulator with fixed capacity.
    let mut acc = SimpleAccumulator::new(&[0.0], Some(10));
    acc.init_histogram(8, 8 /* 2^8 is max value */);

    let sampler = Sampler::from_bins(
        vec![(5, 10), (15, 10), (25, 10), (35, 10)],
        10, /* bin width */
    );

    let mut rng = rand::thread_rng();

    for _i in 0..100 {
        let a = sampler.sample(&mut rng);
        print!("{a} ");
        acc.push(a as f64);
    }

    let hist = acc.histogram();
    println!("computed hist = {hist:?}");
}
