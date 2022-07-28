//! Tests

use simple_accumulator::SimpleAccumulator;

#[test]
fn test_sanity_push_in_fixed_capacity() {
    const CAPACITY : usize = 3;
    let mut acc = SimpleAccumulator::with_fixed_capacity::<f64>(&vec![], CAPACITY, true);

    let data = vec![0.0, 1.1, 2.2, 3.3, 4.4];
    for &v in &data {
        acc.push(v);
    }
    println!("{:?}", acc);
    assert_eq!(acc.vec.len(), 3);
    assert_eq!(acc.vec, vec![3.3, 4.4, 2.2]);

    acc.push(5.5);
    assert_eq!(acc.vec.len(), 3);
    assert_eq!(acc.vec, vec![3.3, 4.4, 5.5]);
}
