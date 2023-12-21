#  SimpleAccumulator

[![Crates.io](https://img.shields.io/crates/v/simple_accumulator)](https://crates.io/crates/simple_accumulator)
[![docs.rs](https://img.shields.io/docsrs/simple_accumulator)](https://docs.rs/simple_accumulator/latest/simple_accumulator/struct.SimpleAccumulator.html)
[![Crates.io](https://img.shields.io/crates/d/simple_accumulator)](https://docs.rs/simple_accumulator/latest/simple_accumulator/struct.SimpleAccumulator.html)

This crate is inspired by [Boost::Accumulator](
https://www.boost.org/doc/libs/1_84_0/doc/html/accumulators.html) which supports
incremental statistical computation (online algorithms). _This is a work in
progress but usable. Please write integration tests before using it in
production._ 

Read
[Documentation](https://docs.rs/simple_accumulator/latest/simple_accumulator/struct.SimpleAccumulator.html)

# Notes

- 2023-12-20: Version 0.6 is a major rewrite that fix many embarassing bugs. In
  0.6+, we are relying on
  [watermill](https://docs.rs/watermill/latest/watermill/#quickstart) crate for
  underlying algorithms.

# Usage: 

```rust
use simple_accumulator::SimpleAccumulator;

fn main() {
    let k = [1, 2, 3, 4];

    // If second argument is `None` then accumulator stores all the data. 
    let mut x = SimpleAccumulator::new(&k, Some(10));

    println!("{:?}", x);
    
    x.push(5);
    println!("{:?}", x);

    print!("{}", x.mean());
    print!("{}", x.median());
    print!("{}", x.variance());
    print!("{}", x.sum());
    print!("{}", x.kurtosis());
    ...
}
```
