# customVector

Usage: 

```rust
use simple_accumulator::SimpleAccumulator;

fn main() {
    let k = [1, 2, 3, 4];

    // Set field `accumulate` to `false` to not update the field values when
    // changed, you will need to run `calculate_all` to get updated values.
    let mut x = SimpleAccumulator::new(&k, true);

    println!("{:#?}", x);
    x.push(5);

    println!("{:#?}", x);

    x.pop();
    println!("{:#?}", x);

    x.remove(2);
    println!("{:#?}", x);
}

```

Output:

```shell
SimpleAccumulator {
    vec: [
        1.0,
        2.0,
        3.0,
        4.0,
    ],
    mean: 2.5,
    population_variance: 1.25,
    standard_deviation: 1.118033988749895,
    min: 1.0,
    max: 4.0,
    median: 2.5,
    len: 4,
}
SimpleAccumulator {
    vec: [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ],
    mean: 3.0,
    population_variance: 1.8,
    standard_deviation: 1.3416407864998738,
    min: 1.0,
    max: 5.0,
    median: 3.0,
    len: 5,
}
SimpleAccumulator {
    vec: [
        1.0,
        2.0,
        3.0,
        4.0,
    ],
    mean: 2.5,
    population_variance: 1.25,
    standard_deviation: 1.118033988749895,
    min: 1.0,
    max: 4.0,
    median: 2.5,
    len: 4,
}
SimpleAccumulator {
    vec: [
        1.0,
        2.0,
        4.0,
    ],
    mean: 2.3333333333333335,
    population_variance: 1.5833333333333333,
    standard_deviation: 1.2583057392117916,
    min: 1.0,
    max: 4.0,
    median: 2.416666666666667,
    len: 3,
}
```