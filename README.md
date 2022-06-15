# customVector

Usage: 

```rust
use custom_vector::CustomVector;

fn main() {
    let k = [1, 2, 3, 4];

    let mut x = CustomVector::new(&k);

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
CustomVector {
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
CustomVector {
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
CustomVector {
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
CustomVector {
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
    median: 2.0,
    len: 3,
}
```