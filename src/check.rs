use rand::Rng;

use super::SimpleAccumulator;

fn check_error_mean() {
    let mut acc = SimpleAccumulator::new::<f64>(&[], true);
    let mut error_mean:Vec<f64> = Vec::new();
    let mut len_per_error_mean:Vec<f64> = Vec::new();
    for _i in 0..1000 {
        for _j in 0..1000{
            let data = rand::thread_rng().gen::<f64>();
            acc.push(data);
        }
        let mean = acc.mean;
        let offline_mean = acc.calculate_mean();
        let error_diff = (offline_mean - mean)/acc.len as f64; 
        error_mean.push(error_diff);
        len_per_error_mean.push(acc.len as f64);
        println!("Error for mean: {}  Number of data points: {} \n", error_diff, acc.len);
    }
}