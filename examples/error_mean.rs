use plotly::common::Mode;
use plotly::{Plot, Scatter};
use rand::Rng;
use simple_accumulator::SimpleAccumulator;

fn main() {
    let mut acc = SimpleAccumulator::new::<f64>(&[], true);
    let mut error_mean: Vec<f64> = Vec::new();
    let mut len_per_error_mean: Vec<f64> = Vec::new();
    let base: i32 = 10;
    let multiplier = base.pow(5) as f64;

    println!("Waiting to plot the error data...");
    for _i in 0..1000 {
        for _j in 0..1000 {
            let data = rand::thread_rng().gen::<f64>();
            acc.push(data);
        }
        let mean = acc.mean;
        let offline_mean = acc.calculate_mean();
        let error_diff = (offline_mean - mean) / acc.len as f64;
        error_mean.push(error_diff * multiplier);
        len_per_error_mean.push(acc.len as f64);
    }

    // Plot the error data
    let trace = Scatter::new(len_per_error_mean, error_mean)
        .name("trace")
        .mode(Mode::LinesMarkers);
    let mut plot = Plot::new();
    plot.add_trace(trace);

    plot.show();
    println!("{}", plot.to_inline_html(Some("error_mean_scatter_plot")));
}
