use rand::Rng;
use std::fs::File;
use std::io::prelude::*;
//use std::io::{self, ErrorKind};
use std::path::Path;

use simple_accumulator::SimpleAccumulator;

use plotly::common::Mode;
//use plotly::layout::{Axis, BarMode, Layout, Legend, TicksDirection};
use plotly::{Plot, Scatter};

fn main() {
    let mut acc = SimpleAccumulator::new::<f64>(&[], true);
    let mut error_mean:Vec<f64> = Vec::new();
    let mut len_per_error_mean:Vec<f64> = Vec::new();
    let base:i32 = 10;
    let multiplier = base.pow(5) as f64;

    //Write error into a text file
    let path = Path::new("error_in_mean.txt");
    let display = path.display();
    // Open a file in write-only mode, returns `io::Result<File>`
    let mut file = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why),
        Ok(file) => file,
    };

    for _i in 0..1000 {
        for _j in 0..1000{
            let data = rand::thread_rng().gen::<f64>();
            acc.push(data);
        }
        let mean = acc.mean;
        let offline_mean = acc.calculate_mean();
        let error_diff = (offline_mean - mean)/acc.len as f64; 
        error_mean.push(error_diff*multiplier);
        len_per_error_mean.push(acc.len as f64);

        // Write the error to `file`, returns `io::Result<()>`
        match file.write_all(format!("Error for mean: {}  Number of data points: {} \n", error_diff*multiplier, acc.len).as_bytes()) {
            Err(why) => panic!("couldn't write to {}: {}", display, why),
            Ok(_) => println!("\n successfully wrote to {}", display),
        }
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