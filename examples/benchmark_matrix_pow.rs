use std::hint::black_box;
use std::time::Instant;
use approx::assert_abs_diff_eq;
use nalgebra::DMatrix;
use subd::subd::catmull_clark::matrices::{build_extended_mats, EV5};

fn main() {
    let exp = 10u32;
    let num_eval = 100_000;

    // Build subdivision matrix
    let (a, _) = build_extended_mats::<f64>(5);

    // Direct pow
    let start = Instant::now();
    for _ in 0..num_eval {
        let _ = black_box(a.pow(exp));
    }
    let time_direct = start.elapsed();

    // Direct ev decomposition
    let start = Instant::now();

    for _ in 0..num_eval {
        let (v, e, v_inv) = EV5.clone();
        let e_pows = e.map_diagonal(|ev| ev.powi(exp as i32));
        let _ = black_box(v * DMatrix::from_diagonal(&e_pows) * v_inv);
        // assert_abs_diff_eq!(res.norm(), 2.2609363841311287, epsilon = 1e-10);
    }
    let time_ev = start.elapsed();

    println!(
        "Took {:?} to calculate {num_eval} direct matrix powers.",
        time_direct
    );
    println!(
        "Took {:?} to calculate {num_eval} matrix powers using EV decomposition.",
        time_ev
    );
    println!(
        "EV decomposition is {} % faster than direct method",
        (time_direct.as_secs_f64() - time_ev.as_secs_f64()) / time_direct.as_secs_f64() * 100.0
    )
}