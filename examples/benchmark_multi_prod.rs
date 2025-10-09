use itertools::Itertools;
use std::hint::black_box;
use std::time::Instant;
use subd::knots::knot_span::KnotSpan;
use subd::knots::knot_vec::KnotVec;
use subd::mesh::knot_mesh::KnotMesh;

fn main() {
    // Number of iterations over the entire mesh. This is usually low.
    // For every computation on the entire mesh exactly one iteration is needed
    const NUM_ITER: usize = 2;

    // Build mesh
    let n = 300;
    let p = 3;
    let xi = KnotVec::<f64>::new_open_uniform(n, p);
    let msh = KnotMesh::from_knots([xi.clone(), xi.clone(), xi.clone()]);

    // Iterate once and collect
    let start = Instant::now();
    let cells = msh.cell_iter().collect_vec();
    for _ in 0..NUM_ITER {
        for cell in &cells {
            black_box(cell);
        }
    }
    let time_collect = start.elapsed();
    let size_collect = size_of::<KnotMesh<f64, 3>>() + size_of::<Vec<[KnotSpan; 3]>>() + cells.len() * size_of::<[KnotSpan; 3]>();

    // Iterate multiple times and never collect
    let start = Instant::now();
    for _ in 0..NUM_ITER {
        for cell in msh.cell_iter() {
            black_box(cell);
        }
    }
    let time_no_collect = start.elapsed();
    let size_no_collect = size_of::<KnotMesh<f64, 3>>();

    println!(
        "Took {:?} to iterate {NUM_ITER} times over all cells (without collection). Requires {:.3e} bytes of storage",
        time_no_collect, size_no_collect
    );
    println!(
        "Took {:?} to iterate {NUM_ITER} times over all cells (with collection). Requires {:.3e} bytes of storage",
        time_collect, size_collect
    );
    println!(
        "Collecting the cells is {:.3} % faster than not collecting them",
        (time_no_collect.as_secs_f64() - time_collect.as_secs_f64()) / time_no_collect.as_secs_f64() * 100.0
    );
    println!(
        "Collecting the cells requires {:.3e} more storage than not collecting them",
        (size_collect - size_no_collect) as f64 / size_no_collect as f64
    );
}