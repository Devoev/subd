use iter_num_tools::lin_space;
use itertools::Itertools;
use nalgebra::{matrix, DVector, RowDVector};
use subd::bspline::de_boor::MultiDeBoor;
use subd::bspline::space::BsplineSpace;
use subd::bspline::spline_geo::SplineGeo;
use subd::knots::knot_span::KnotSpan;
use subd::knots::knot_vec::KnotVec;
use subd::mesh::bezier::BezierMesh;
use subd::mesh::knot_mesh::KnotMesh;
use subd::plot::plot_fn_msh;

fn main() {
    // Define (lowest order) geometrical mapping
    let space_geo = BsplineSpace::new_open_uniform([2, 2], [1, 1]);
    let control_points = matrix![
        0.0, 0.0;
        1.0, 0.0;
        0.0, 1.0;
        1.0, 1.0
    ];
    let map = SplineGeo::from_matrix(control_points, &space_geo).unwrap();

    // Construct mesh and space
    let p = 3;
    let n = 8 + p + 1;
    let xi = KnotVec::<f64>::new_open_uniform(n, p);
    let msh_ref = KnotMesh::from_knots([xi.clone(), xi.clone()]);
    let msh = BezierMesh::new(msh_ref, map.clone());
    let basis = MultiDeBoor::from_knots([xi.clone(), xi.clone()], [n, n], [p, p]);
    let space = BsplineSpace::new(basis);

    // Plot boundary basis functions
    let bnd_basis = |elem: &[KnotSpan; 2], x: [f64; 2]| -> f64 {
        space.eval_on_elem(elem, x)[0]
        // let mut global = RowDVector::zeros(space.dim());
        // space.populate_global(&mut global, x);
        // global[10]
    };

    plot_fn_msh(&msh, &bnd_basis, 10, |patch, num| {
        let [u_range, v_range] = patch.ref_elem.ranges();
        (lin_space(u_range, num).collect_vec(), lin_space(v_range, num).collect_vec())
    }).show();
}