use itertools::Itertools;
use nalgebra::{center, point, DMatrix, OMatrix, Point2, SMatrix, U2, U4};
use nalgebra_sparse::CooMatrix;
use std::f64::consts::PI;
use std::iter::zip;
use subd::cells::quad::QuadNodes;
use subd::diffgeo::chart::Chart;
use subd::element::quad::Quad;
use subd::element::traits::Element;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreBi;
use subd::quadrature::traits::Quadrature;
use subd::space::eval_basis::EvalBasis;
use subd::space::Space;
use subd::subd::lin_subd::edge_basis::WhitneyEdgeQuad;

fn main() {
    // Define geometry
    let coords = make_geo(1.0, 5);

    // Define mesh
    let faces = vec![
        QuadNodes::new(0, 10, 1, 2),
        QuadNodes::new(0, 2, 3, 4),
        QuadNodes::new(0, 4, 5, 6),
        QuadNodes::new(0, 6, 7, 8),
        QuadNodes::new(0, 8, 9, 10),
    ];
    let mut msh = QuadVertexMesh::new(coords, faces);
    msh = msh.lin_subd().unpack();

    // Define edge space
    let space = Space::<f64, _>::new(WhitneyEdgeQuad::new(&msh));

    // Define quadrature
    let ref_quad = GaussLegendreBi::with_degrees(2, 2);
    let quad = PullbackQuad::new(ref_quad);

    // Build mass matrix
    // let mass_matrix = Hodge::new(&msh, &space).assemble(quad);
    // let mass_matrix = DMatrix::from(&mass_matrix);

    // Create empty matrix
    let mut mij = CooMatrix::<f64>::zeros(space.dim(), space.dim());

    // Iteration over all mesh elements
    for (elem, cell) in msh.elem_cell_iter() {
        // Build local space and local mass matrix
        let (sp_local, idx) = space.local_space_with_idx(cell);

        // Local assembly
        // Evaluate all basis functions and store in 'buf'
        let ref_elem = elem.parametric_element();
        let buf: Vec<OMatrix<f64, U2, U4>> = quad.nodes_ref::<f64, Quad<f64, 2>>(&ref_elem)
            .map(|p| sp_local.basis.eval(p)).collect();
        let buf_g_inv: Vec<SMatrix<f64, 2, 2>> = quad.nodes_ref::<f64, Quad<f64, 2>>(&ref_elem)
            .map(|p| {
                let j = elem.geo_map().eval_diff(p);
                (j.transpose() * j).try_inverse().unwrap()
            }).collect();

        // Calculate pullback of product uv
        let uv_pullback = |b: &OMatrix<f64, U2, U4>, g_inv: &SMatrix<f64, 2, 2>, i: usize, j: usize| {
            // Eval basis
            let bi = b.column(i);
            let bj = b.column(j);

            // Calculate integrand
            (bi.transpose() * g_inv * bj).x
            // bi.dot(&bj)
        };

        // Integrate over all combinations of b[i] * b[j] and integrate
        let num_basis = sp_local.dim();
        let mij_iter = (0..num_basis).cartesian_product(0..num_basis)
            .map(|(i, j)| {
                let integrand = zip(&buf, &buf_g_inv)
                    .map(|(b, g_inv)| uv_pullback(b, g_inv, i, j));
                quad.integrate_elem(&elem, integrand)
            });

        // Assemble matrix
        let mij_local = DMatrix::from_iterator(num_basis, num_basis, mij_iter);

        // Fill global mass matrix with local entries
        let idx_local_global = idx.enumerate();
        for ((i_local, i), (j_local, j)) in idx_local_global.clone().cartesian_product(idx_local_global) {
            mij.push(i, j, mij_local[(i_local, j_local)]);
        }
    }

    let mass_matrix = DMatrix::from(&mij);
    // println!("{}", mass_matrix);
    // println!("{:?}", mass_matrix.shape());
    println!("{}", mass_matrix.rank(1e-10));
    let evs = mass_matrix.eigenvalues().unwrap();
    println!("{}", evs);
}

/// Constructs the center and corner points of a regular `n`-gon of radius `r`.
fn make_geo(r: f64, n: usize) -> Vec<Point2<f64>> {
    // Angle between segments
    let phi = 2.0*PI / n as f64;

    let mut coords = vec![point![0.0, 0.0]];
    for i in 0..n {
        let phi_i = phi * i as f64;
        let phi_j = phi * (i + 1) as f64;
        let pi = point![r * phi_i.cos(), r * phi_i.sin()];
        let pj = point![r * phi_j.cos(), r * phi_j.sin()];
        coords.push(pi);
        coords.push(center(&pi, &pj));
    }
    coords
}