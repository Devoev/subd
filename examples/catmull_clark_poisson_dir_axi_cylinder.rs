//! This example numerically solves the *Poisson* problem with Dirichlet boundary conditions
//! using isogeometric analysis with axisymmetric Catmull-Clark basis functions.
//! The problem ist defined as
//! ```text
//! -div grad u = 0   in Ω
//!           u = g   on ∂Ω
//! ```
//! with `Ω` being a cylinder of radius `1`.
//! The discretization is performed on a 2D axisymmetric slice of the cylinder.

use iter_num_tools::lin_space;
use itertools::{izip, Itertools};
use nalgebra::{point, DMatrix, DVector, Dyn, Matrix2, OMatrix, Point2, Vector1, U2};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use num_traits::Zero;
use std::collections::BTreeSet;
use std::io;
use std::iter::zip;
use std::process::Command;
use subd::cells::quad::QuadNodes;
use subd::cells::traits::ToElement;
use subd::cg::cg;
use subd::diffgeo::chart::Chart;
use subd::element::traits::Element;
use subd::mesh::face_vertex::QuadVertexMesh;
use subd::operator::bc::DirichletBc;
use subd::operator::linear_form::LinearForm;
use subd::plot::plot_fn_msh;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::GaussLegendreBi;
use subd::quadrature::traits::Quadrature;
use subd::space::eval_basis::EvalGrad;
use subd::space::lin_combination::LinCombination;
use subd::space::Space;
use subd::subd::catmull_clark::basis::{CatmarkBasis, CatmarkPatchBasis};
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::patch::{CatmarkPatch, CatmarkPatchNodes};
use subd::subd::catmull_clark::quadrature::SubdUnitSquareQuad;
use subd::subd::catmull_clark::space::CatmarkSpace;

/// Number of refinements for the convergence study.
const NUM_REFINE: u8 = 5;

fn main() -> io::Result<()> {
    // Define problem
    // let u = |p: Point2<f64>| Vector1::new(p.y); // Voltage at top and bottom plates
    // let u = |p: Point2<f64>| Vector1::new(p.x*1.25 - 0.25); // Voltage at inner and outer hulls (linear version)
    let u = |p: Point2<f64>| Vector1::new((p.x * 5.0).ln() / 5f64.ln()); // Voltage at inner and outer hulls (log version)

    let r0 = 0.2;
    let coords_square = vec![
        point![r0, 0.0],
        point![1.0, 0.0],
        point![1.0, 1.0],
        point![r0, 1.0]
    ];

    // Define mesh
    let quads = vec![QuadNodes::new(0, 1, 2, 3)];
    let mut quad_msh = QuadVertexMesh::new(coords_square, quads);

    // Convergence study
    let mut n_dofs = vec![];
    let mut errs = vec![];
    for i in 0..NUM_REFINE {
        // Print info
        println!("Iteration {} / {NUM_REFINE}", i+1);

        // Refine and construct Catmark mesh
        quad_msh = quad_msh.catmark_subd().unpack();
        let msh = CatmarkMesh::from(quad_msh.clone());

        // Solve problem
        let (n_dof, err_l2) = solve(&msh, u);

        // Save and print
        n_dofs.push(n_dof);
        errs.push(err_l2);
        println!("  Absolute L2 error ||u - u_h||_2 = {:.7}", err_l2);
    }

    // Write data
    let mut writer = csv::Writer::from_path("examples/errs.csv")?;
    writer.write_record(["n_dofs", "err_l2"])?;
    for data in zip(n_dofs, errs) {
        writer.serialize(data)?;
    }
    writer.flush()?;

    // Call octave plotting function
    Command::new("octave")
        .arg("error_plot.m")
        .current_dir("examples/")
        .output()?;

    Ok(())
}

/// Solves the problem with the boundary data `g` and solution `u` on the given `msh`.
/// Returns the number of DOFs and the L2 error.
fn solve(msh: &CatmarkMesh<f64, 2>, u: impl Fn(Point2<f64>) -> Vector1<f64>) -> (usize, f64) {
    // Define space
    let basis = CatmarkBasis(msh);
    let space = CatmarkSpace::new(basis);

    // Define quadrature
    let p = 4;
    let ref_quad = GaussLegendreBi::with_degrees(p, p);
    let quad = PullbackQuad::new(SubdUnitSquareQuad::new(ref_quad, 1));

    // Assemble system
    let form = LinearForm::new(msh, &space, |_p| Vector1::zero());
    let k_coo = assemble_axi_2d(msh, &space, &quad);
    let f = form.assemble(&quad); // todo: this is only correct, because f = 0. Otherwise, implement axial symmetric
    let k = CsrMatrix::from(&k_coo);

    // Calculate boundary info
    let (idx_bc, u_bc): (BTreeSet<usize>, DVector<f64>) = msh.coords
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            // Voltage at top and bottom plate
            // if p.y == 0.0 { Some((i, 0.0)) }
            // else if p.y == 1.0 { Some((i, 1.0)) }
            // else { None }

            // Voltage at inner and outer hulls
            if p.x == 0.2 { Some((i, 0.0)) }
            else if p.x == 1.0 { Some((i, 1.0)) }
            else { None }
        })
        .unzip();

    // Deflate system
    let dirichlet = DirichletBc::new(msh.num_nodes(), idx_bc, u_bc);
    let (k, f) = dirichlet.deflate(k, f);

    // Solve system
    let uh_dof = cg(&k, &f, f.clone(), f.len(), 1e-13);

    // Inflate system
    let uh = dirichlet.inflate(uh_dof);
    let uh = space.linear_combination(uh)
        .expect("Number of coefficients doesn't match dimension of discrete space");

    // Calculate error
    let err_l2 = error_l2_axi_2d(msh, &quad, &uh, &u);

    // Plot
    // let plot_fn = |cell: &CatmarkPatchNodes, x: (f64, f64)| {
    //     let elem = cell.to_element(&msh.coords);
    //     let p = elem.geo_map().eval(x);
    //     // u(p).x - uh.eval_on_elem(cell, x).x
    //     uh.eval_on_elem(cell, x).x
    // };
    // plot_fn_msh(msh, &plot_fn, 10, |_, num| {
    //     let grid = lin_space(0.0..=1.0, num).collect_vec();
    //     (grid.clone(), grid)
    // }).show();

    (space.dim(), err_l2)
}

/// Quadrature rule for this example.
type CatmarkQuadrature = PullbackQuad<SubdUnitSquareQuad<f64, GaussLegendreBi, 2>>;

/// Assembles the stiffness matrix for the axial-symmetric geometry.
fn assemble_axi_2d(msh: &CatmarkMesh<f64, 2>, space: &CatmarkSpace<f64, 2>, quad: &CatmarkQuadrature) -> CooMatrix<f64> {
    // Create empty matrix
    let mut kij = CooMatrix::zeros(space.dim(), space.dim());

    // Iteration over all mesh elements
    for (elem, cell) in msh.elem_cell_iter() {
        // Build local space and local stiffness matrix
        let (sp_local, idx) = space.local_space_with_idx(cell);
        let kij_local = assemble_axi_2d_local(&elem, &sp_local, quad);

        // Fill global stiffness matrix with local entries
        let idx_local_global = idx.enumerate();
        for ((i_local, i), (j_local, j)) in idx_local_global.clone().cartesian_product(idx_local_global) {
            kij.push(i, j, kij_local[(i_local, j_local)]);
        }
    }

    kij
}

/// Assembles the local stiffness matrix for the axial-symmetric geometry.
fn assemble_axi_2d_local(elem: &CatmarkPatch<f64, 2>, sp_local: &Space<f64, CatmarkPatchBasis>, quad: &CatmarkQuadrature) -> DMatrix<f64> {
    let ref_elem = elem.parametric_element();
    let geo_map = elem.geo_map();

    // Collect radii
    let buf_radii: Vec<_> = quad.nodes_elem(elem)
        .map(|p| p.x) // first coordinate = rho
        .collect();

    // Evaluate gradients
    let buf_grads: Vec<_> = quad.nodes_ref::<f64, CatmarkPatch<f64, 2>>(&ref_elem)
        .map(|p| sp_local.basis.eval_grad(p))
        .collect();

    // Evaluate inverse gram matrices
    let buf_g_inv: Vec<_> = quad.nodes_ref::<f64, CatmarkPatch<f64, 2>>(&ref_elem)
        .map(|p| {
            let j = geo_map.eval_diff(p);
            (j.transpose() * j).try_inverse().unwrap()
        })
        .collect();

    // Calculate pullback of product grad_u * grad_v
    let gradu_gradv_pullback = |grad_b: &OMatrix<f64, U2, Dyn>, g_inv: &Matrix2<f64>, i: usize, j: usize| {
        // Get gradients
        let grad_bi = grad_b.column(i);
        let grad_bj = grad_b.column(j);

        // Calculate integrand
        (grad_bi.transpose() * g_inv * grad_bj).x
    };

    // Integrate over all combinations of grad_b[i] * grad_b[j] and integrate
    let num_basis = sp_local.dim();
    let kij = (0..num_basis).cartesian_product(0..num_basis)
        .map(|(i, j)| {
            // Integrand for stiffness matrix - Add factor of `rho` to fix Jacobian in cylindrical coordinates
            let integrand = izip!(&buf_grads, &buf_g_inv, &buf_radii)
                .map(|(b_grad, g_inv, rho)| rho * gradu_gradv_pullback(b_grad, g_inv, i, j));

            // Perform integration
            quad.integrate_elem(elem, integrand)
        });

    // Assemble matrix
    DMatrix::from_iterator(num_basis, num_basis, kij)
}

/// Calculates the L2 error for the axial-symmetric geometry.
fn error_l2_axi_2d(msh: &CatmarkMesh<f64, 2>, quad: &CatmarkQuadrature, uh: &LinCombination<f64, CatmarkBasis<f64, 2>>, u: &impl Fn(Point2<f64>) -> Vector1<f64>) -> f64 {
    // Iterate over every element and calculate error element-wise
    msh.elem_cell_iter()
        .map(|(elem, cell)| {
            // Get geometrical and reference element
            let parametric_elem = elem.parametric_element();

            // Collect radii
            let buf_radii: Vec<_> = quad.nodes_elem(&elem)
                .map(|p| p.x) // first coordinate = rho
                .collect();

            // Evaluate functions at quadrature nodes of element
            let uh = quad.nodes_ref::<f64, CatmarkPatch<f64, 2>>(&parametric_elem).map(|x| uh.eval_on_elem(cell, x));
            let u = quad.nodes_elem(&elem).map(u);

            // Calculate L2 error on element
            let du_norm_squared = zip(uh, u).map(|(uh, u)| (uh - u).norm_squared());
            let integrand = zip(du_norm_squared, buf_radii).map(|(du, rho)| du * rho);
            quad.integrate_elem(&elem, integrand)
        })
        .sum::<f64>()
        .sqrt()
}