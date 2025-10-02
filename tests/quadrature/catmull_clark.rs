//! Test quadrature rules on a Catmull-Clark surface around an irregular patch.

use approx::assert_abs_diff_eq;
use crate::common::mesh_examples::make_pentagon_mesh;
use subd::mesh::cell_topology::Mesh;
use subd::quadrature::pullback::PullbackQuad;
use subd::quadrature::tensor_prod::{GaussLegendreBi, GaussLegendreMulti};
use subd::quadrature::traits::Quadrature;
use subd::subd::catmull_clark::mesh::CatmarkMesh;
use subd::subd::catmull_clark::quadrature::SubdUnitSquareQuad;

#[test]
fn test() {
    // Define quadrature
    let p = 2;
    let m_max = 15;
    let gauss_quad = GaussLegendreBi::with_degrees(p, p);
    let lin_quad = PullbackQuad::new(gauss_quad.clone());
    let subd_quad = SubdUnitSquareQuad::new(gauss_quad.clone(), m_max);
    let catmark_quad = PullbackQuad::new(subd_quad);

    // Make mesh
    let quad_msh = make_pentagon_mesh().lin_subd().unpack();
    let catmark_msh = CatmarkMesh::from(quad_msh.clone());

    // Perform integration on linear surface
    let area_exact = quad_msh.elems.iter()
        .map(|elem| {
            let patch = quad_msh.geo_elem(&elem);
            lin_quad.integrate_fn_elem(&patch, |_| 1.0)
        })
        .sum::<f64>();

    // println!("Linear mesh has area {} (exact value)", area_exact);

    // Perform integration on Catmull-Clark surface
    let area = catmark_msh.elems.iter()
        .map(|elem| {
            let patch = catmark_msh.geo_elem(&elem);
            catmark_quad.integrate_fn_elem(&patch, |_| 1.0)
        })
        .sum::<f64>();

    // println!("Catmull-Clark mesh has area {}", area);
    // println!("Relative error of Catmull-Clark area = {}", (area_exact - area).abs());
    // println!("Absolute error of Catmull-Clark area = {} %", (area_exact - area).abs() / area_exact * 100.0);

    assert_abs_diff_eq!(area_exact, area, epsilon = 1e-5);
}