use approx::assert_abs_diff_eq;
use nalgebra::{DefaultAllocator, Dim, OMatrix};
use nalgebra::allocator::Allocator;
use rand::random_range;
use crate::common::mesh_examples::make_unit_square_mesh;
use subd::bspline::de_boor::MultiDeBoor;
use subd::bspline::space::BsplineSpace;
use subd::knots::knot_vec::KnotVec;
use subd::mesh::knot_mesh::KnotMesh;
use subd::subd::lin_subd::basis::{PlBasisQuad, PlSpaceQuad};

#[test]
fn pl_basis() {
    // Standard PL basis
    let msh_pl = make_unit_square_mesh().lin_subd().unpack();
    let space_pl = PlSpaceQuad::new(PlBasisQuad(&msh_pl));

    // Lowest order B-Spline basis
    let xi = KnotVec::<f64>::new_open_uniform(3, 1);
    let msh_spline = KnotMesh::from_knots([xi.clone(), xi.clone()]);
    let basis = MultiDeBoor::from_knots(msh_spline.knots.clone(), [3, 3], [1, 1]);
    let space_spline = BsplineSpace::new(basis);

    // Get first element
    let elem_pl = &msh_pl.elems[0];
    let elem_spline = msh_spline.elems().next().unwrap();

    // Get random parametric values
    let u = random_range(0.0..=1.0);
    let v = random_range(0.0..=1.0);

    /// Reorder to columns to match B-Spline IGA lexicographical ordering.
    fn to_lex_ordering<R: Dim, C: Dim>(matrix: &mut OMatrix<f64, R, C>) where DefaultAllocator: Allocator<R, C> {
        matrix.swap_columns(1, 2);
        matrix.swap_columns(1, 3);
    }

    // Compare evaluation
    let mut eval_pl = space_pl.eval_on_elem(&elem_pl, (u, v));
    let eval_spline = space_spline.eval_on_elem(&elem_spline, [u / 2.0, v / 2.0]);
    to_lex_ordering(&mut eval_pl);
    assert_abs_diff_eq!(eval_pl.as_slice(), eval_spline.as_slice());

    // Compare gradient evaluation
    let mut eval_pl = space_pl.eval_grad_on_elem(&elem_pl, (u, v));
    let eval_spline = space_spline.eval_grad_on_elem(&elem_spline, [u / 2.0, v / 2.0]);
    to_lex_ordering(&mut eval_pl);
    assert_abs_diff_eq!(eval_pl.as_slice(), eval_spline.as_slice()); // todo: why is a scale factor of 2 required here?
}