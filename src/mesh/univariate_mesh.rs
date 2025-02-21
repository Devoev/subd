// use crate::knots::knot_vec::{KnotVec, ParametricBezierInterval};
// use crate::mesh::Mesh;
// use nalgebra::{Point, RealField};
// 
// /// A one-dimensional Bezier mesh of a [`SplineCurve`].
// #[derive(Debug, Clone)]
// pub struct UnivariateMesh<T : RealField, const D : usize> {
//     /// The spline curve parametrizing the domain.
//     curve: SplineCurve<T, D>,
//     /// The parametric Bezier mesh.
//     parametric_mesh: KnotVec<T>,
// }
// 
// impl<T: RealField, const D: usize> UnivariateMesh<T, D> {
// 
//     /// Constructs a new [`UnivariateMesh`].
//     pub fn new(curve: SplineCurve<T, D>, parametric_mesh: KnotVec<T>) -> Self {
//         UnivariateMesh { curve, parametric_mesh }
//     }
// }
// 
// impl<'a, T: RealField + Copy, const D: usize> Mesh for &'a UnivariateMesh<T, D> {
//     type NodeIter = impl Iterator<Item=Point<T, D>>;
//     type ElemIter = impl Iterator<Item=BezierElement1D<'a, T, D>>;
// 
//     fn num_nodes(self) -> usize {
//         self.parametric_mesh.num_nodes()
//     }
// 
//     fn nodes(self) -> Self::NodeIter {
//         self.parametric_mesh.nodes().map(|t| self.curve.eval(*t))
//     }
// 
//     fn num_elems(self) -> usize {
//         self.parametric_mesh.num_elems()
//     }
// 
//     fn elems(self) -> Self::ElemIter {
//         self.parametric_mesh.elems()
//             .map(|q| BezierElement1D::new(q, &self.curve))
//     }
// }
// 
// impl<T : RealField + Copy, const D : usize> IntoIterator for &UnivariateMesh<T, D> {
//     type Item = Point<T, D>;
//     type IntoIter = impl Iterator<Item = Self::Item>;
// 
//     fn into_iter(self) -> Self::IntoIter {
//         self.nodes()
//     }
// }
// 
// /// A 1D Bezier element.
// pub struct BezierElement1D<'a, T : RealField + Copy, const D : usize> {
//     /// The associated parametric element.
//     parametric_elem: ParametricBezierInterval<T>,
//     /// The curve mapping the parametric element into physical domain.
//     spline_curve: &'a SplineCurve<T, D>
// }
// 
// impl<T: RealField + Copy, const D : usize> BezierElement1D<'_, T, D> {
// 
//     /// Constructs a new [`BezierElement1D`].
//     fn new(parametric_elem: ParametricBezierInterval<T>, spline_curve: &SplineCurve<T, D>) -> BezierElement1D<'_, T, D> {
//         BezierElement1D { parametric_elem, spline_curve }
//     }
// }