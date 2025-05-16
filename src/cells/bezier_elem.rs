use nalgebra::{Const, Dyn, RealField};
use crate::bspline::de_boor::DeBoorMulti;
use crate::bspline::spline_geo::SplineGeo;
use crate::cells::geo;
use crate::cells::hyper_rectangle::HyperRectangle;

// todo: is BezierElem even needed somewhere?

/// A Bezier element embedded in [`M`]-dimensional Euclidean space.
/// It is the image of a [`D`]-dimensional reference [HyperRectangle] through the geometric mapping.
pub struct BezierElem<'a, T: RealField + Copy, const D: usize, const M: usize> {
    /// The reference element in a cartesian mesh.
    pub ref_elem: HyperRectangle<T, D>,

    // todo: see BezierMesh
    /// Spline parametrization mapping the reference mesh to the physical Bezier mesh.
    pub geo_map: &'a SplineGeo<'a, T, [T; D], DeBoorMulti<T, D>, M, Dyn>
}

impl <'a, T: RealField + Copy, const D: usize, const M: usize> BezierElem<'a, T, D, M> {
    /// Constructs a new [`BezierElem`] from the given `ref_elem` and `geo_map`.
    pub fn new(ref_elem: HyperRectangle<T, D>, geo_map: &'a SplineGeo<'a, T, [T; D], DeBoorMulti<T, D>, M, Dyn>) -> Self {
        BezierElem { ref_elem, geo_map }
    }
}

impl <'a, T: RealField + Copy, const D: usize, const M: usize> geo::Cell<Const<D>> for BezierElem<'a, T, D, M> {
    type Parametrization = &'a SplineGeo<'a, T, [T; D], DeBoorMulti<T, D>, M, Dyn>;

    fn parametrization(&self) -> Self::Parametrization {
        self.geo_map
    }
}