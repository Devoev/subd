use crate::bspline::spline_geo::SplineGeo;
use crate::cells::geo;
use crate::cells::hyper_rectangle::HyperRectangle;
use nalgebra::RealField;

// todo: is BezierElem even needed somewhere?
//  bezier elem should provide access to a local basis. 
//  It needs information about knot spans, such that local evaluations are possible

/// A Bezier element embedded in [`M`]-dimensional Euclidean space.
/// It is the image of a [`D`]-dimensional reference [HyperRectangle] through the geometric mapping.
#[derive(Debug, Clone, Copy)]
pub struct BezierElem<'a, T: RealField + Copy, const D: usize, const M: usize> {
    /// The reference element in a cartesian mesh.
    pub ref_elem: HyperRectangle<T, D>,

    // todo: see BezierMesh
    /// Spline parametrization mapping the reference mesh to the physical Bezier mesh.
    pub geo_map: &'a SplineGeo<'a, T, [T; D], D, M>
}

impl <'a, T: RealField + Copy, const D: usize, const M: usize> BezierElem<'a, T, D, M> {
    /// Constructs a new [`BezierElem`] from the given `ref_elem` and `geo_map`.
    pub fn new(ref_elem: HyperRectangle<T, D>, geo_map: &'a SplineGeo<'a, T, [T; D], D, M>) -> Self {
        BezierElem { ref_elem, geo_map }
    }
}

impl <'a, T: RealField + Copy, const D: usize, const M: usize> geo::Cell<T, [T; D], D, M> for BezierElem<'a, T, D, M> {
    type RefCell = HyperRectangle<T, D>;
    type GeoMap = &'a SplineGeo<'a, T, [T; D], D, M>;

    fn ref_cell(&self) -> Self::RefCell {
        self.ref_elem
    }

    fn geo_map(&self) -> Self::GeoMap {
        self.geo_map
    }
}