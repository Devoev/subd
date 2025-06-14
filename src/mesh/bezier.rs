use crate::bspline::spline_geo::SplineGeo;
use crate::cells::bezier_elem::BezierElem;
use crate::mesh::cartesian::CartMesh;
use crate::mesh::geo;
use crate::mesh::geo::Mesh;
use nalgebra::RealField;

/// A Bezier mesh embedded in [`M`]-dimensional Euclidean space.
/// Generated by a NURBS or B-Spline map of the [`D`]-dimensional parametric [CartMesh].
/// An exemplar mapping can in 2D be schematically visualized as
/// ```text
/// ^ v                           +--+----
/// |    +---+---+---+            |   ⟍    ⟍
/// |    |   |   |   |            |    / +---+
/// |    +---+---+---+       F    +----+   ⟍    ⟍
/// |    |   |   |   |     ====>  |   /  ⟍   ⟍    ⟍
/// |    +---+---+---+            +--+     +--+-----+
/// |    |   |   |   |                 ⟍ ⟋ ⟍   ⟍   |
/// |    +---+---+---+                   |   |   |   |
/// |                 u                  +---+---+---+
/// +------------------>
/// ```
/// where `F` is the geometrical mapping.
pub struct BezierMesh<'a, T: RealField + Copy, const D: usize, const M: usize> {
    /// The cartesian mesh of the parametric domain.
    pub ref_mesh: CartMesh<T, D>,
    
    // todo:
    //  1. allow for different geo maps. Especially Nurbs maps
    //  2. don't require DeBoorMulti. For that, possibly encode D in BsplineBasis trait
    /// Spline parametrization mapping the reference mesh to the physical Bezier mesh.
    pub geo_map: SplineGeo<'a, T, [T; D], D, M>
}

impl <'a, T: RealField + Copy, const D: usize, const M: usize> BezierMesh<'a, T, D, M> {
    /// Constructs a new [`BezierMesh`] from the given `ref_mesh` and `geo_map`.
    pub fn new(ref_mesh: CartMesh<T, D>, geo_map: SplineGeo<'a, T, [T; D], D, M>) -> Self {
        BezierMesh { ref_mesh, geo_map }
    }
    
    /// Returns an iterator over all elements in this mesh.
    pub fn elems(&self) -> impl Iterator<Item=BezierElem<T, D, M>> {
        self.ref_mesh.elems()
            .map(|ref_elem| BezierElem::new(ref_elem, &self.geo_map))
    }
}

impl <'a, T: RealField + Copy, const D: usize, const M: usize> geo::Mesh<'a, T, [T; D], D, M, BezierElem<'a, T, D, M>> for BezierMesh<'a, T, D, M> {
    type Elems = impl Iterator<Item=BezierElem<'a, T, D, M>>;

    fn elems(&'a self) -> Self::Elems {
        self.ref_mesh.elems()
            .map(move |ref_elem| BezierElem::new(ref_elem, &self.geo_map))
    }
}