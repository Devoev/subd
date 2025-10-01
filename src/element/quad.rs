use nalgebra::{Point, RealField, SVector};
use crate::element::traits;
use crate::element::lerp::BiLerp;
use crate::element::unit_cube::UnitCube;

/// An `M` dimensional quadrilateral element, defined by four `vertices`.
#[derive(Debug, Clone, Copy)]
pub struct Quad<T: RealField, const M: usize> {
    /// Corner vertices of the quadrilateral.
    pub vertices: [Point<T, M>; 4]
}

impl<T: RealField, const M: usize> Quad<T, M> {
    /// Constructs a new `Quad` from the given `vertices`.
    pub fn new(vertices: [Point<T, M>; 4]) -> Self {
        Quad { vertices }
    }

    /// Computes the centroid of this face.
    pub fn centroid(&self) -> Point<T, M> {
        let centroid = self.vertices
            .iter()
            .map(|p| &p.coords)
            .sum::<SVector<T, M>>() / T::from_f64(4.0).unwrap();
        Point::from(centroid)
    }
}

impl <T: RealField + Copy, const M: usize> traits::Element<T> for Quad<T, M> {
    type ParametricElement = UnitCube<2>;
    type GeoMap = BiLerp<T, M>;

    fn parametric_element(&self) -> Self::ParametricElement {
        UnitCube
    }

    fn geo_map(&self) -> Self::GeoMap {
        BiLerp::new(self.vertices)
    }
}