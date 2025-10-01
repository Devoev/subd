use nalgebra::{Point, RealField};
use crate::cells::edge::DirectedEdge;
use crate::element::traits;
use crate::element::lerp::Lerp;
use crate::element::unit_cube::UnitCube;
use crate::mesh::face_vertex::QuadVertexMesh;

/// A line segment, i.e. a straight line bounded by 2 points
/// in [`M`]-dimensional space.
pub struct LineSegment<T: RealField, const M: usize> {
    pub vertices: [Point<T, M>; 2]
}

impl<T: RealField, const M: usize> LineSegment<T, M> {

    /// Constructs a new [`LineSegment`] from the given `vertices`.
    pub fn new(vertices: [Point<T, M>; 2]) -> Self {
        LineSegment { vertices }
    }

    /// Constructs a new [`LineSegment`] from the given `topology` and `msh`.
    pub fn from_msh(topology: DirectedEdge, msh: &QuadVertexMesh<T, M>) -> Self {
        LineSegment::new(topology.0.map(|n| msh.coords(n).clone()))
    }
}

impl <T: RealField + Copy, const M: usize> traits::Element<T> for LineSegment<T, M> {
    type ParametricElement = UnitCube<1>;
    type GeoMap = Lerp<T, M>;

    fn parametric_element(&self) -> Self::ParametricElement {
        UnitCube
    }

    fn geo_map(&self) -> Self::GeoMap {
        Lerp::new(self.vertices[0], self.vertices[1])
    }
}
