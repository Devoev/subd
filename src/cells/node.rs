use crate::cells::traits::{Cell, CellConnectivity, ToElement};
use crate::mesh::traits::VertexStorage;
use nalgebra::{Const, DimName, DimNameSub, Point, Scalar, U0};

/// Node in a mesh represented by a global *linear* index.
pub type Node = usize;

impl Cell for Node {
    type Dim = U0;
    type Node = usize;

    fn nodes(&self) -> &[Self::Node] {
        &[*self]
    }
}

// impl <T: Scalar, const M: usize> ToGeoCell<T, Const<M>> for NodeIdx {
//     type GeoCell = Point<T, M>;
//     type Coords = Vec<Point<T, M>>;
//
//     fn to_geo_cell(&self, coords: &Self::Coords) -> Self::GeoCell {
//         coords.vertex(self.0)
//     }
// }

impl CellConnectivity for Node {
    fn is_connected<M: DimName>(&self, other: &Self, _dim: M) -> bool
    where
        U0: DimNameSub<M>
    {
        self == other
    }
}