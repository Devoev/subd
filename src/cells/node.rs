use crate::cells::traits::{Cell, CellConnectivity, ToElement};
use crate::mesh::traits::VertexStorage;
use nalgebra::{Const, DimName, DimNameSub, Point, Scalar, U0};

/// Index of a node aka. vertex in a mesh. Represented by a global index.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct NodeIdx(pub usize);

impl Cell for NodeIdx {
    type Dim = U0;
    type Node = usize;

    fn nodes(&self) -> &[Self::Node] {
        &[self.0]
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

impl CellConnectivity for NodeIdx {
    fn is_connected<M: DimName>(&self, other: &Self, _dim: M) -> bool
    where
        U0: DimNameSub<M>
    {
        self == other
    }
}