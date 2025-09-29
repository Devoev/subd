use nalgebra::{Const, DefaultAllocator, DimName, DimNameSub, Point, Scalar, U0};
use nalgebra::allocator::Allocator;
use crate::cells::topo::{Cell, CellToNodes};
use crate::mesh::traits::VertexStorage;

/// Index of a node aka. vertex in a mesh. Represented by a global index.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct NodeIdx(pub usize);

impl <T: Scalar, const M: usize> Cell<T, Const<M>> for NodeIdx {
    type GeoCell = Point<T, M>;
    type Coords = Vec<Point<T, M>>;

    fn nodes(&self) -> &[crate::mesh::traits::NodeIdx<T, Self::Coords>] {
        todo!()
        &[]
    }

    fn to_geo_cell(&self, coords: &Self::Coords) -> Self::GeoCell {
        coords.vertex(self.0)
    }
}

impl CellToNodes for NodeIdx {
    type Dim = U0;

    fn nodes(&self) -> &[NodeIdx] {
        &[] // todo: return self?
    }

    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        U0: DimNameSub<M>
    {
        self == other
    }
}