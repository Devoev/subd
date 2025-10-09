use nalgebra::{DimName, DimNameDiff, DimNameSub, U1};
use crate::cells::traits::{CellConnectivity, CellBoundary, Cell};

/// Topology of a chain inside a mesh.
pub trait Chain<C: Cell> {
    /// Returns a slice of all [cell topologies](CellTopo) in this chain.
    fn cells(&self) -> &[C];
}

/// A [topological chain](Chain) with a boundary.
pub trait ChainBoundary<C: CellBoundary>: Chain<C> where C::Dim: DimNameSub<U1> {
    /// Topology of the [`K`]`-1`-dimensional boundary of this chain.
    type Boundary: Chain<C::SubCell>;

    /// Returns the [boundary topology](Self::Boundary) of this chain.
    fn boundary(&self) -> Self::Boundary;
}