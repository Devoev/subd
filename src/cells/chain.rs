use nalgebra::{DimName, DimNameDiff, DimNameSub, U1};
use crate::cells::cell::{Cell, CellBoundary};

/// Topology of a [`K`]-chain inside a mesh.
pub trait Chain<K: DimName, C: Cell<K>> {
    /// Returns a slice of all [cell topologies](CellTopo) in this chain.
    fn cells(&self) -> &[C];
}

/// A [topological chain](Chain) with a boundary.
pub trait ChainBoundary<K: DimName + DimNameSub<U1>, C: CellBoundary<K>>: Chain<K, C> {
    /// Topology of the [`K`]`-1`-dimensional boundary of this chain.
    type Boundary: Chain<DimNameDiff<K, U1>, C::SubCell>;

    /// Returns the [boundary topology](Self::Boundary) of this chain.
    fn boundary(&self) -> Self::Boundary;
}

impl <K: DimName, C: Cell<K>> Chain<K, C> for () {
    fn cells(&self) -> &[C] {
        &[]
    }
}