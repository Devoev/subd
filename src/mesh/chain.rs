use crate::mesh::cell::{CellBoundaryTopo, CellTopo};
use nalgebra::{DimName, DimNameDiff, DimNameSub, U1};

/// Topology of a [`K`]-chain inside a mesh.
pub trait ChainTopo<K: DimName, C: CellTopo<K>> {
    /// Returns a slice of all [cell topologies](CellTopo) in this chain.
    fn cells(&self) -> &[C];
}

/// A [topological chain](ChainTopo) with a boundary.
pub trait ChainBoundaryTopo<K: DimName + DimNameSub<U1>, C: CellBoundaryTopo<K>>: ChainTopo<K, C> {
    /// Topology of the [`K`]`-1`-dimensional boundary of this chain.
    type Boundary: ChainTopo<DimNameDiff<K, U1>, C::BoundaryCell>;
    
    /// Returns the [boundary topology](Self::Boundary) of this chain.
    fn boundary(&self) -> Self::Boundary;
}

impl <K: DimName, C: CellTopo<K>> ChainTopo<K, C> for () {
    fn cells(&self) -> &[C] {
        &[]
    }
}