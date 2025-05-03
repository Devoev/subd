use crate::mesh::cell::CellTopo;
use nalgebra::DimName;

/// Topology of a `K`-chain inside a mesh.
pub trait ChainTopo<K: DimName, C: CellTopo<K>> {

    /// Topology of the `K-1`-dimensional boundary of this chain.
    type Boundary;

    /// Returns the [boundary topology](Self::Boundary) of this chain.
    fn boundary(&self) -> Self::Boundary;
    
    /// Returns a slice of all [cell topologies](CellTopo) in this chain.
    fn cells(&self) -> &[C];
}

impl <K: DimName, C: CellTopo<K>> ChainTopo<K, C> for () {
    type Boundary = ();

    fn boundary(&self) -> Self::Boundary { }

    fn cells(&self) -> &[C] {
        &[]
    }
}