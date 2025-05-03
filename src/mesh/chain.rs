use nalgebra::{DimName, DimNameDiff, DimNameSub, U1};
use crate::mesh::cell::CellTopo;

/// Topology of a `K`-chain inside a mesh.
pub trait ChainTopo<K: DimName + DimNameSub<U1>, C: CellTopo<K>> {

    /// Topology of the `K-1`-dimensional boundary of this chain.
    type Boundary<L: DimName>;

    /// Returns the [boundary topology](Self::Boundary) of this chain.
    fn boundary(&self) -> Self::Boundary<DimNameDiff<K, U1>>;
    
    /// Returns a slice of all [cell topologies](CellTopo) in this chain.
    fn cells(&self) -> &[C];
}