use crate::mesh::vertex::VertexTopo;
use nalgebra::{Const, DimName, DimNameDiff, DimNameSub, U1};
use crate::mesh::chain::ChainTopo;

/// Topology of a [`K`]-cell inside a mesh.
pub trait CellTopo<K: DimName> {
    /// Returns a slice of all node indices in a mesh corresponding to the corner vertices of the cell.
    fn nodes(&self) -> &[VertexTopo];

    /// Returns `true` if this cell is connected to the `other` cell
    /// by an [`M`]-dimensional (sub-)cell with `M <= K`.
    fn connected_to<const M: usize>(&self, other: &Self) -> bool
        where K: DimNameSub<Const<M>>;
    
    /// Returns `true` if the cell contains the given `node`.
    fn contains_node(&self, node: VertexTopo) -> bool {
        self.nodes().contains(&node)
    }
}

/// A [topological cell](CellTopo) with a boundary.
pub trait CellBoundaryTopo<K: DimName + DimNameSub<U1>>: CellTopo<K> {
    /// Cell topology of the individual cells of the boundary chain.
    type BoundaryCell: CellTopo<DimNameDiff<K, U1>>;
    /// Topology of the [`K`]`-1`-dimensional boundary of this cell.
    type Boundary: ChainTopo<DimNameDiff<K, U1>, Self::BoundaryCell>;

    /// Returns the [boundary topology](Self::Boundary) of this cell.
    fn boundary(&self) -> Self::Boundary;
}

/// A [topological cell](CellTopo) with an ordering of its nodes.
pub trait OrderedCellTopo<K: DimName>: CellTopo<K> {
    
    /// Returns a (globally) sorted copy of this cell. 
    /// 
    /// The new ordering satisfies
    /// for cells `c₁` and `c₂` with the same nodes, that
    /// ```text
    /// sorted(c₁) = sorted(c₂)
    /// ```
    fn sorted(&self) -> Self;
}

impl <K: DimName> CellTopo<K> for () {
    fn nodes(&self) -> &[VertexTopo] {
        &[]
    }

    fn connected_to<const M: usize>(&self, other: &Self) -> bool
    where
        K: DimNameSub<Const<M>>
    {
        false
    }
}