use crate::mesh::vertex::VertexTopo;
use nalgebra::DimName;

/// Topology of a `K`-cell inside a mesh.
pub trait CellTopo<K: DimName> {
    
    /// Topology of the `K-1`-dimensional boundary of this cell.
    type Boundary;
    
    /// Returns the [boundary topology](Self::Boundary) of this cell.
    fn boundary(&self) -> Self::Boundary;
    
    /// Returns a slice of all node indices in a mesh corresponding to the corner vertices of the cell.
    fn nodes(&self) -> &[VertexTopo];
    
    /// Returns `true` if the cell contains the given `node`.
    fn contains_node(&self, node: VertexTopo) -> bool {
        self.nodes().contains(&node)
    }
}

impl <K: DimName> CellTopo<K> for () {
    type Boundary = ();

    fn boundary(&self) -> Self::Boundary {}

    fn nodes(&self) -> &[VertexTopo] {
        &[]
    }
}