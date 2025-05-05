//! Topology of a tensor product mesh.

use crate::cells::vertex::VertexTopo;

// todo: rethink this
pub struct TensorProd<const K: usize> {
    /// Breakpoints for each parametric direction.
    pub breaks: [Vec<VertexTopo>; K]
}