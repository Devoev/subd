use crate::cells::vertex::VertexTopo;
use crate::cells::topo;
use nalgebra::Const;

/// Topology of a mesh consisting of cells of type [`C`].
pub trait MeshTopology<'a, const K: usize, C: topo::Cell<Const<K>>> {
    /// Node iterator.
    type Nodes: Iterator<Item = VertexTopo>;

    /// Element iterator.
    type Elems: Iterator<Item = C>;

    /// Returns the total number of nodes in this mesh.
    fn num_nodes(&self) -> usize;

    /// Returns the total number of elements in this mesh.
    fn num_elems(&self) -> usize;

    /// Returns an iterator over all nodes in this mesh.
    fn nodes(&'a self) -> Self::Nodes;

    /// Returns an iterator over all elements in this mesh.
    fn elems(&'a self) -> Self::Elems;
}