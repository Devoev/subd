pub mod univariate_mesh;

/// A mesh of a `D`-dimensional geometry.
pub trait Mesh {
    
    /// Node iterator.
    type NodeIter : Iterator;

    /// Element iterator.
    type ElemIter : Iterator;

    /// Returns the number of nodes in the mesh.
    fn num_nodes(self) -> usize;

    /// Returns an iterator over the meshes nodes.
    fn nodes(self) -> Self::NodeIter;

    /// Returns the number of elements in the mesh.
    fn num_elems(self) -> usize;

    /// Returns an iterator over the meshes elements.
    fn elems(self) -> Self::ElemIter;
}