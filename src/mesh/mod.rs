pub mod univariate_mesh;

/// A mesh of a `D`-dimensional geometry.
pub trait Mesh {

    /// Node of the mesh.
    type Node;

    /// Element of the mesh.
    type Elem;

    /// Returns the number of nodes in the mesh.
    fn num_nodes(&self) -> usize;

    /// Returns an iterator over the meshes nodes.
    fn nodes(&self) -> impl Iterator<Item=Self::Node>;

    /// Returns the number of elements in the mesh.
    fn num_elems(&self) -> usize;

    /// Returns an iterator over the meshes elements.
    fn elems(&self) -> impl Iterator<Item=Self::Elem>;
}