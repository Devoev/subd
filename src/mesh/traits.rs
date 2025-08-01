use crate::cells::node::NodeIdx;
use crate::cells::geo;
use crate::index::dimensioned::Dimensioned;
use nalgebra::RealField;

/// Topology of a mesh consisting of cells.
pub trait MeshTopology<'a, const K: usize> {
    /// Topological element in the mesh.
    type Elem; //: topo::Cell<Const<K>>; todo: add bound

    /// Node iterator.
    type NodeIter: Iterator<Item = NodeIdx>;

    /// Element iterator.
    type ElemIter: Iterator<Item = Self::Elem>;

    /// Returns the total number of nodes in this mesh.
    fn num_nodes(&self) -> usize;

    /// Returns the total number of elements in this mesh.
    fn num_elems(&self) -> usize;

    /// Returns an iterator over all nodes in this mesh.
    fn node_iter(&'a self) -> Self::NodeIter;

    /// Returns an iterator over all elements in this mesh.
    fn elem_iter(&'a self) -> Self::ElemIter;
}

/// Mesh consisting of connected cells, also called *elements*.
pub trait Mesh<'a, T: RealField, X: Dimensioned<T, K>, const K: usize, const M: usize>: MeshTopology<'a, K> {
    /// Geometric element in the mesh.
    type GeoElem: geo::Cell<T, X, K, M>;

    /// Returns the geometric element corresponding to the given topological `elem`.
    fn geo_elem(&'a self, elem: &Self::Elem) -> Self::GeoElem;
}

// todo: update signature of elems and add more methods
//  maybe change lifetime parameter