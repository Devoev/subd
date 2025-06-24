use nalgebra::{Const, RealField};
use crate::cells::{geo, topo};
use crate::cells::node::NodeIdx;
use crate::index::dimensioned::Dimensioned;

/// Mesh consisting of cells of type [`C`], also called *elements*.
pub trait Mesh<'a, T: RealField, X: Dimensioned<T, K>, const K: usize, const M: usize> {
    /// Topological element in the mesh.
    type Elem; //: topo::Cell<Const<K>>;

    /// Geometric element in the mesh.
    type GeoElem: geo::Cell<T, X, K, M>;

    /// Node iterator.
    type NodesIter: Iterator<Item = NodeIdx>;

    /// Element iterator.
    type ElemsIter: Iterator<Item = Self::Elem>;

    /// Returns the total number of nodes in this mesh.
    fn num_nodes(&self) -> usize;

    /// Returns the total number of elements in this mesh.
    fn num_elems(&self) -> usize;

    /// Returns an iterator over all nodes in this mesh.
    fn nodes(&'a self) -> Self::NodesIter;

    /// Returns an iterator over all elements in this mesh.
    fn elems(&'a self) -> Self::ElemsIter;

    /// Returns the geometric element corresponding to the given topological `elem`.
    fn geo_elem(&'a self, elem: Self::Elem) -> Self::GeoElem;
}

// todo: update signature of elems and add more methods
//  maybe change lifetime parameter