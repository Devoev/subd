use nalgebra::{Const, RealField};
use crate::cells::geo;
use crate::index::dimensioned::Dimensioned;

/// Mesh consisting of cells of type [`C`], also called *elements*.
pub trait Mesh<'a, T: RealField, X: Dimensioned<T, K>, const K: usize, const M: usize, C: geo::Cell<T, X, K, M>> {
    /// Element iterator.
    type Elems: Iterator<Item = C>;

    /// Returns an iterator over all elements in this mesh.
    fn elems(&'a self) -> Self::Elems;
}

// todo: update signature of elems and add more methods
//  maybe change lifetime parameter