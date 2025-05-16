use nalgebra::Const;
use crate::cells::geo;

/// Mesh consisting of cells of type [`C`], also called *elements*.
pub trait Mesh<const K: usize, C: geo::Cell<Const<K>>> {
    /// Element iterator.
    type Elems: Iterator<Item = C>;
    
    /// Returns an iterator over all elements in this mesh.
    fn elems(&self) -> Self::Elems;
}