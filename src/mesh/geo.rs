// todo: implement geometric Cell trait and make C: Cell

/// Mesh consisting of cells of type [`C`], also called *elements*.
pub trait Mesh<const K: usize, C> {
    /// Element iterator.
    type Elems: Iterator<Item = C>;
    
    /// Returns an iterator over all elements in this mesh.
    fn elems(&self) -> Self::Elems;
}