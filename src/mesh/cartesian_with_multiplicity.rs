use crate::cells::cartesian::CartCell;
use crate::index::dimensioned::{DimShape, Strides};
use crate::knots::breaks_with_multiplicity::BreaksWithMultiplicity;
use crate::knots::knot_span::KnotSpan;
use crate::mesh::cartesian::{CartMesh, NodesIter};
use crate::mesh::traits::{Mesh, MeshTopology};
use itertools::{Itertools, MultiProduct};
use nalgebra::{Point, RealField};
use std::iter::{zip, Map};
use std::vec::IntoIter;

// todo: this is in huge parts copied from cartesian.rs. Merge the two implementations

/// Cartesian mesh built by tensor product of [`BreaksWithMultiplicity<T>`].
/// Same as [`CartMesh`] but with additional information about multiplicities of breakpoints.
pub struct CartWithMultiplicity<T, const D: usize> {
    /// Breakpoints for each parametric direction.
    pub breaks: [BreaksWithMultiplicity<T>; D],

    /// Shape of the parametric directions.
    pub dim_shape: DimShape<D>,

    /// Strides for each parametric direction.
    pub strides: Strides<D>
}

impl <T, const D: usize> CartWithMultiplicity<T, D> {
    /// Constructs a new [`CartWithMultiplicity`] from the given `breaks`, `dim_shape` and `strides`.
    ///
    /// # Panics
    /// If the shape of the `breaks` does not equal the shape of `dim_shape`,
    /// the function will panic.
    pub fn new(breaks: [BreaksWithMultiplicity<T>; D], dim_shape: DimShape<D>, strides: Strides<D>) -> Self {
        let n_breaks = breaks.iter().map(|b| b.len()).collect_vec();
        let n_shape = dim_shape.0;
        assert_eq!(n_breaks, n_shape,
                   "Shape of `breaks` (is {:?}) doesn't equal shape of `dim_shape` (is {:?})",
                   n_breaks, n_shape);
        CartWithMultiplicity { breaks, dim_shape, strides }
    }
}

/// An iterator over the elements (`[KnotSpan; D]`) of a [`CartWithMultiplicity`] mesh.
pub type KnotSpanIter<const D: usize> = Map<MultiProduct<IntoIter<KnotSpan>>, fn(Vec<KnotSpan>) -> [KnotSpan; D]>;

impl <T: RealField + Copy, const D: usize> CartWithMultiplicity<T, D> {
    /// Constructs the vertex point at the given multi-index position `idx`.
    pub fn vertex(&self, idx: [usize; D]) -> Point<T, D> {
        zip(idx, &self.breaks)
            .map(|(i, breaks)| breaks[i].1)
            .collect_array::<D>()
            .unwrap()
            .into()
    }

    /// Returns an iterator over all elements (multi knot spans) in lexicographical order.
    pub fn elems(&self) -> KnotSpanIter<D>{
        self.breaks.iter()
            .map(|b| b.knot_spans())
            .multi_cartesian_product()
            .map(|vec| vec.into_iter().collect_array().unwrap())
    }
}

impl<'a, T: RealField + Copy, const K: usize> MeshTopology<'a, K> for CartWithMultiplicity<T, K> {
    type Elem = [KnotSpan; K];
    type NodeIter = NodesIter<'a, K>;
    type ElemIter = KnotSpanIter<K>;

    fn num_nodes(&self) -> usize {
        self.dim_shape.len()
    }

    fn num_elems(&self) -> usize {
        let mut dim_shape_elems = self.dim_shape;
        dim_shape_elems.shrink(1);
        dim_shape_elems.len()
    }

    fn node_iter(&'a self) -> Self::NodeIter {
        todo!()
    }

    fn elem_iter(&'a self) -> Self::ElemIter {
        self.elems()
    }
}

impl<'a, T: RealField + Copy, const K: usize> Mesh<'a, T, [T; K], K, K> for CartWithMultiplicity<T, K> {
    type GeoElem = CartCell<T, K>;

    fn geo_elem(&'a self, elem: Self::Elem) -> Self::GeoElem {
        let idx = elem.map(|span| span.0);
        let idx_a = idx;
        let idx_b = idx_a.map(|i| i + 1);
        let a = self.vertex(idx_a);
        let b = self.vertex(idx_b);
        CartCell::new(a, b)
    }
}