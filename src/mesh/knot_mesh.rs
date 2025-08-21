use crate::cells::cartesian::CartCell;
use crate::index::dimensioned::{DimShape, Strides};
use crate::knots::breaks_with_multiplicity::BreaksWithMultiplicity;
use crate::knots::increments::Increments;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;
use crate::mesh::cartesian::{CartMesh, NodesIter};
use crate::mesh::traits::{Mesh, MeshTopology};
use itertools::{Itertools, MultiProduct};
use nalgebra::{Point, RealField};
use std::iter::{zip, Map};
use std::vec::IntoIter;

// todo: this is in huge parts copied from cartesian.rs. Merge the two implementations

/// Cartesian mesh built by tensor product of [`KnotVec<T>`].
/// Same as [`CartMesh`] but with additional information about multiplicities of breakpoints.
#[derive(Debug, Clone)]
pub struct KnotMesh<T, const D: usize> {
    /// Knots for each parametric direction.
    pub knots: [KnotVec<T>; D],

    /// Breakpoint index increments for each parametric direction.
    pub increments: [Increments; D], // todo: replace this with span indices

    /// Shape of the parametric directions.
    pub dim_shape: DimShape<D>,

    /// Strides for each parametric direction.
    pub strides: Strides<D>
}

impl <T, const D: usize> KnotMesh<T, D> {
    /// Constructs a new [`KnotMesh`] from the given `breaks`, `dim_shape` and `strides`.
    ///
    /// # Panics
    /// If the shape of the `breaks` does not equal the shape of `dim_shape`,
    /// the function will panic.
    pub fn new(knots: [KnotVec<T>; D], increments: [Increments; D], dim_shape: DimShape<D>, strides: Strides<D>) -> Self {
        // let n_breaks = knots.iter().map(|xi| xi.len()).collect_vec();
        // let n_shape = dim_shape.0;
        // assert_eq!(n_breaks, n_shape,
        //            "Shape of `breaks` (is {:?}) doesn't equal shape of `dim_shape` (is {:?})",
        //            n_breaks, n_shape);
        KnotMesh { knots, increments, dim_shape, strides }
    }

    // /// Constructs a new [`KnotMesh`] from the given `breaks`.
    // /// The topological information for the shape and strides is constructed from the shape of the breaks.
    // pub fn from_breaks_with_multiplicity(breaks: [BreaksWithMultiplicity<T>; D]) -> Self {
    //     let shape = breaks.iter().map(|b| b.len()).collect_array().unwrap();
    //     let dim_shape = DimShape(shape);
    //     KnotMesh { knots: breaks, dim_shape, strides: Strides::from(dim_shape) }
    // }
}

impl <T: RealField, const D: usize> KnotMesh<T, D> {
    // todo: this implementation is ugly and inefficient. Probably change this.
    /// Constructs a new [`KnotMesh`] from the given `knots`.
    /// The breaks and knot multiplicities are computed using [`BreaksWithMultiplicity::from_knots`].
    pub fn from_knots(knots: [KnotVec<T>; D]) -> Self {
        let breaks = knots.clone().map(|xi| BreaksWithMultiplicity::from_knots(xi));
        let shape = breaks.iter().map(|zeta| zeta.len()).collect_array().unwrap();
        let dim_shape = DimShape(shape);
        let increments = breaks.map(|zeta| Increments::from_multiplicities(zeta));
        KnotMesh { knots, increments, dim_shape, strides: Strides::from(dim_shape) }
    }
}

/// An iterator over the elements (`[KnotSpan; D]`) of a [`KnotMesh`] mesh.
pub type KnotSpanIter<const D: usize> = Map<MultiProduct<IntoIter<KnotSpan>>, fn(Vec<KnotSpan>) -> [KnotSpan; D]>;

impl <T: RealField + Copy, const D: usize> KnotMesh<T, D> {
    /// Constructs the vertex point at the given multi-index position `idx`.
    pub fn vertex(&self, idx: [usize; D]) -> Point<T, D> {
        zip(idx, &self.knots)
            .map(|(i, xi)| xi[i])
            .collect_array::<D>()
            .unwrap()
            .into()
    }

    /// Returns an iterator over all elements (multi knot spans) in lexicographical order.
    pub fn elems(&self) -> KnotSpanIter<D>{
        self.increments.iter()
            .map(|increments| increments.span_indices().collect_vec())
            .multi_cartesian_product()
            .map(|vec| vec.into_iter().collect_array().unwrap())
    }
}

impl<'a, T: RealField + Copy, const K: usize> MeshTopology<'a, K> for KnotMesh<T, K> {
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

impl<'a, T: RealField + Copy, const K: usize> Mesh<'a, T, K, K> for KnotMesh<T, K> {
    type GeoElem = CartCell<T, K>;

    fn geo_elem(&'a self, elem: &Self::Elem) -> Self::GeoElem {
        let idx = elem.map(|span| span.0);
        let idx_a = idx;
        let idx_b = idx_a.map(|i| i + 1);
        let a = self.vertex(idx_a);
        let b = self.vertex(idx_b);
        CartCell::new(a, b)
    }
}