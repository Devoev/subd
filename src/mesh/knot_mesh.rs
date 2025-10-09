use crate::index::dimensioned::DimShape;
use crate::knots::breaks_with_multiplicity::BreaksWithMultiplicity;
use crate::knots::increments::Increments;
use crate::knots::knot_span::{KnotSpan, MultiKnotSpan};
use crate::knots::knot_vec::KnotVec;
use crate::mesh::cartesian::Cartesian;
use crate::mesh::cell_topology::{CellTopology};
use crate::mesh::Mesh;
use itertools::{Itertools, MultiProduct};
use nalgebra::{Const, OPoint, RealField, Scalar};
use std::iter::{zip, Map, Once};
use std::vec::IntoIter;
use crate::knots::breaks::Breaks;
use crate::mesh::vertex_storage::VertexStorage;

/// `D`-variate cartesian product of [knot vectors](KnotVec).
///
/// Similar to `MultiBreaks` but breakpoints are replaced with knot vectors.
#[derive(Debug, Clone)]
pub struct MultiKnotVec<T, const D: usize>([KnotVec<T>; D]); // todo: add nodes_shape, similar to MultiBreaks

impl <T: Scalar, const D: usize> MultiKnotVec<T, D> {
    /// Constructs a new [`MultiKnotVec`] with the given `knots` for each parametric direction.
    pub fn from_knots(knots: [KnotVec<T>; D]) -> Self {
        MultiKnotVec(knots)
    }

    /// Returns the internal array of knot vectors in each parametric direction.
    pub fn as_knots(&self) -> &[KnotVec<T>; D] {
        &self.0
    }
}

impl <T: Scalar + Copy, const D: usize> VertexStorage<T> for MultiKnotVec<T, D> {
    type GeoDim = Const<D>;
    type NodeIdx = [usize; D];
    type NodeIter = Once<[usize; D]>;

    fn len(&self) -> usize {
        self.0.iter().map(|xi| xi.len()).product()
    }

    fn is_empty(&self) -> bool {
        self.0.iter().any(|xi| xi.is_empty())
    }

    fn node_iter(&self) -> Self::NodeIter {
        todo!("Implement using multi cartesian product")
    }

    fn vertex(&self, i: Self::NodeIdx) -> OPoint<T, Self::GeoDim> {
        zip(i, &self.0)
            .map(|(i, xi)| xi[i])
            .collect_array::<D>()
            .unwrap()
            .into()
    }
}

/// Topology of a structured Cartesian grid with knot increments.
#[derive(Debug, Clone)]
pub struct CartesianWithIncrements<const D: usize> {
    /// Topology of the structured Cartesian grid without duplicate knots.
    pub cartesian: Cartesian<D>,

    /// Breakpoint index increments for each parametric direction.
    pub increments: [Increments; D], // todo: replace this with span indices
}

impl <const D: usize> CartesianWithIncrements<D> {
    /// Constructs a new `CartesianWithIncrements` with the given `shape` and `increments`.
    pub fn with_shape_and_increments(shape: DimShape<D>, increments: [Increments; D]) -> Self {
        CartesianWithIncrements { cartesian: Cartesian::new(shape), increments }
    }
}

/// An iterator over the elements (`[KnotSpan;D]`) of a [`KnotMesh`] mesh.
pub type KnotSpanIter<const D: usize> = Map<MultiProduct<IntoIter<KnotSpan>>, fn(Vec<KnotSpan>) -> MultiKnotSpan<D>>;

impl <const D: usize> CellTopology for CartesianWithIncrements<D> {
    type Cell = MultiKnotSpan<D>;
    type BorrowCell<'a> = MultiKnotSpan<D>;
    type CellIter<'a> = KnotSpanIter<D>;

    fn len(&self) -> usize {
        self.cartesian.len()
    }

    fn is_empty(&self) -> bool {
        self.cartesian.is_empty()
    }

    fn cell_iter(&self) -> Self::CellIter<'_> {
        self.increments.iter()
            .map(|increments| increments.span_indices().collect_vec())
            .multi_cartesian_product()
            .map(|vec| MultiKnotSpan(vec.into_iter().collect_array().unwrap()))
    }
}

/// Cartesian mesh built by tensor product of [`KnotVec<T>`].
///
/// Same as a Cartesian mesh but with multiplicities of breakpoints.
pub type KnotMesh<T, const D: usize> = Mesh<T, MultiKnotVec<T, D>, CartesianWithIncrements<D>>;

impl <T: RealField + Copy, const D: usize> KnotMesh<T, D> {
    // todo: this implementation is ugly and inefficient. Probably change this.
    /// Constructs a new [`KnotMesh`] from the given `knots`.
    /// The breaks and knot multiplicities are computed using [`BreaksWithMultiplicity::from_knots`].
    pub fn from_knots(knots: [KnotVec<T>; D]) -> Self {
        // Compute breaks with multiplicities
        let breaks = knots
            .clone()
            .map(BreaksWithMultiplicity::from_knots);

        // Compute shape of breaks
        let shape = DimShape::from_breaks(breaks
            .clone()
            .map(Breaks::from)
        );

        // Compute increments
        let increments = breaks
            .map(Increments::from_multiplicities);

        KnotMesh::from_coords_and_cells(MultiKnotVec(knots), CartesianWithIncrements::with_shape_and_increments(shape, increments))
    }
}