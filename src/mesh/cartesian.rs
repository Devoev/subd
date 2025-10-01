use crate::cells::cartesian::CartCellIdx;
use crate::index::dimensioned::{DimShape, MultiRange, Strides};
use crate::knots::breaks::Breaks;
use crate::mesh::traits::{MeshTopology, VertexStorage};
use crate::mesh::Mesh;
use itertools::Itertools;
use nalgebra::{Const, OPoint, Point, RealField, Scalar};
use std::iter::{zip, Map};

/// `D`-variate cartesian product of [breakpoints](Breaks).
///
/// Given `d` breakpoint vectors `zi`, the cartesian product is defined as
/// `z = z1 × ... × zd`. The elements of that product are gridpoints in a cartesian mesh.
#[derive(Clone, Debug)]
pub struct MultiBreaks<T, const D: usize> {
    /// Breaks in each parametric direction.
    breaks: [Breaks<T>; D],

    /// Shape of the structured gridpoints in each parametric direction.
    nodes_shape: DimShape<D>
}

impl <T, const D: usize> MultiBreaks<T, D> {
    /// Constructs new `MultiBreaks` from the given `breaks` in each parametric direction.
    pub fn new(breaks: [Breaks<T>; D]) -> Self {
        let nodes_shape = DimShape::new_of_breaks(&breaks);
        MultiBreaks { breaks, nodes_shape }
    }

    /// Returns the internal array of breaks in each parametric direction.
    pub fn as_breaks(&self) -> &[Breaks<T>; D] {
        &self.breaks
    }
}

impl <T: Scalar + Copy, const D: usize> VertexStorage<T> for MultiBreaks<T, D> {
    type GeoDim = Const<D>;
    type NodeIdx = [usize; D];
    type NodeIter = MultiRange<[usize; D]>;

    fn len(&self) -> usize {
        self.nodes_shape.len()
    }

    fn node_iter(&self) -> Self::NodeIter {
        self.nodes_shape.multi_range()
    }

    fn vertex(&self, i: [usize; D]) -> OPoint<T, Self::GeoDim> {
        let coords = zip(i, &self.breaks)
            .map(|(i, zeta)| zeta[i])
            .collect_array()
            .unwrap();
        Point::from(coords)
    }
}

/// Topology of a structured Cartesian grid.
#[derive(Clone, Debug)]
pub struct Cartesian<const D: usize> {
    /// Shape of the cartesian cells in each parametric directions.
    pub cells_shape: DimShape<D>,

    /// Strides for each parametric direction.
    pub strides: Strides<D>
}

impl <const D: usize> Cartesian<D> {
    /// Constructs a new [`Cartesian`] topology of the given `nodes_shape`.
    pub fn new(mut nodes_shape: DimShape<D>) -> Self {
        // There are exactly one less cells than nodes in each direction
        nodes_shape.shrink(1);
        let strides = Strides::from(nodes_shape);
        Cartesian { cells_shape: nodes_shape, strides }
    }
}

/// An iterator over the elements ([`CartCellIdx<D>`]) of a [`CartMesh`] mesh.
pub type CartCellIter<const D: usize> = Map<MultiRange<[usize; D]>, fn([usize; D]) -> CartCellIdx<D>>;

impl <'a, const D: usize> MeshTopology for &'a Cartesian<D> {
    type Cell = CartCellIdx<D>;
    type CellIter = CartCellIter<D>;

    fn len(&self) -> usize {
        self.cells_shape.len()
    }

    fn into_cell_iter(self) -> Self::CellIter {
        self.cells_shape.multi_range().map(CartCellIdx)
    }
}

impl<const D: usize> MeshTopology for Cartesian<D> {
    type Cell = CartCellIdx<D>;
    type CellIter = CartCellIter<D>;

    fn len(&self) -> usize {
        (&self).len()
    }

    fn into_cell_iter(self) -> Self::CellIter {
        (&self).into_cell_iter()
    }
}

/// Cartesian mesh built by [tensor product breaks](MultiBreaks).
///
/// The grid formed by the mesh nodes can in 2D be schematically visualized as
/// ```text
///        ^
///        |
/// by[ny]---   +---+---+---+
///        |    |   |   |   |
///        |    +---+---+---+
///        |    |   |   |   |
///        |    +---+---+---+
///        |    |   |   |   |
///  by[0]---   +---+---+---+
///        |
///        +----|-----------|--->
///           bx[0]       bx[nx]
/// ```
/// where `bx` and `by` are the breakpoints for the `x` and `y` direction respectively.
pub type CartMesh<T, const D: usize> = Mesh<T, MultiBreaks<T, D>, Cartesian<D>>;

impl<T: RealField + Copy, const D: usize> CartMesh<T, D> {
    /// Constructs a new [`CartMesh`] from the given `breaks`.
    ///
    /// The topological information for the shape and strides is constructed from the shape of the breaks.
    pub fn with_breaks(breaks: [Breaks<T>; D]) -> Self {
        let breaks = MultiBreaks::new(breaks);
        let cartesian_topology = Cartesian::new(breaks.nodes_shape);
        CartMesh::with_coords_and_cells(breaks, cartesian_topology)
    }
}

impl<T: RealField, const D: usize> CartMesh<T, D> {
    /// Returns an iterator over all multi-indices in this grid.
    pub fn indices(&self) -> MultiRange<[usize; D]> {
        self.cells.cells_shape.multi_range()
    }
}