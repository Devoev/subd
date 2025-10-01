use crate::cells::cartesian::CartCellIdx;
use crate::index::dimensioned::{DimShape, MultiRange, Strides};
use crate::knots::breaks::Breaks;
use crate::mesh::traits::{MeshTopology, VertexStorage};
use crate::mesh::Mesh;
use itertools::Itertools;
use nalgebra::{Const, OPoint, Point, RealField, Scalar};
use std::iter::{zip, Map, Once};

/// `D`-variate cartesian product of [breakpoints](Breaks).
///
/// Given `d` breakpoint vectors `zi`, the cartesian product is defined as
/// `z = z1 × ... × zd`. The elements of that product are gridpoints in a cartesian mesh.
#[derive(Clone, Debug)]
pub struct MultiBreaks<T, const D: usize>([Breaks<T>; D]);

impl <T, const D: usize> MultiBreaks<T, D> {
    /// Returns the internal array of breaks in each parametric direction.
    pub fn as_breaks(&self) -> &[Breaks<T>; D] {
        &self.0
    }
}

impl <T: Scalar, const D: usize> VertexStorage<T> for MultiBreaks<T, D> {
    type GeoDim = Const<D>;
    type NodeIdx = [usize; D];
    type NodeIter = Once<[usize; D]>; // todo

    fn num_nodes(&self) -> usize {
        self.0.iter().map(|zeta| zeta.len()).product()
    }

    fn node_iter(&self) -> Self::NodeIter {
        todo!("Implement node using multi cartesian product");
    }

    fn vertex(&self, i: [usize; D]) -> OPoint<T, Self::GeoDim> {
        let coords = zip(i, self.0)
            .map(|(i, zeta)| zeta[i])
            .collect_array()
            .unwrap();
        Point::from(coords)
    }
}

/// Topology of a structured Cartesian grid.
pub struct Cartesian<const D: usize> {
    /// Shape of the structured nodes in each parametric directions.
    pub dim_shape: DimShape<D>, // todo: possibly only save element info instead of node info

    /// Strides for each parametric direction.
    pub strides: Strides<D>
}

impl <const D: usize> Cartesian<D> {
    /// Constructs a new [`Cartesian`] topology of the given `shape`.
    pub fn with_shape(shape: DimShape<D>) -> Self {
        let strides = Strides::from(shape);
        Cartesian { dim_shape: shape, strides }
    }
}

/// An iterator over the elements ([`CartCellIdx<D>`]) of a [`CartMesh`] mesh.
pub type CartCellIter<const D: usize> = Map<MultiRange<[usize; D]>, fn([usize; D]) -> CartCellIdx<D>>;

impl<const D: usize> MeshTopology for Cartesian<D> {
    type Cell = CartCellIdx<D>;
    type CellIter = CartCellIter<D>;

    fn num_cells(&self) -> usize {
        let mut dim_shape_elems = self.dim_shape;
        dim_shape_elems.shrink(1);
        dim_shape_elems.len()
    }

    fn into_cell_iter(self) -> Self::CellIter {
        let mut dim_shape_elems = self.dim_shape;
        dim_shape_elems.shrink(1);
        dim_shape_elems.multi_range().map(CartCellIdx)
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
        let breaks = MultiBreaks(breaks);
        let shape = DimShape::new_of_breaks(&breaks);
        CartMesh::with_coords_and_cells(breaks, Cartesian::with_shape(shape))
    }
}

impl<T: RealField, const D: usize> CartMesh<T, D> {
    /// Returns an iterator over all multi-indices in this grid.
    pub fn indices(&self) -> MultiRange<[usize; D]> {
        self.cells.dim_shape.multi_range()
    }
}