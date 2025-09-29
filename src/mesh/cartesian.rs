use std::io::empty;
use crate::cells::cartesian::{CartCell, CartCellIdx};
use crate::index::dimensioned::{DimShape, MultiRange, Strides};
use itertools::Itertools;
use nalgebra::{Const, OPoint, Point, RealField, Scalar};
use std::iter::{once, zip, Map, Once};
use crate::cells::node::NodeIdx;
use crate::index::multi_index::MultiIndex;
use crate::knots::breaks::Breaks;
use crate::mesh::traits::{Mesh, MeshTopology, VertexStorage};

impl <T: Scalar, const D: usize> VertexStorage<T> for [Breaks<T>; D] {
    type GeoDim = Const<D>;
    type NodeIter = Once<NodeIdx>; // todo

    fn num_nodes(&self) -> usize {
        todo!()
    }

    fn node_iter(&self) -> Self::NodeIter {
        todo!();
        once(NodeIdx(0))
    }

    fn vertex(&self, i: NodeIdx) -> OPoint<T, Self::GeoDim> {
        // todo: because i is a linear index, it has to first be turned into a multi index
        //  solution => use a multi index directly and encode that in the VertexStorage trait
        todo!()
    }
}

/// Cartesian mesh built by tensor product of [`Breaks<T>`].
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
pub struct CartMesh<T: RealField, const D: usize> {
    /// Breakpoints for each parametric direction.
    pub breaks: [Breaks<T>; D],

    /// Shape of the parametric directions.
    pub dim_shape: DimShape<D>,

    /// Strides for each parametric direction.
    pub strides: Strides<D>
}

impl<T: RealField + Copy, const D: usize> CartMesh<T, D> {
    /// Constructs a new [`CartMesh`] from the given `breaks`, `dim_shape` and `strides`.
    ///
    /// # Panics
    /// If the shape of the `breaks` does not equal the shape of `dim_shape`,
    /// the function will panic.
    pub fn new(breaks: [Breaks<T>; D], dim_shape: DimShape<D>, strides: Strides<D>) -> Self {
        let n_breaks = breaks.iter().map(|b| b.len()).collect_vec();
        let n_shape = dim_shape.0;
        assert_eq!(n_breaks, n_shape,
                   "Shape of `breaks` (is {:?}) doesn't equal shape of `dim_shape` (is {:?})",
                   n_breaks, n_shape);
        CartMesh { breaks, dim_shape, strides }
    }

    /// Constructs a new [`CartMesh`] from the given `breaks`.
    /// The topological information for the shape and strides is constructed from the shape of the breaks.
    pub fn from_breaks(breaks: [Breaks<T>; D]) -> Self {
        let shape = breaks.iter().map(|b| b.len()).collect_array().unwrap();
        let dim_shape = DimShape(shape);
        CartMesh { breaks, dim_shape, strides: Strides::from(dim_shape) }
    }
    
    /// Constructs the vertex point at the given multi-index position `idx`.
    pub fn vertex(&self, idx: [usize; D]) -> Point<T, D> {
        zip(idx, &self.breaks)
            .map(|(i, breaks)| breaks[i])
            .collect_array::<D>()
            .unwrap()
            .into()
    }
}

/// An iterator over the linear nodes ([`NodeIdx`]) of a [`CartMesh`] mesh.
pub struct NodesIter<'a, const D: usize> {
    iter: MultiRange<[usize; D]>,
    strides: &'a Strides<D>
}

impl <'a, const D: usize> NodesIter<'a, D> {
    /// Constructs a new [`NodesIter`] from the given `iter` and `strides`.
    pub fn new(iter: MultiRange<[usize; D]>, strides: &'a Strides<D>) -> Self {
        NodesIter { iter, strides }
    }

    /// Constructs a enw [`NodesIter`] from the given cartesian `msh`.
    pub fn from_msh<T: RealField>(msh: &'a CartMesh<T, D>) -> Self {
        NodesIter::new(msh.indices(), &msh.strides)
    }
}

// todo: this implementation doesn't make much sense. Either just return the multi-indices,
//  or iterate over linear indices directly. Flattening the multi-indices is way to expensive

impl<const D: usize> Iterator for NodesIter<'_, D> {
    type Item = NodeIdx;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|idx| NodeIdx(idx.into_lin(self.strides)))
    }
}

/// An iterator over the elements ([`CartCellIdx<D>`]) of a [`CartMesh`] mesh.
pub type ElemsIter<const D: usize> = Map<MultiRange<[usize; D]>, fn([usize; D]) -> CartCellIdx<D>>;

impl<T: RealField, const D: usize> CartMesh<T, D> {
    /// Returns an iterator over all multi-indices in this grid.
    pub fn indices(&self) -> MultiRange<[usize; D]> {
        self.dim_shape.multi_range()
    }

    /// Returns an iterator over all nodes with linear indices in increasing index order.
    pub fn nodes(&self) -> NodesIter<'_, D> {
        NodesIter::from_msh(self)
    }

    /// Returns an iterator over all elements in lexicographical order.
    pub fn elems(&self) -> ElemsIter<D> {
        let mut dim_shape_elems = self.dim_shape;
        dim_shape_elems.shrink(1);
        dim_shape_elems.multi_range().map(CartCellIdx)
    }
}

impl<T: RealField + Copy, const K: usize> MeshTopology for CartMesh<T, K> {
    type Elem = CartCellIdx<K>;
    type ElemIter = ElemsIter<K>;

    fn num_elems(&self) -> usize {
        let mut dim_shape_elems = self.dim_shape;
        dim_shape_elems.shrink(1);
        dim_shape_elems.len()
    }

    fn elem_iter(&self) -> Self::ElemIter {
        self.elems()
    }
}