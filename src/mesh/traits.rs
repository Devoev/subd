use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Dyn, OMatrix, OPoint, Scalar};
use std::marker::PhantomData;
use std::ops::Range;

/// Topology of a mesh consisting of cells.
///
/// A topological mesh is essentially a collection of cells, also called elements.
/// This trait mainly provides iteration over all mesh elements using [`Self::into_elem_iter`].
pub trait MeshTopology {
    /// Topological element in the mesh.
    type Elem; //: topo::Cell<Const<K>>; todo: add bound

    /// Element iterator.
    type ElemIter: Iterator<Item = Self::Elem>;

    /// Returns the total number of elements in this mesh.
    fn num_elems(&self) -> usize;

    /// Creates an iterator over all elements in this mesh.
    fn into_elem_iter(self) -> Self::ElemIter;
}

/// Storage for the geometrical vertex points of a mesh.
///
/// Each vertex point of a mesh is represented by an [`OPoint<T,Self::GeoDim>`]
/// with [`Self::GeoDim`] being the dimension of the embedding Euclidean space.
///
/// Access to individual vertices is given by the [`Self::vertex`] method,
/// given an index of type [`Self::NodeIdx`]. This is usually either a `usize`
/// or a multi-index.
/// Iteration over all vertices can be achieved
/// by the [`Self::node_iter`] and [`Self::num_nodes`] methods.
pub trait VertexStorage<T: Scalar> where DefaultAllocator: Allocator<Self::GeoDim> {
    /// Dimension of the embedding Euclidean space.
    type GeoDim: DimName;

    /// Node index defining a global ordering of vertices.
    type NodeIdx;

    /// Node iterator.
    type NodeIter: Iterator<Item = Self::NodeIdx>;

    /// Returns the total number of nodes in this mesh.
    fn num_nodes(&self) -> usize;

    /// Iterates over all nodes in this mesh in ascending order.
    fn node_iter(&self) -> Self::NodeIter;

    /// Gets the vertex point of the `i`-th node in the mesh.
    fn vertex(&self, i: Self::NodeIdx) -> OPoint<T, Self::GeoDim>; // todo: possibly also allow for multi index?
}

/// The node index of the vertex storage `Coords`.
pub type NodeIdx<T, Coords> = <Coords as VertexStorage<T>>::NodeIdx;

/// Vector of control points.
impl <T: Scalar, M: DimName> VertexStorage<T> for Vec<OPoint<T, M>>
where DefaultAllocator: Allocator<M>
{
    type GeoDim = M;
    type NodeIdx = usize;
    type NodeIter = Range<usize>;

    fn num_nodes(&self) -> usize {
        self.len()
    }

    fn node_iter(&self) -> Self::NodeIter {
        0..self.len()
    }

    fn vertex(&self, i: usize) -> OPoint<T, M> {
        self[i]
    }
}

/// Matrix of row-wise control points.
impl <T: Scalar, M: DimName> VertexStorage<T> for OMatrix<T, Dyn, M>
where DefaultAllocator: Allocator<M>
{
    type GeoDim = M;
    type NodeIdx = usize;
    type NodeIter = Range<usize>;

    fn num_nodes(&self) -> usize {
        self.nrows()
    }

    fn node_iter(&self) -> Self::NodeIter {
        0..self.nrows()
    }

    fn vertex(&self, i: usize) -> OPoint<T, Self::GeoDim> {
        OPoint::from(self.row(i))
    }
}

/// A mesh consisting of connected topological cells and vertex coordinates.
pub struct Mesh<T, Coords, Cells> {
    /// Coordinate storage.
    pub coords: Coords,

    /// Mesh cell topology.
    pub cells: Cells,

    _phantom_data: PhantomData<T>,
}

impl <T, Coords, Cells> Mesh<T, Coords, Cells>
where T: Scalar,
      Coords: VertexStorage<T>,
      Cells: MeshTopology,
      DefaultAllocator: Allocator<Coords::GeoDim>
{
    /// Constructs a new [`Mesh<T,Coords,Cells>`] with the given `coords` and `cells`.
    pub fn with_coords_and_cells(coords: Coords, cells: Cells) -> Self {
        Mesh { coords, cells, _phantom_data: PhantomData }
    }
}

// todo: update signature of elems and add more methods
//  maybe change lifetime parameter