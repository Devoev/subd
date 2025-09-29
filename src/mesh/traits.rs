use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, DimName, Dyn, OMatrix, OPoint, Scalar};
use std::marker::PhantomData;
use std::ops::Range;

/// Topology of a mesh consisting of cells.
pub trait MeshTopology {
    /// Topological element in the mesh.
    type Elem; //: topo::Cell<Const<K>>; todo: add bound

    /// Element iterator.
    type ElemIter: Iterator<Item = Self::Elem>;

    /// Returns the total number of elements in this mesh.
    fn num_elems(&self) -> usize;

    /// Returns an iterator over all elements in this mesh.
    fn elem_iter(&self) -> Self::ElemIter;
}

/// Storage for the geometrical vertex points of a mesh.
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
pub struct Mesh<T: Scalar, Coords: VertexStorage<T>, Cells: MeshTopology>
    where DefaultAllocator: Allocator<Coords::GeoDim>
{
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