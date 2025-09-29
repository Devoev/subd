//! Data structures for an element-to-vertex mesh.
//! In 2D the mesh is a [face-vertex mesh](https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes).

use crate::cells::node::NodeIdx;
use crate::cells::topo::{CellToNodes, CellBoundary};
use crate::mesh::traits::{Mesh, MeshTopology};
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, DimNameDiff, DimNameSub, Dyn, OMatrix, OPoint, Point, RealField, Scalar, U1};
use std::iter::{once, Map};
use std::marker::PhantomData;
use std::ops::Range;
use std::vec::IntoIter;

/// Element-to-vertex connectivity of a mesh.
///
/// Directly stores each (volumetric) element inside a [`Vec<C>`], 
/// with each element pointing to its corner node indices.
pub struct ElemToVertex<C> {
    /// Element connectivity vector.
    pub elems: Vec<C>
}

impl <C: Clone> MeshTopology for ElemToVertex<C> {
    type Elem = C;
    type ElemIter = IntoIter<Self::Elem>;
    
    fn num_elems(&self) -> usize {
        self.elems.len()
    }

    fn elem_iter(&self) -> Self::ElemIter {
        self.elems.clone().into_iter()
    }
}

/// Element-to-vertex mesh.
pub type ElemToVertexMesh<T, C, const M: usize> = Mesh<T, Vec<Point<T, M>>, ElemToVertex<C>>;

impl <T: Scalar, C: Clone, const M: usize> ElemToVertexMesh<T, C, M> {
    /// Constructs a new [`ElemVertexMesh`] from the given `coords` and `elems`.
    pub fn new(coords: Vec<Point<T, M>>, elems: Vec<C>) -> Self {
        ElemToVertexMesh::with_coords_and_cells(coords, ElemToVertex { elems })
    }
}

/// Element-vertex mesh, with topological connectivity information
/// of [`K`]-dimensional cells [`C`]
/// with geometric data of the coordinates of each [`M`]-dimensional vertex.
#[derive(Debug, Clone)]
pub struct ElemVertexMesh<T: RealField, C: CellToNodes<Const<K>>, const K: usize, const M: usize> {
    /// Coordinates of the meshes vertices.
    pub coords: Vec<Point<T, M>>,

    /// Element connectivity vector.
    pub elems: Vec<C>,
}

impl <T: RealField, C: CellToNodes<Const<K>>, const K: usize, const M: usize> ElemVertexMesh<T, C, K, M> {
    // todo: replace panic with result
    /// Constructs a new [`ElemVertexMesh`] from the given `coords` and `elems`.
    /// 
    /// # Panics
    /// If the number of points in the given `coords` vector 
    /// does not equal the number of nodes in the `topology`,
    /// the function will panic.
    pub fn new(coords: Vec<Point<T, M>>, elems: Vec<C>) -> Self {
        // todo: update this
        let num_nodes = elems.iter()
            .flat_map(C::nodes)
            .max()
            .expect("Elements vector `elems` must not be empty.")
            .0 + 1;
        assert_eq!(coords.len(), num_nodes,
                   "Length of `coords` (is {}) doesn't equal `num_nodes` (is {})", coords.len(), num_nodes);
        ElemVertexMesh { coords, elems }
    }

    // todo: rename function
    /// Constructs a new [`ElemVertexMesh`] from the given matrix `coords` of row-wise control points
    /// and `elems` connectivity.
    pub fn from_matrix<N: Dim>(coords: OMatrix<T, N, Const<M>>, elems: Vec<C>) -> Self
        where DefaultAllocator: Allocator<N, Const<M>>
    {
        let coords = coords.row_iter()
            .map(|row| Point::from(row.transpose()))
            .collect();
        ElemVertexMesh::new(coords, elems)
    }

    /// Returns the [`Point`] of the given `node` index.
    pub fn coords(&self, node: NodeIdx) -> &Point<T, M> {
        &self.coords[node.0]
    }

    /// Constructs the matrix of control points.
    pub fn coords_matrix(&self) -> OMatrix<T, Dyn, Const<M>> {
        let rows = self.coords
            .iter()
            .map(|point| point.coords.transpose())
            .collect_vec();
        OMatrix::from_rows(&rows)
    }

    /// Finds all elements which contain the given `node` and returns them as an iterator.
    pub fn elems_of_node(&self, node: NodeIdx) -> impl Iterator<Item = &C> {
        self.elems
            .iter()
            .filter(move |elem| elem.contains_node(node))
    }

    /// Finds all elements adjacent (aka. connected by a `K-1` dimensional sub-cell)
    /// to the given `elem` and returns them as an iterator.
    pub fn adjacent_elems<'a>(&'a self, elem: &'a C) -> impl Iterator<Item = &'a C> + 'a
    where Const<K>: DimNameSub<U1> + DimNameSub<DimNameDiff<Const<K>, U1>>
    {
        self.elems
            .iter()
            .filter(move |e| e.is_connected(elem, Const::<K>.sub(U1)))
    }
}

impl <T: RealField, C: CellBoundary<Const<K>>, const K: usize, const M: usize> ElemVertexMesh<T, C, K, M>
    where Const<K>: DimNameSub<U1> + DimNameSub<DimNameDiff<Const<K>, U1>>
{
    /// Returns `true` if given `elem` is at the boundary of the mesh,
    /// i.e. it has less than [`C::NUM_SUB_CELLS`] adjacent elements.
    pub fn is_boundary_elem(&self, elem: &C) -> bool {
        self.adjacent_elems(elem).count() < C::NUM_SUB_CELLS
    }

    // todo: is_boundary_node_general is inefficient. Update this by not calling is_boundary_elem ?
    // fixme: it is INCORRECT. Consider the case of width 2. The nodes in between the 2 elements
    //  would be considered boundary nodes by this function.
    /// Returns `true` if the given `node` is a boundary node,
    /// i.e. all elements containing the node are boundary elements.
    fn is_boundary_node_general(&self, node: NodeIdx) -> bool {
        self.elems_of_node(node).all(|elem| self.is_boundary_elem(elem))
    }

    /// Returns an iterator over all boundary nodes in this mesh.
    pub fn boundary_nodes(&self) -> impl Iterator<Item =NodeIdx> + '_ {
        // self.node_iter().filter(|&n| self.is_boundary_node_general(n))
        todo!();
        once(NodeIdx(0))
    }

    // todo: add info about regular/ irregular adjacency
}

/// An iterator that yields the [nodes indices](NodeIdx) of an element-vertex mesh.
pub type NodesIter = Map<Range<usize>, fn(usize) -> NodeIdx>;

/// An iterator that yields the elements of an element-vertex mesh.
pub type ElemsIter<'a, C> = std::slice::Iter<'a, C>;