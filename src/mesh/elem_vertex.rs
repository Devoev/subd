//! Data structures for an element-to-vertex mesh.
//! In 2D the mesh is a [face-vertex mesh](https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes).

use crate::cells::node::NodeIdx;
use crate::cells::topo::{CellBoundary, CellConnectivity};
use crate::mesh::traits::{Mesh, MeshTopology};
use itertools::Itertools;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, Dim, DimName, DimNameDiff, DimNameSub, Dyn, OMatrix, Point, RealField, Scalar, U1};
use std::iter::once;
use std::slice::Iter;
use std::vec::IntoIter;

/// Vector of topological cells.
///
/// Contains the mesh connectivity information by
/// directly storing each (volumetric) element inside a [`Vec<C>`],
/// with each element pointing to its corner node indices.
pub struct ElemVec<C>(pub Vec<C>);

impl <C> MeshTopology for ElemVec<C> {
    type Elem = C;
    type ElemIter = IntoIter<Self::Elem>;
    
    fn num_elems(&self) -> usize {
        self.0.len()
    }

    fn into_elem_iter(self) -> Self::ElemIter {
        self.0.into_iter()
    }
}

impl <'a, C> MeshTopology for &'a ElemVec<C> {
    type Elem = &'a C;
    type ElemIter = Iter<'a, C>;

    fn num_elems(&self) -> usize {
        self.0.len()
    }

    fn into_elem_iter(self) -> Self::ElemIter {
        self.0.iter()
    }
}

/// Element-to-vertex mesh represented by vectors of elements and vertices each.
///
/// The coordinates are stored inside a [`Vec`] of [`Point<T,M>`]
/// and the topological cells inside an [`ElemVec<C>`].
pub type ElemVertexMesh<T, C, const M: usize> = Mesh<T, Vec<Point<T, M>>, ElemVec<C>>;

impl <T: Scalar, C: Clone, const M: usize> ElemVertexMesh<T, C, M> {
    /// Constructs a new [`ElemVertexMesh`] from the given `coords` and `elems`.
    pub fn new(coords: Vec<Point<T, M>>, elems: Vec<C>) -> Self {
        ElemVertexMesh::with_coords_and_cells(coords, ElemVec(elems))
    }
}

impl <T: RealField, C: CellConnectivity + Clone, const M: usize> ElemVertexMesh<T, C, M> {
    // todo: rename function and move to conversion trait of VertexStorage
    /// Constructs a new [`ElemVertexMesh`] from the given matrix `coords` of row-wise control points
    /// and `elems` connectivity.
    pub fn from_coords_matrix<N: Dim>(coords: OMatrix<T, N, Const<M>>, elems: Vec<C>) -> Self
        where DefaultAllocator: Allocator<N, Const<M>>
    {
        let coords = coords.row_iter()
            .map(|row| Point::from(row.transpose()))
            .collect();
        ElemVertexMesh::new(coords, elems)
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
        self.cells.0
            .iter()
            .filter(move |elem| elem.contains_node(node))
    }

    /// Finds all elements adjacent (aka. connected by a `K-1` dimensional sub-cell)
    /// to the given `elem` and returns them as an iterator.
    pub fn adjacent_elems<'a>(&'a self, elem: &'a C) -> impl Iterator<Item = &'a C> + 'a
    where C::Dim: DimNameSub<U1> + DimNameSub<DimNameDiff<C::Dim, U1>>
    {
        self.cells.0
            .iter()
            .filter(move |e| e.is_connected(elem, C::Dim::name().sub(U1)))
    }
}

impl <T: RealField, C: CellBoundary + Clone, const M: usize> ElemVertexMesh<T, C, M>
    where C::Dim: DimNameSub<U1> + DimNameSub<DimNameDiff<C::Dim, U1>>
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