//! Mesh data structures.

use std::marker::PhantomData;
use nalgebra::{DefaultAllocator, Scalar};
use nalgebra::allocator::Allocator;
use crate::cells::traits::{ElemOfCell, ToElement};
use crate::element::traits::ElemAllocator;
use crate::mesh::cell_topology::{ElementTopology, CellTopology};
use crate::mesh::vertex_storage::VertexStorage;

pub mod elem_vertex;
pub mod cartesian;
pub mod bezier;
pub mod cell_topology;
pub mod face_vertex;
pub mod incidence;
pub mod knot_mesh;
pub mod vertex_storage;

/// The most generic mesh type.
///
/// A mesh consists of connected topological cells [`Cells`], defining the partition of the domain
/// and vertex coordinates [`Coords`] defining the embedding into Euclidean space.
#[derive(Copy, Clone, Debug)]
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
      Cells: CellTopology,
      DefaultAllocator: Allocator<Coords::GeoDim>
{
    /// Constructs a new [`Mesh<T,Coords,Cells>`] with the given `coords` and `cells`.
    pub fn with_coords_and_cells(coords: Coords, cells: Cells) -> Self {
        Mesh { coords, cells, _phantom_data: PhantomData }
    }

    /// Returns the total number of elements or cells in `self`.
    pub fn num_elems(&self) -> usize {
        self.cells.len()
    }

    /// Returns the total number of nodes in `self`.
    pub fn num_nodes(&self) -> usize {
        self.coords.len()
    }

    /// Consumes `self` and returns an iterator over all topological cells in this mesh.
    pub fn into_cell_iter(self) -> Cells::CellIter {
        self.cells.into_cell_iter()
    }
}

impl <'a, T, Coords, Cells> Mesh<T, Coords, Cells>
where T: Scalar,
      Coords: VertexStorage<T>,
      &'a Cells: 'a + CellTopology,
      DefaultAllocator: Allocator<Coords::GeoDim>
{
    /// Returns an iterator over all topological cells in this mesh.
    pub fn cell_iter(&'a self) -> <&'a Cells as CellTopology>::CellIter {
        self.cells.into_cell_iter()
    }
}

// todo: introduce Iterator types IntoElemIter, IntoElemCellIter...

/// The geometrical element of the [`Mesh<T,Coords,Cells>`].
pub type ElemOfMesh<T, Coords, Cells> = ElemOfCell<T, <Cells as CellTopology>::Cell, <Coords as VertexStorage<T>>::GeoDim>;

impl <T, Coords, Cells> Mesh<T, Coords, Cells>
where T: Scalar,
      Coords: VertexStorage<T>,
      Cells: ElementTopology<T, Coords>,
      DefaultAllocator: Allocator<Coords::GeoDim> + ElemAllocator<T, ElemOfMesh<T, Coords, Cells>>
{
    /// Consumes `self` and returns an iterator over all geometrical elements in this mesh.
    pub fn into_elem_iter(self) -> impl Iterator<Item = ElemOfMesh<T, Coords, Cells>> {
        self.cells.into_cell_iter().map(move |cell| cell.to_element(&self.coords))
    }

    /// Consumes `self` and returns an iterator over `(elem,cell)` pairs.
    pub fn into_elem_cell_iter(self) -> impl Iterator<Item = (ElemOfMesh<T, Coords, Cells>, Cells::Cell)>  {
        self.cells.into_cell_iter().map(move |cell| (cell.to_element(&self.coords), cell))
    }
}

impl <'a, T, Coords, Cells> Mesh<T, Coords, Cells>
where T: Scalar,
      Coords: VertexStorage<T>,
      &'a Cells: 'a + ElementTopology<T, Coords>,
      DefaultAllocator: Allocator<Coords::GeoDim> + ElemAllocator<T, ElemOfMesh<T, Coords, &'a Cells>>
{
    /// Returns an iterator over all geometrical elements in this mesh.
    pub fn elem_iter(&'a self) -> impl Iterator<Item = ElemOfMesh<T, Coords, &'a Cells>> {
        self.cell_iter().map(move |cell| cell.to_element(&self.coords))
    }

    /// Returns an iterator over `(elem,cell)` pairs.
    pub fn elem_cell_iter(&'a self) -> impl Iterator<Item = (ElemOfMesh<T, Coords, &'a Cells>, <&'a Cells as CellTopology>::Cell)>  {
        self.cell_iter().map(move |cell| (cell.to_element(&self.coords), cell))
    }
}