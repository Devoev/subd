//! Mesh data structures.

use std::marker::PhantomData;
use nalgebra::{DefaultAllocator, Scalar};
use nalgebra::allocator::Allocator;
use crate::mesh::traits::{MeshTopology, VertexStorage};

pub mod elem_vertex;
pub mod cartesian;
pub mod bezier;
pub mod traits;
pub mod face_vertex;
pub mod incidence;
pub mod knot_mesh;

/// The most generic mesh type.
///
/// A mesh consists of connected topological cells [`Cells`], defining the partition of the domain
/// and vertex coordinates [`Coords`] defining the embedding into Euclidean space.
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

    /// Consumes `self` and returns an iterator over all topological cells in this mesh.
    pub fn into_cell_iter(self) -> Cells::CellIter {
        self.cells.into_cell_iter()
    }
}

impl <'a, T, Coords, Cells: 'a> Mesh<T, Coords, Cells>
where T: Scalar,
      Coords: VertexStorage<T>,
      &'a Cells: MeshTopology,
      DefaultAllocator: Allocator<Coords::GeoDim>
{
    /// Returns an iterator over references to topological cells in this mesh.
    pub fn cell_iter(&self) -> <&'a Cells as MeshTopology>::CellIter {
        self.cells.into_cell_iter()
    }
}