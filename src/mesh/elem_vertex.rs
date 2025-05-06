//! Data structures for an element-to-vertex mesh.
//! In 2D the mesh is a [face-vertex mesh](https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes).

use crate::cells::cell::Cell;
use crate::cells::line_segment::LineSegment;
use crate::cells::quad::{Quad, QuadTopo};
use crate::cells::vertex::VertexTopo;
use crate::subd::patch::Patch;
use nalgebra::{Const, Point2, RealField};
use crate::mesh::elem_vertex_topo as topo;

/// Element-vertex mesh of topology [`ElementVertexTopo`]
/// with geometric data of the coordinates of each vertex.
pub struct ElementVertexMesh<const K: usize, T: RealField, C: Cell<Const<K>>> {
    /// Coordinates of the meshes vertices.
    pub coords: Vec<Point2<T>>,
    /// Topological connectivity of the elements.
    pub topology: topo::ElementVertex<K, C>,
}

/// A face-vertex mesh with `2`-dimensional faces [`C`].
pub type FaceVertexMesh<T, C> = ElementVertexMesh<2, T, C>;

/// A face-vertex mesh with quadrilateral faces.
pub type QuadVertexMesh<T> = FaceVertexMesh<T, QuadTopo>;

impl<T: RealField> QuadVertexMesh<T> {
    /// Returns the [`Point2`] of the given `node` index.
    pub fn coords(&self, node: VertexTopo) -> &Point2<T> {
        &self.coords[node.0]
    }

    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item=LineSegment<T>> + '_ {
        self.topology.edges().map(|edge_top| LineSegment::from_msh(edge_top, self))
    }

    /// Returns an iterator over all faces in this mesh.
    pub fn faces(&self) -> impl Iterator<Item=Quad<T>> + '_ {
        self.topology.elems.iter().map(|&face| Quad::from_msh(face, self))
    }

    /// Finds the patch of the regular or irregular `face`.
    pub fn find_patch(&self, face: QuadTopo) -> Patch<T> {
        todo!()
    }

    /// Returns an iterator over all patches in this mesh.
    pub fn patches(&self) -> impl Iterator<Item =Patch<T>> {
        self.topology.elems.iter().map(|&face| self.find_patch(face))
    }
}