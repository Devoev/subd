//! Data structures for an element-to-vertex mesh.
//! In 2D the mesh is a [face-vertex mesh](https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes).

use crate::cells::line_segment::LineSegment;
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadTopo};
use crate::cells::topo::Cell;
use crate::mesh::elem_vertex_topo as topo;
use crate::subd_legacy::patch::Patch;
use nalgebra::{Const, Point, RealField};

/// Element-vertex mesh of [`K`]-dimensional topology [`ElementVertexTopo`]
/// with geometric data of the coordinates of each [`M`]-dimensional vertex.
pub struct ElementVertexMesh<T: RealField, C: Cell<Const<K>>, const K: usize, const M: usize> {
    /// Coordinates of the meshes vertices.
    pub coords: Vec<Point<T, M>>,
    /// Topological connectivity of the elements.
    pub topology: topo::ElementVertex<K, C>,
}

/// A face-vertex mesh with `2`-dimensional faces [`C`].
pub type FaceVertexMesh<T, C, const M: usize> = ElementVertexMesh<T, C, 2, M>;

/// A face-vertex mesh with quadrilateral faces.
pub type QuadVertexMesh<T, const M: usize> = FaceVertexMesh<T, QuadTopo, M>;

impl <T: RealField, C: Cell<Const<K>>, const K: usize, const M: usize> ElementVertexMesh<T, C, K, M> {
    // todo: replace panic with result
    /// Constructs a new [`ElementVertexMesh`] from the given `coords` and `topology`.
    /// 
    /// # Panics
    /// If the number of points in the given `coords` vector 
    /// does not equal the number of nodes in the `topology`,
    /// the function will panic.
    pub fn new(coords: Vec<Point<T, M>>, topology: topo::ElementVertex<K, C>) -> Self {
        assert_eq!(coords.len(), topology.num_nodes, 
                   "Length of `coords` (is {}) doesn't equal `num_nodes` (is {})", coords.len(), topology.num_nodes);
        ElementVertexMesh { coords, topology }
    }

    /// Returns the [`Point`] of the given `node` index.
    pub fn coords(&self, node: NodeIdx) -> &Point<T, M> {
        &self.coords[node.0]
    }
}

impl<T: RealField, const M: usize> QuadVertexMesh<T, M> {
    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item=LineSegment<T, M>> + '_ {
        self.topology.edges().map(|edge_top| LineSegment::from_msh(edge_top, self))
    }

    /// Returns an iterator over all faces in this mesh.
    pub fn faces(&self) -> impl Iterator<Item=Quad<T, M>> + '_ {
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