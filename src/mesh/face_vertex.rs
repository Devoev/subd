//! Data structures for a [face-vertex mesh](https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes).

use crate::mesh::cell::{CellBoundaryTopo, CellTopo};
use crate::mesh::line_segment::{LineSegment, LineSegmentTopo};
use crate::mesh::quad::{Quad, QuadTopo};
use crate::subd::patch::Patch;
use itertools::Itertools;
use nalgebra::{DimName, DimNameSub, Point2, RealField, U1, U2};
use std::marker::PhantomData;
use crate::mesh::chain::ChainTopo;
use crate::mesh::vertex::VertexTopo;

/// Topology of [`K`]-dimensional element-vertex mesh.
/// The elements are `K`-cells of type [`C`].
pub struct ElementVertexTopo<K: DimName, C: CellTopo<K>> {
    /// Element connectivity vector.
    pub elems: Vec<C>,

    _phantoms: PhantomData<K>,
}

impl <K: DimName, C: CellTopo<K>> ElementVertexTopo<K, C> {
    
    /// Constructs a new [`ElementVertexTopo`] from the given `elems` topology vector.
    pub fn new(elems: Vec<C>) -> Self {
        ElementVertexTopo { elems, _phantoms: PhantomData }
    }

    /// Finds all elements which contain the given `node` and returns them as an iterator.
    pub fn elems_of_node(&self, node: VertexTopo) -> impl Iterator<Item = &C> {
        self.elems
            .iter()
            .filter(move |elem| elem.contains_node(node))
    }
    
    // todo: move methods for adjacency computation here
    
    // todo: move methods for boundary computation here (add chains for that)
}

/// A face-vertex mesh topology with `2`-dimensional faces [`C`].
type FaceVertexTopo<C> = ElementVertexTopo<U2, C>;

// todo: add sub-traits for edges faces and vertices
impl <E: CellTopo<U1> + Clone, F: CellBoundaryTopo<U2, BoundaryCell=E>> FaceVertexTopo<F> {

    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges_(&self) -> impl Iterator<Item = E> + '_ {
        self.elems.iter()
            .map(|face| face.boundary().cells()[0].clone())
    }
    
    // todo: move edges and valence methods here
}

/// A face-vertex mesh topology of quadrilateral faces.
type QuadVertexTopo = FaceVertexTopo<QuadTopo>;

impl QuadVertexTopo {
    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item = LineSegmentTopo> + '_ {
        self.elems.iter()
            .flat_map(|&face| face.edges())
            .map(|edge| edge.sorted())
            .unique()
    }

    /// Returns all edges connected to the given `node`.
    pub fn edges_of_node(&self, node: VertexTopo) -> impl Iterator<Item = LineSegmentTopo> + '_ {
        self.edges().filter(move |edge| edge.0.contains(&node))
    }

    /// Calculates the valence of the given `node`, i.e. the number of edges connected to the node.
    pub fn valence(&self, node: VertexTopo) -> usize {
        self.edges_of_node(node).count()
    }

    /// Returns `true` if the face is regular.
    pub fn is_regular(&self, face: QuadTopo) -> bool {
        face.nodes().iter().all(|node| self.valence(*node) == 4)
    }

    /// Finds the irregular node of the given `face`, if any exists.
    pub fn irregular_node_of_face(&self, face: QuadTopo) -> Option<VertexTopo> {
        face.nodes()
            .into_iter()
            .find(|&v| self.valence(v) != 4)
    }

    /// Returns all adjacent faces to `face`.
    pub fn adjacent_faces(&self, face: QuadTopo) -> impl Iterator<Item = (usize, &QuadTopo)> {
        self.elems
            .iter()
            .enumerate()
            .filter(move |(_, f)| f.is_adjacent(face))
    }

    /// Returns whether the given `face` is a boundary face, i.e. it has less than `4` adjacent faces.
    pub fn is_boundary_face(&self, face: QuadTopo) -> bool {
        self.adjacent_faces(face).count() < 4
    }

    /// Returns whether the given `node` is a boundary node,
    /// i.e. all faces containing the node are boundary faces.
    pub fn is_boundary_node(&self, node: VertexTopo) -> bool {
        self.elems_of_node(node).all(|&f| self.is_boundary_face(f))
    }

    /// Finds all boundary nodes of the given `face`,
    /// i.e. all irregular nodes, assuming the face is a boundary face.
    pub fn boundary_nodes_of_face(&self, face: QuadTopo) -> Vec<VertexTopo> {
        face.nodes().into_iter().filter(|&v| self.valence(v) != 4).collect()
    }
}

/// Element-vertex mesh of topology [`ElementVertexTopo`]
/// with geometric data of the coordinates of each vertex.
pub struct ElementVertexMesh<T: RealField, K: DimName + DimNameSub<U1>, C: CellTopo<K>> {
    /// Coordinates of the meshes vertices.
    pub coords: Vec<Point2<T>>,
    /// Topological connectivity of the elements.
    pub topology: ElementVertexTopo<K, C>,
}

/// A face-vertex mesh with `2`-dimensional faces [`C`].
pub type FaceVertexMesh<T, C> = ElementVertexMesh<T, U2, C>;

/// A face-vertex mesh with quadrilateral faces.
pub type QuadVertexMesh<T> = FaceVertexMesh<T, QuadTopo>;

impl<T: RealField> QuadVertexMesh<T> {
    /// Returns the [`Point2`] of the given `node` index.
    pub fn coords(&self, node: VertexTopo) -> &Point2<T> {
        &self.coords[node.0]
    }

    // todo: possibly move this method to topology
    /// Returns the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.coords.len()
    }

    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item=LineSegment<T>> + '_ {
        self.topology.edges().map(|edge_top| LineSegment::from_msh(edge_top, self))
    }

    /// Returns an iterator over all faces in this mesh.
    pub fn faces(&self) -> impl Iterator<Item=Quad<T>> + '_ {
        self.topology.elems.iter().map(|&face| Quad::from_msh(face, self))
    }
    
    // todo: possibly move this method to topology
    /// Returns an iterator over the indices of all boundary nodes in this mesh.
    pub fn boundary_nodes(&self) -> impl Iterator<Item = VertexTopo> + '_ {
        (0..self.num_nodes())
            .map(VertexTopo)
            .filter(|&n| self.topology.is_boundary_node(n))
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