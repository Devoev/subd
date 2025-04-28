use crate::subd::edge::sort_edge;
use crate::subd::face::{edges_of_face, are_adjacent};
use crate::subd::patch::{Patch};
use itertools::Itertools;
use nalgebra::{
    Point2, RealField, Vector2,
};
use std::ops::{Deref, DerefMut};

pub type Node = usize;
pub type Edge = [Node; 2];
pub type Face = [Node; 4];

/// Topological mesh of quadrilateral elements.
#[derive(Debug, Clone, Default)]
pub struct TopologicalMesh {
    /// Face connectivity vector.
    pub faces: Vec<Face>,
}

impl TopologicalMesh {

    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.faces.iter()
            .flat_map(|&face| edges_of_face(face))
            .map(sort_edge)
            .unique()
    }

    /// Returns all edges connected to the given `node`.
    pub fn edges_of_node(&self, node: Node) -> impl Iterator<Item = Edge> + '_ {
        self.edges().filter(move |edge| edge.contains(&node))
    }

    /// Calculates the valence of the given `node`, i.e. the number of edges connected to the node.
    pub fn valence(&self, node: Node) -> usize {
        self.edges_of_node(node).count()
    }

    /// Returns all `(index,face)`-pairs of faces who have the given `node` as a vertex.
    pub fn faces_of_node(&self, node: Node) -> impl Iterator<Item = (usize, &Face)> {
        self.faces
            .iter()
            .enumerate()
            .filter(move |(_, face)| face.contains(&node))
    }

    /// Returns `true` if the face is regular.
    pub fn is_regular(&self, face: Face) -> bool {
        face.iter().all(|node| self.valence(*node) == 4)
    }

    /// Finds the irregular node of the given `face`, if any exists.
    pub fn irregular_node_of_face(&self, face: Face) -> Option<Node> {
        face.into_iter()
            .find(|&v| self.valence(v) != 4)
    }

    /// Returns all adjacent faces to `face`.
    pub fn adjacent_faces(&self, face: Face) -> impl Iterator<Item = (usize, &Face)> {
        self.faces
            .iter()
            .enumerate()
            .filter(move |(_, f)| are_adjacent(f, &face))
    }

    /// Returns whether the given `face` is a boundary face, i.e. it has less than `4` adjacent faces.
    pub fn is_boundary_face(&self, face: Face) -> bool {
        self.adjacent_faces(face).count() < 4
    }

    /// Returns whether the given `node` is a boundary node,
    /// i.e. all faces containing the node are boundary faces.
    pub fn is_boundary_node(&self, node: Node) -> bool {
        self.faces_of_node(node).all(|(_, &f)| self.is_boundary_face(f))
    }

    /// Finds all boundary nodes of the given `face`,
    /// i.e. all irregular nodes, assuming the face is a boundary face.
    pub fn boundary_nodes_of_face(&self, face: Face) -> Vec<Node> {
        face.into_iter().filter(|&v| self.valence(v) != 4).collect()
    }
}

/// Geometric mesh of quadrilateral elements.
#[derive(Debug, Clone, Default)]
pub struct QuadMesh<T: RealField> {
    /// Coordinates of the meshes vertices.
    pub nodes: Vec<Point2<T>>,
    /// Connectivity of this mesh, i.e. the topological mesh.
    pub connectivity: TopologicalMesh,
}

impl<T: RealField> Deref for QuadMesh<T> {
    type Target = TopologicalMesh;

    fn deref(&self) -> &Self::Target {
        &self.connectivity
    }
}

impl<T: RealField> DerefMut for QuadMesh<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.connectivity
    }
}

impl<T: RealField + Copy> QuadMesh<T> {
    /// Returns the point of the given `node` index.
    pub fn node(&self, node: Node) -> Point2<T> {
        self.nodes[node]
    }

    // todo: possibly move this method to TopologicalMesh
    /// Returns the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the nodes of the given `edge`.
    pub fn nodes_of_edge(&self, edge: &Edge) -> [Point2<T>; 2] {
        edge.map(|n| self.node(n))
    }

    /// Returns the nodes of the given `face`.
    pub fn nodes_of_face(&self, face: Face) -> [Point2<T>; 4] {
        face.map(|node| self.node(node))
    }

    /// Computes the centroid of the given `face`.
    pub fn centroid(&self, face: Face) -> Point2<T> {
        let corners = self.nodes_of_face(face);
        let centroid =
            corners.iter().map(|p| p.coords).sum::<Vector2<T>>() / T::from_f64(4.0).unwrap();
        Point2::from(centroid)
    }

    // todo: possibly move this method to TopologicalMesh
    /// Returns an iterator over the indices of all boundary nodes in this mesh.
    pub fn boundary_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        (0..self.num_nodes()).filter(|&n| self.is_boundary_node(n))
    }

    /// Finds the patch of the regular or irregular `face`.
    pub fn find_patch(&self, face: Face) -> Patch<T> {
        Patch::find(self, face)
    }
    
    /// Returns an iterator over all patches in this mesh.
    pub fn patches(&self) -> impl Iterator<Item = Patch<T>> {
        self.faces.iter().map(|&face| self.find_patch(face))
    }
}
