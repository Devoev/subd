use crate::subd::edge::sort_edge;
use crate::subd::face::{edges_of_face, are_adjacent};
use crate::subd::patch::{ExtendedPatch, Patch};
use itertools::Itertools;
use nalgebra::{
    Point2, RealField, Vector2,
};
use std::ops::{Deref, DerefMut};

pub type Node = usize;
pub type Edge = [Node; 2];
pub type Face = [Node; 4];

#[derive(Debug, Clone, Default)]
pub struct LogicalMesh {
    pub faces: Vec<Face>,
}

#[derive(Debug, Clone, Default)]
pub struct QuadMesh<T: RealField> {
    pub nodes: Vec<Point2<T>>,

    pub logical_mesh: LogicalMesh,
}

impl<T: RealField> Deref for QuadMesh<T> {
    type Target = LogicalMesh;

    fn deref(&self) -> &Self::Target {
        &self.logical_mesh
    }
}

impl<T: RealField> DerefMut for QuadMesh<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.logical_mesh
    }
}

impl<T: RealField + Copy> QuadMesh<T> {
    /// Returns the point of the given `node` index.
    pub fn node(&self, node: Node) -> Point2<T> {
        self.nodes[node]
    }

    /// Returns the number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Finds the index of the given face.
    pub fn face_idx(&self, face: Face) -> usize {
        self.faces.iter().position(|f| f == &face).unwrap()
    }

    /// Returns an iterator over all unique and sorted edges in this mesh.
    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.faces.iter()
            .flat_map(|&face| edges_of_face(face))
            .map(sort_edge)
            .unique()
    }

    /// Returns the nodes of the given `edge`.
    pub fn nodes_of_edge(&self, edge: &Edge) -> [Point2<T>; 2] {
        edge.map(|n| self.node(n))
    }

    /// Returns the nodes of the given `face`.
    pub fn nodes_of_face(&self, face: Face) -> [Point2<T>; 4] {
        face.map(|node| self.node(node))
    }

    /// Returns all `(index,face)`-pairs of faces who have the given `node` as a vertex.
    pub fn faces_of_node(&self, node: Node) -> impl Iterator<Item = (usize, &Face)> {
        self.faces
            .iter()
            .enumerate()
            .filter(move |(_, face)| face.contains(&node))
    }

    /// Returns all edges connected to the given `node`.
    pub fn edges_of_node(&self, node: Node) -> impl Iterator<Item = Edge> + '_ {
        self.edges().filter(move |edge| edge.contains(&node))
    }

    /// Calculates the valence of the given `node`, i.e. the number of edges connected to the node.
    pub fn valence(&self, node: Node) -> usize {
        self.edges_of_node(node).count()
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

    /// Returns whether the given `face` is a boundary face, i.e. it has less than `4` adjacent faces.
    pub fn is_boundary_face(&self, face: Face) -> bool {
        self.adjacent_faces(face).count() < 4
    }

    /// Returns whether the given `node` is a boundary node, 
    /// i.e. all faces containing the node are boundary faces.
    pub fn is_boundary_node(&self, node: Node) -> bool {
        self.faces_of_node(node).all(|(_, &f)| self.is_boundary_face(f))
    }
    
    /// Returns an iterator over the indices of all boundary nodes in this mesh.
    pub fn boundary_nodes(&self) -> impl Iterator<Item = Node> + '_ {
        (0..self.num_nodes()).filter(|&n| self.is_boundary_node(n))
    }
    
    /// Finds all boundary nodes of the given `face`,
    /// i.e. all irregular nodes, assuming the face is a boundary face.
    pub fn boundary_nodes_of_face(&self, face: Face) -> Vec<Node> {
        face.into_iter().filter(|&v| self.valence(v) != 4).collect()
    }

    /// Computes the centroid of the given `face`.
    pub fn centroid(&self, face: Face) -> Point2<T> {
        let corners = self.nodes_of_face(face);
        let centroid =
            corners.iter().map(|p| p.coords).sum::<Vector2<T>>() / T::from_f64(4.0).unwrap();
        Point2::from(centroid)
    }

    /// Returns all adjacent faces to `face`.
    pub fn adjacent_faces(&self, face: Face) -> impl Iterator<Item = (usize, &Face)> {
        self.faces
            .iter()
            .enumerate()
            .filter(move |(_, f)| are_adjacent(f, &face))
    }

    /// Returns the one-ring around the given `node`.
    pub fn face_ring(&self, node: Node) -> Vec<&Face> {
        let mut faces = self.faces_of_node(node).map(|(_, f)| f).collect_vec();
        let mut sorted_faces = vec![faces.pop().unwrap()];
        while !faces.is_empty() {
            let prev = sorted_faces.last().unwrap();
            let (idx, next) = faces
                .iter()
                .find_position(|f| are_adjacent(f, prev))
                .unwrap();
            sorted_faces.push(next);
            faces.remove(idx);
        }
        sorted_faces
    }

    /// Returns the one-ring of nodes around the given `node`.
    pub fn node_ring(&self, node: Node) -> Vec<Node> {
        let mut faces = self.face_ring(node).into_iter().cloned().collect_vec();

        faces
            .iter_mut()
            .flat_map(|face| {
                let idx = face.iter().position(|v| *v == node).unwrap();
                face.rotate_left(idx);
                face[1..=2].iter().copied()
            })
            .collect_vec()
    }

    /// Finds the patch of the regular or irregular `face`.
    pub fn find_patch(&self, face: Face) -> Patch<T> {
        // todo: describe how the starting node is selected or change
        let start = if self.is_boundary_face(face) {
            // Get the irregular (=boundary) node, such that the preceding node is regular
            face.into_iter().enumerate()
                .find_map(|(idx, node)| {
                    let next_idx = (idx + 1) % 4;
                    let next_node = face[next_idx];
                    (self.valence(node) == 4 && self.valence(next_node) != 4).then_some(next_node)
                }).unwrap()
        } else if !self.is_regular(face) {
            // Get irregular node, if face is irregular
            self.irregular_node_of_face(face).unwrap()
        } else {
            // Get arbitrary node, if face is regular
            face[0]
        };

        Patch::find(self, face, start)
    }
    
    /// Returns an iterator over all patches in this mesh.
    pub fn patches(&self) -> impl Iterator<Item = Patch<T>> {
        self.faces.iter().map(|&face| self.find_patch(face))
    }

    /// Finds an extended patch of the irregular `face`.
    pub fn find_patch_ext(&self, face: Face) -> ExtendedPatch<T> {
        ExtendedPatch::find(self, face)
    }
}
