use crate::subd::face::{edges_of_face, is_adjacent, sort_face};
use crate::subd::patch::{ExtendedPatch, Patch};
use itertools::{chain, Itertools};
use nalgebra::{
    center, matrix, vector, DMatrix, DVector, MatrixXx2
    , Point2, RealField, RowDVector, SMatrix, SVector, Vector2,
};
use std::collections::HashMap;
use std::iter::once;
use std::ops::{Deref, DerefMut};
use crate::subd::basis::eval_regular;

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

    /// Returns the nodes of the given `edge`.
    pub fn nodes_of_edge(&self, edge: &Edge) -> [Point2<T>; 2] {
        edge.map(|n| self.node(n))
    }

    /// Returns the nodes of the given `face`.
    pub fn nodes_of_face(&self, face: Face) -> [Point2<T>; 4] {
        face.map(|node| self.node(node))
    }

    /// Returns all faces who have the given `node` as a vertex.
    pub fn faces_of_node(&self, node: Node) -> impl Iterator<Item = (usize, &Face)> {
        self.faces
            .iter()
            .enumerate()
            .filter(move |(_, face)| face.contains(&node))
    }

    /// Calculates the valence of the given `node`.
    pub fn valence(&self, node: Node) -> usize {
        self.faces_of_node(node).count()
    }

    /// Finds the irregular node of the given `face`, if any exists.
    pub fn irregular_node_of_face(&self, face: Face) -> Option<Node> {
        face.into_iter()
            .find(|&v| self.valence(v) != 4)
    }

    /// Returns `true` if the face is regular.
    pub fn is_regular(&self, face: Face) -> bool {
        face.iter().all(|node| self.valence(*node) == 4)
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
            .filter(move |(_, f)| is_adjacent(f, &face))
    }

    /// Returns the one-ring around the given `node`.
    pub fn face_ring(&self, node: Node) -> Vec<&Face> {
        let mut faces = self.faces_of_node(node).map(|(_, f)| f).collect_vec();
        let mut sorted_faces = vec![faces.pop().unwrap()];
        while !faces.is_empty() {
            let prev = sorted_faces.last().unwrap();
            let (idx, next) = faces
                .iter()
                .find_position(|f| is_adjacent(f, prev))
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
        Patch::find(self, face, self.irregular_node_of_face(face).unwrap_or(face[0]))
    }

    /// Finds an extended patch of the irregular `face`.
    pub fn find_patch_ext(&self, face: Face) -> ExtendedPatch<T> {
        ExtendedPatch::find(self, face)
    }

    /// Evaluates the patch of `face` at the parametric point `(u,v)`.
    pub fn eval_patch(&self, face: Face, u: T, v: T) -> SVector<T, 2> {
        // Convert patch control points to matrix
        let patch_nodes = self.find_patch(face).nodes_regular();
        let rows = patch_nodes
            .iter()
            .map(|node| self.node(*node).coords)
            .collect_vec();
        let coords = SMatrix::<T, 2, 16>::from_columns(&rows);

        // Evaluate basis functions at (u,v)
        let b = eval_regular(u, v);

        // Compute surface point
        coords * b
    }
}
