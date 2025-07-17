mod stencil;
mod mesh;

use crate::cells::line_segment::NodePair;
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadTopo};
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::incidence::{edge_to_node_incidence, face_to_edge_incidence};
use crate::mesh::traits::MeshTopology;
use itertools::Itertools;
use nalgebra::{center, matrix, Const, DVector, Dyn, OMatrix, Point, RealField, U4};
use nalgebra_sparse::{CooMatrix, CsrMatrix};
use std::collections::HashMap;

/// Linear subdivision of a quad-vertex mesh.
#[derive(Debug, Clone)]
pub struct LinSubd<T: RealField, const M: usize>(pub QuadVertexMesh<T, M>);

impl <T: RealField, const M: usize> LinSubd<T, M> {
    /// Constructs a new [`LinSubd`] from the given quad-vertex mesh `msh`.
    pub fn new(msh: QuadVertexMesh<T, M>) -> Self {
        LinSubd(msh)
    }

    /// Refines this mesh by applying linear subdivision once.
    pub fn refine(&mut self) {
        let mut edge_midpoints = HashMap::<NodePair, NodeIdx>::new();
        let mut faces = Vec::<QuadTopo>::new();

        // Refine every mesh face
        for i in 0..self.0.elems.len() {
            let face = self.0.elems[i];

            // Get edges of initial face
            let edges = face.edges();

            // Calculate and add new mid-edge points
            let midpoints = edges.map(|mut edge| {
                // Sort edge, to avoid duplicates
                edge.sort();

                // Computes the midpoint of the edge.
                match edge_midpoints.get(&edge) {
                    Some(node) => *node,
                    None => {
                        let a = self.0.coords(edge.start());
                        let b = self.0.coords(edge.end());
                        let node = NodeIdx(self.0.num_nodes());
                        self.0.coords.push(center(a, b));
                        edge_midpoints.insert(edge, node);
                        node
                    }
                }
            });

            // Calculate new center point
            let quad = Quad::from_msh(face, &self.0);
            let center = quad.centroid();
            let m = NodeIdx(self.0.num_nodes());
            self.0.coords.push(center);

            // Add subdivided faces
            let [a, b, c, d] = face.nodes();
            let [ab, bc, cd, da] = midpoints;

            faces.push(QuadTopo([a, ab, m, da]));
            faces.push(QuadTopo([ab, b, bc, m]));
            faces.push(QuadTopo([m, bc, c, cd]));
            faces.push(QuadTopo([da, m, cd, d]));
        }

        // Update faces
        self.0.elems = faces
    }

    /// Refines this mesh. Instead of iteration over all quads,
    /// this method first adds new edge-midpoints and then quad-midpoints.
    fn refine_alt(&mut self) {
        // Create edge-midpoints
        let mut edge_midpoints = HashMap::<NodePair, NodeIdx>::new();
        let edges = self.0.edges().collect_vec();
        for edge in edges {
            let start = self.0.coords(edge.start());
            let end = self.0.coords(edge.end());
            let midpoint = center(start, end);
            let node = NodeIdx(self.0.num_nodes());

            self.0.coords.push(midpoint);
            edge_midpoints.insert(edge, node);
        }

        // Create face-midpoints
        let mut face_midpoints = HashMap::<QuadTopo, NodeIdx>::new();
        for i in 0..self.0.num_elems() {
            let face = self.0.elems[i];
            let quad = Quad::from_msh(face, &self.0);
            let midpoint = quad.centroid();
            let node = NodeIdx(self.0.num_nodes());

            self.0.coords.push(midpoint);
            face_midpoints.insert(face, node);
        }

        // Update connectivity
        let mut faces = Vec::<QuadTopo>::new();
        for face_idx in 0..self.0.elems.len() {
            let face = self.0.elems[face_idx];
            let [a, b, c, d] = face.nodes();
            let [ab, bc, cd, da] = face
                .edges()
                .map(|edge| *edge_midpoints.get(&edge.sorted()).unwrap());
            let &m = face_midpoints.get(&face).unwrap();

            // Add subdivided faces
            faces.push(QuadTopo([a, ab, m, da]));
            faces.push(QuadTopo([ab, b, bc, m]));
            faces.push(QuadTopo([m, bc, c, cd]));
            faces.push(QuadTopo([da, m, cd, d]));
        }
        self.0.elems = faces
    }

    /// Refines the mesh. Refinement is performed using matrix-vector multiplications.
    fn refine_alt_mat(&mut self) {
        // Define subdivision (?) matrix
        let S = matrix![
            0.5, 0.5, 0.0, 0.0;
            0.0, 0.5, 0.5, 0.0;
            0.0, 0.0, 0.5, 0.5;
            0.5, 0.0, 0.0, 0.5;
            0.25, 0.25, 0.25, 0.25
        ].cast::<T>();

        // Get coordinate matrix
        let rows = self.0.coords.iter()
            .map(|p| p.coords.transpose())
            .collect_vec();
        let c = OMatrix::<T, Dyn, Const<M>>::from_rows(&rows);

        // Node to edge matrix
        let (row_idx, col_idx, val) = edge_to_node_incidence(&self.0).disassemble();
        let val = val.iter()
            .map(|v| {
                T::from_f64(v.abs() as f64 / 2.0).unwrap()
            })
            .collect_vec();
        let node_to_edge = CooMatrix::try_from_triplets(
            self.0.edges().count(), self.0.num_nodes(), col_idx, row_idx, val
        ).unwrap();
        let subd_edges = CsrMatrix::from(&node_to_edge);

        // Face to edge adjacency
        let face_to_edge = face_to_edge_incidence(&self.0);
        let face_to_edge = CsrMatrix::from(&face_to_edge).transpose();

        // Node to face matrix
        let mut node_to_face = CooMatrix::new(self.0.num_elems(), self.0.num_nodes());
        for (i, elem) in self.0.elems.iter().enumerate() {
            let [a, b, c, d] = elem.nodes();
            node_to_face.push(i, a.0, T::from_f64(0.25).unwrap());
            node_to_face.push(i, b.0, T::from_f64(0.25).unwrap());
            node_to_face.push(i, c.0, T::from_f64(0.25).unwrap());
            node_to_face.push(i, d.0, T::from_f64(0.25).unwrap());
        }
        let subd_faces = CsrMatrix::from(&node_to_face);

        // Add coordinates
        let edge_points = subd_edges * &c;
        let idx_first_edge = self.0.num_nodes();
        for coords in edge_points.row_iter() {
            self.0.coords.push(Point::from(coords.into_owned().transpose()));
        }

        let idx_first_face = self.0.num_nodes();
        let face_points = subd_faces * &c;
        for coords in face_points.row_iter() {
            self.0.coords.push(Point::from(coords.into_owned().transpose()));
        }

        // Update connectivity
        let mut faces = Vec::<QuadTopo>::new();
        for face_idx in 0..self.0.num_elems() {
            let face = self.0.elems[face_idx];
            let [a, b, c, d] = face.nodes();

            // todo: this doesn't work because the edge indices are not (necessarily) in order ab,bc,cd,da
            let [ab, bc, cd, da] = face_to_edge.row(face_idx)
                .col_indices()
                .iter()
                .map(|idx| NodeIdx(idx + idx_first_edge))
                .collect_array()
                .unwrap();

            // let [ab, bc, cd, da] = face
            //     .edges()
            //     .map(|edge| NodeIdx(edge.start().0 + idx_first_edge)); // todo: this is incorrect! Get midpoint index

            let m = NodeIdx(idx_first_face + face_idx);
            // Add subdivided faces
            faces.push(QuadTopo([a, ab, m, da]));
            faces.push(QuadTopo([ab, b, bc, m]));
            faces.push(QuadTopo([m, bc, c, cd]));
            faces.push(QuadTopo([da, m, cd, d]));
        }

        self.0.elems = faces;
    }
}