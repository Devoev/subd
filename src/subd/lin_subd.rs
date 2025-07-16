use std::collections::HashMap;
use itertools::Itertools;
use nalgebra::{center, matrix, stack, Const, DefaultAllocator, RealField, SMatrix, SVector, U4, U5};
use nalgebra::allocator::Allocator;
use crate::cells::line_segment::NodePair;
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadTopo};
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::MeshTopology;

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
    pub fn refine_alt(&mut self) {
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
            let [&ab, &bc, &cd, &da] = face
                .edges()
                .map(|edge| edge_midpoints.get(&edge.sorted()).unwrap());
            let &m = face_midpoints.get(&face).unwrap();

            // Add subdivided faces
            faces.push(QuadTopo([a, ab, m, da]));
            faces.push(QuadTopo([ab, b, bc, m]));
            faces.push(QuadTopo([m, c, c, cd]));
            faces.push(QuadTopo([da, m, cd, d]));
        }
        self.0.elems = faces
    }

    /// Refines the mesh. Refinement is performed using matrix-vector multiplications.
    pub fn refine_alt_mat(&mut self) {
        // Define subdivision (?) matrix
        let S = matrix![
            0.5, 0.5, 0.0, 0.0;
            0.0, 0.5, 0.5, 0.0;
            0.0, 0.0, 0.5, 0.5;
            0.5, 0.0, 0.0, 0.5;
            0.25, 0.25, 0.25, 0.25
        ].cast::<T>();

        let mut edge_midpoints = HashMap::<NodePair, NodeIdx>::new();

        // Refine every mesh face
        for i in 0..self.0.elems.len() {
            let face = self.0.elems[i];
            let quad = Quad::from_msh(face, &self.0);
            let [a, b, c, d] = quad.vertices;
            let c = stack![a.coords, b.coords, c.coords, d.coords].transpose();
            let c = S.clone() * c;
            let ab = c.row(0);
            let bc = c.row(1);
            let cd = c.row(2);
            let da = c.row(3);
            let m = c.row(4);

            todo!("append nodes to coords array and update face connectivity")
        }
    }
}