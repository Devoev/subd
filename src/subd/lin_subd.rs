use std::collections::HashMap;
use nalgebra::{center, RealField};
use crate::cells::line_segment::LineSegmentTopo;
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadTopo};
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::MeshTopology;

/// Linear subdivision of a quad-vertex mesh.
pub struct LinSubd<T: RealField, const M: usize>(pub QuadVertexMesh<T, M>);

impl <T: RealField, const M: usize> LinSubd<T, M> {
    /// Constructs a new [`LinSubd`] from the given quad-vertex mesh `msh`.
    pub fn new(msh: QuadVertexMesh<T, M>) -> Self {
        LinSubd(msh)
    }

    /// Refines this mesh by applying linear subdivision once.
    pub fn refine(&mut self) {
        let mut edge_midpoints = HashMap::<LineSegmentTopo, NodeIdx>::new();
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
}