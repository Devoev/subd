use std::collections::HashMap;
use itertools::Itertools;
use nalgebra::{center, RealField};
use crate::subd::face::{edges_of_face, sort_face};
use crate::subd::mesh::{Edge, Face, Node, QuadMesh};

impl <T: RealField + Copy> QuadMesh<T> {

    /// Linearly subdivides this mesh.
    pub fn lin_subd(&mut self) {
        let mut edge_midpoints = HashMap::<Edge, Node>::new();
        let mut faces = Vec::<Face>::new();

        // Refine every mesh face
        for i in 0..self.faces.len() {
            let face = self.faces[i];

            // Get edges of initial face
            let edges = edges_of_face(face);

            // Calculate and add new mid-edge points
            let midpoints = edges.map(|mut edge| {
                // Sort edge, to avoid duplicates
                edge.sort();

                // Computes the midpoint of the edge.
                match edge_midpoints.get(&edge) {
                    Some(node) => *node,
                    None => {
                        let [a, b] = self.nodes_of_edge(&edge);
                        let node = self.num_nodes();
                        self.nodes.push(center(&a, &b));
                        edge_midpoints.insert(edge, node);
                        node
                    }
                }
            });

            // Calculate new center point
            let center = self.centroid(face);
            let m = self.num_nodes();
            self.nodes.push(center);

            // Add subdivided faces
            let [a, b, c, d] = face;
            let [ab, bc, cd, da] = midpoints;

            faces.push([a, ab, m, da]);
            faces.push([ab, b, bc, m]);
            faces.push([m, bc, c, cd]);
            faces.push([da, m, cd, d]);
        }

        // Update faces
        self.faces = faces
    }

    /// Computes the dual of this mesh.
    pub fn dual(&mut self) {
        // Get dual nodes by calculating centroids of faces
        let dual_nodes = self
            .faces
            .iter()
            .map(|face| self.centroid(*face))
            .collect_vec();

        // Get dual faces by finding all dual nodes/ primal faces
        let dual_faces: Vec<Face> = (0..self.num_nodes())
            .flat_map(|node| {
                let adjacent_faces = self.faces_of_node(node).map(|(i, _)| i as Node);
                let dual_face = adjacent_faces.collect_array()?;
                let vertices = dual_face.map(|v| dual_nodes[v]);
                Some(sort_face(dual_face, vertices, self.node(node)))
            })
            .collect_vec();

        self.nodes = dual_nodes;
        self.faces = dual_faces;
    }

    /// Refine this mesh by repeated averaging algorithm.
    pub fn repeated_averaging(&mut self, levels: usize, p: usize) {
        for _ in 0..levels {
            self.lin_subd();
            for _ in 0..p - 1 {
                self.dual();
            }
        }
    }
}