use crate::cells::line_segment::LineSegment;
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadTopo};
use crate::mesh::elem_vertex::ElemVertexMesh;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::{Mesh, MeshTopology};
use crate::subd::lin_subd::stencil::{LinEdge, LinFace};
use nalgebra::RealField;
use std::collections::HashMap;

// todo: should LinSubd be really an ElemVertexMesh with a new type of elements?
//  shouldn't it rather be a wrapper around a QuadVertexMesh (or possibly any mesh)?
//  Take inspiration of nalgebra's decompositions

/// Linear subdivision of a quad-vertex mesh.
pub type LinSubd<T, const M: usize> = ElemVertexMesh<T, LinFace, 2, M>;

impl <T: RealField + Copy, const M: usize> LinSubd<T, M> {
    /// Constructs a new [`LinSubd`] from the given quad-vertex mesh.
    fn from_quad_msh(quad_msh: QuadVertexMesh<T, M>) -> Self {
        let elems = quad_msh.elems.into_iter()
            .map(LinFace)
            .collect();
        LinSubd::new(quad_msh.coords, elems)
    }

    /// Refines this mesh by applying linear subdivision once.
    pub fn refine(&mut self) {
        let mut edge_midpoints = HashMap::<LinEdge, NodeIdx>::new();
        let mut faces = Vec::<LinFace>::new();

        // Refine every mesh face
        for i in 0..self.elems.len() {
            let face = self.elems[i];

            // Get edges of initial face
            let edges = face.edges();

            // Calculate and add new mid-edge points
            let midpoints = edges.map(|mut edge| {
                // Sort edge, to avoid duplicates
                edge.0.sort();

                // Computes the midpoint of the edge.
                match edge_midpoints.get(&edge) {
                    Some(node) => *node,
                    None => {
                        let line = self.geo_edge(&edge);
                        let node = NodeIdx(self.num_nodes());
                        self.coords.push(LinEdge::refine(&line));
                        edge_midpoints.insert(edge, node);
                        node
                    }
                }
            });

            // Calculate new center point
            let quad = self.geo_elem(&face);
            let center = LinFace::refine(&quad);
            let m = NodeIdx(self.num_nodes());
            self.coords.push(center);

            // Add subdivided faces
            let [a, b, c, d] = face.nodes();
            let [ab, bc, cd, da] = midpoints;

            faces.push(LinFace(QuadTopo([a, ab, m, da])));
            faces.push(LinFace(QuadTopo([ab, b, bc, m])));
            faces.push(LinFace(QuadTopo([m, bc, c, cd])));
            faces.push(LinFace(QuadTopo([da, m, cd, d])));
        }

        // Update faces
        self.elems = faces
    }

    /// Constructs the geometric line segment corresponding to the given `edge`.
    fn geo_edge(&self, edge: &LinEdge) -> LineSegment<T, M> {
        let [a, b] = &edge.0.0;
        LineSegment::new([*self.coords(*a), *self.coords(*b)])
    }
}

impl<'a, T: RealField + Copy, const M: usize> Mesh<'a, T, (T, T), 2, M> for LinSubd<T, M> {
    type GeoElem = Quad<T, M>;

    fn geo_elem(&'a self, elem: Self::Elem) -> Self::GeoElem {
        Quad::new(elem.nodes().map(|n| *self.coords(n)))
    }
}