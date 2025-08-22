//! Implementation of linear subdivision as a mesh.

use std::collections::HashMap;
use nalgebra::{center, DimName, DimNameSub, Point, RealField, U2};
use crate::cells::line_segment::{LineSegment, DirectedEdge};
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadNodes};
use crate::cells::topo;
use crate::mesh::elem_vertex::ElemVertexMesh;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::{Mesh, MeshTopology};

/// A linearly-subdivided quad with subdivision stencil
/// ```text
/// 1/4 --- 1/4
///  |       |
///  |   ○   |
///  |       |
/// 1/4 --- 1/4
/// ```
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
struct LinFace(pub QuadNodes);

impl LinFace {
    /// Returns the nodes as in [`QuadNodes::nodes`].
    pub fn nodes(&self) -> [NodeIdx; 4] {
        self.0.0
    }

    /// Returns the edges as in [`QuadNodes::edges`] as `LinEdge`s.
    pub fn edges(&self) -> [LinEdge; 4] {
        self.0.edges().map(LinEdge)
    }

    /// Linearly subdivides the given `quad` coordinates.
    pub fn refine<T: RealField, const M: usize>(quad: &Quad<T, M>) -> Point<T, M> {
        quad.centroid()
    }
}

impl topo::Cell<U2> for LinFace {
    fn nodes(&self) -> &[NodeIdx] {
        &self.0.0
    }

    fn is_connected<M: DimName>(&self, other: &Self, dim: M) -> bool
    where
        U2: DimNameSub<M>
    {
        self.0.is_connected(&other.0, dim)
    }
}

/// A linearly-subdivided edge with subdivision stencil
/// ```text
/// 1/2 --- ○ --- 1/2
/// ```
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
struct LinEdge(pub(crate) DirectedEdge);

impl LinEdge {
    /// Linearly subdivision the given `line` coordinates.
    pub fn refine<T: RealField, const M: usize>(line: &LineSegment<T, M>) -> Point<T, M> {
        let [a, b] = &line.vertices;
        center(a, b)
    }
}

// todo: remove this below!

/// Linear subdivision of a quad-vertex mesh.
type LinSubdMesh<T, const M: usize> = ElemVertexMesh<T, LinFace, 2, M>;

impl <T: RealField + Copy, const M: usize> LinSubdMesh<T, M> {
    /// Constructs a new [`crate::subd::lin_subd::refine::LinSubd`] from the given quad-vertex mesh.
    fn from_quad_msh(quad_msh: QuadVertexMesh<T, M>) -> Self {
        let elems = quad_msh.elems.into_iter()
            .map(LinFace)
            .collect();
        LinSubdMesh::new(quad_msh.coords, elems)
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
            let quad = self.geo_elem(&&face);
            let center = LinFace::refine(&quad);
            let m = NodeIdx(self.num_nodes());
            self.coords.push(center);

            // Add subdivided faces
            let [a, b, c, d] = face.nodes();
            let [ab, bc, cd, da] = midpoints;

            faces.push(LinFace(QuadNodes([a, ab, m, da])));
            faces.push(LinFace(QuadNodes([ab, b, bc, m])));
            faces.push(LinFace(QuadNodes([m, bc, c, cd])));
            faces.push(LinFace(QuadNodes([da, m, cd, d])));
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

impl<'a, T: RealField + Copy, const M: usize> Mesh<'a, T, 2, M> for LinSubdMesh<T, M> {
    type GeoElem = Quad<T, M>;

    fn geo_elem(&'a self, elem: &Self::Elem) -> Self::GeoElem {
        Quad::new(elem.nodes().map(|n| *self.coords(n)))
    }
}