use crate::cells::line_segment::UndirectedEdge;
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadTopo};
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::MeshTopology;
use nalgebra::{center, RealField};
use std::collections::HashMap;

/// Stencil for a linearly-subdivided edge
/// ```text
/// 1/2 --- ○ --- 1/2
/// ```
#[derive(Debug, Clone, Default)]
pub struct EdgeMidpointStencil {
    /// Edge-to-midpoint map.
    edge_midpoints: HashMap<UndirectedEdge, NodeIdx>
}

impl EdgeMidpointStencil {
    /// Constructs a new [`EdgeMidpointStencil`].
    pub fn new() -> Self {
        EdgeMidpointStencil { edge_midpoints: HashMap::new() }
    }

    /// Returns the midpoint node corresponding to the given `edge` or `None`,
    /// if the edge is not refined yet.
    pub fn get(&self, edge: &UndirectedEdge) -> Option<&NodeIdx> {
        self.edge_midpoints.get(edge)
    }

    /// Refines the given `edge` and adds the coordinates of the new midpoint to the `quad_msh`.
    /// The index of the midpoint node is inserted into `edge_midpoints` and returned.
    pub fn refine<T: RealField, const M: usize>(
        &mut self,
        quad_msh: &mut QuadVertexMesh<T, M>,
        edge: UndirectedEdge,
    ) -> NodeIdx {
        let a = quad_msh.coords(edge.first());
        let b = quad_msh.coords(edge.second());
        let node = NodeIdx(quad_msh.num_nodes());
        quad_msh.coords.push(center(a, b));
        self.edge_midpoints.insert(edge, node);
        node
    }

    /// Returns the midpoint node corresponding to the given `edge`.
    /// If the edge is not refined yet the midpoint is first calculated using [EdgeMidpointStencil::refine].
    pub fn get_or_refine<T: RealField, const M: usize>(
        &mut self,
        quad_msh: &mut QuadVertexMesh<T, M>,
        edge: UndirectedEdge,
    ) -> NodeIdx {
        match self.get(&edge) {
            Some(node) => *node,
            None => {
                self.refine(quad_msh, edge)
            }
        }
    }
}

/// Stencil for a linearly-subdivided quadrilateral face
/// ```text
/// 1/4 --- 1/4
///  |       |
///  |   ○   |
///  |       |
/// 1/4 --- 1/4
/// ```
#[derive(Debug, Clone, Default)]
pub struct FaceMidpointStencil;

impl FaceMidpointStencil {
    /// Constructs a new [`FaceMidpointStencil`].
    pub fn new() -> Self {
        FaceMidpointStencil
    }

    /// Refines the given `face` and adds the coordinates of the new midpoint to the `quad_msh`.
    /// The index of the midpoint node is returned.
    pub fn refine<T: RealField, const M: usize>(
        &mut self,
        quad_msh: &mut QuadVertexMesh<T, M>,
        face: QuadTopo,
    ) -> NodeIdx {
        let quad = Quad::from_msh(face, quad_msh);
        let center = quad.centroid();
        quad_msh.coords.push(center);
        NodeIdx(quad_msh.num_nodes() - 1)
    }
}