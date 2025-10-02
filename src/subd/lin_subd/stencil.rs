use crate::cells::edge::UndirectedEdge;
use crate::cells::node::Node;
use crate::cells::quad::QuadNodes;
use crate::cells::traits::ToElement;
use crate::mesh::face_vertex::QuadVertexMesh;
use nalgebra::{center, RealField};
use std::collections::HashMap;
use crate::mesh::vertex_storage::VertexStorage;

/// Stencil for a linearly-subdivided edge
/// ```text
/// 1/2 --- ○ --- 1/2
/// ```
#[derive(Debug, Clone, Default)]
pub struct EdgeMidpointStencil {
    /// Edge-to-midpoint map.
    edge_midpoints: HashMap<UndirectedEdge, Node>
}

impl EdgeMidpointStencil {
    /// Constructs a new [`EdgeMidpointStencil`].
    pub fn new() -> Self {
        EdgeMidpointStencil { edge_midpoints: HashMap::new() }
    }

    /// Returns the midpoint node corresponding to the given `edge` or `None`,
    /// if the edge is not refined yet.
    pub fn get(&self, edge: &UndirectedEdge) -> Option<&Node> {
        self.edge_midpoints.get(edge)
    }

    /// Refines the given `edge` and adds the coordinates of the new midpoint to the `quad_msh`.
    /// The index of the midpoint node is inserted into `edge_midpoints` and returned.
    pub fn refine<T: RealField, const M: usize>(
        &mut self,
        quad_msh: &mut QuadVertexMesh<T, M>,
        edge: UndirectedEdge,
    ) -> Node {
        let a = quad_msh.coords.vertex(edge.first());
        let b = quad_msh.coords.vertex(edge.second());
        let node = quad_msh.num_nodes();
        quad_msh.coords.push(center(&a, &b));
        self.edge_midpoints.insert(edge, node);
        node
    }

    /// Returns the midpoint node corresponding to the given `edge`.
    /// If the edge is not refined yet the midpoint is first calculated using [EdgeMidpointStencil::refine].
    pub fn get_or_refine<T: RealField, const M: usize>(
        &mut self,
        quad_msh: &mut QuadVertexMesh<T, M>,
        edge: UndirectedEdge,
    ) -> Node {
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
    pub fn refine<T: RealField + Copy, const M: usize>(
        &mut self,
        quad_msh: &mut QuadVertexMesh<T, M>,
        face: QuadNodes,
    ) -> Node {
        let quad = face.to_element(&quad_msh.coords);
        let center = quad.centroid();
        quad_msh.coords.push(center);
        quad_msh.num_nodes() - 1
    }
}