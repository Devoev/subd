use crate::cells::quad::QuadNodes;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::subd::lin_subd::stencil::{EdgeMidpointStencil, FaceMidpointStencil};
use nalgebra::RealField;
use crate::cells::line_segment::UndirectedEdge;
use crate::cells::node::NodeIdx;

/// Linear subdivision of a quad-vertex mesh.
#[derive(Debug, Clone)]
pub struct LinSubd<T: RealField, const M: usize> {
    /// The refined mesh.
    refined_mesh: QuadVertexMesh<T, M>,
}

impl <T: RealField, const M: usize> LinSubd<T, M> {
    /// Constructs a new [`LinSubd`] from the given quad-vertex mesh.
    pub fn new(mut quad_msh: QuadVertexMesh<T, M>) -> Self {
        LinSubd::do_refine(&mut quad_msh);
        LinSubd { refined_mesh: quad_msh }
    }

    /// Refines the given `quad_msh` once.
    fn do_refine(quad_msh: &mut QuadVertexMesh<T, M>) {
        let mut edge_stencil = EdgeMidpointStencil::new();
        let mut face_stencil = FaceMidpointStencil::new();

        let mut refined_faces = Vec::<QuadNodes>::new();
        let mut add_face_nodes = |a: NodeIdx, b: NodeIdx, c: NodeIdx, d: NodeIdx| {
            refined_faces.push(QuadNodes([a, b, c, d]))
        };

        // Refine every mesh face
        for i in 0..quad_msh.elems.len() {
            let face = quad_msh.elems[i];

            // Calculate and add new mid-edge points for each edge
            let midpoints = face
                .undirected_edges()
                .map(|edge| edge_stencil.get_or_refine(quad_msh, edge));

            // Calculate new center point
            let m = face_stencil.refine(quad_msh, face);

            // Add subdivided faces
            let [a, b, c, d] = face.nodes();
            let [ab, bc, cd, da] = midpoints;

            add_face_nodes(a, ab, m, da);
            add_face_nodes(ab, b, bc, m);
            add_face_nodes(m, bc, c, cd);
            add_face_nodes(da, m, cd, d);
        }

        // Update faces
        quad_msh.elems = refined_faces
    }

    /// Retrieves the refined quad-vertex mesh.
    pub fn unpack(self) -> QuadVertexMesh<T, M> {
        self.refined_mesh
    }

    /// Performs linear subdivision again.
    pub fn lin_subd(self) -> LinSubd<T, M> {
        LinSubd::new(self.refined_mesh)
    }
}

impl <T: RealField, const M: usize> QuadVertexMesh<T, M> {
    /// Linearly subdivides this mesh.
    pub fn lin_subd(self) -> LinSubd<T, M> {
        LinSubd::new(self)
    }
}