use crate::cells::quad::QuadTopo;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::subd::lin_subd::stencil::{EdgeMidpointStencil, FaceMidpointStencil};
use nalgebra::RealField;
use crate::cells::line_segment::UndirectedEdge;

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
        let mut faces = Vec::<QuadTopo>::new();

        // Refine every mesh face
        for i in 0..quad_msh.elems.len() {
            let face = quad_msh.elems[i];

            // Get edges of initial face
            let edges = face.edges();

            // Calculate and add new mid-edge points
            let midpoints = edges.map(|edge| {
                // Sort edge, to avoid duplicates
                let edge = UndirectedEdge::from(edge);

                // Computes the midpoint of the edge.
                edge_stencil.get_or_refine(quad_msh, edge)
            });

            // Calculate new center point
            let m = face_stencil.refine(quad_msh, face);

            // Add subdivided faces
            let [a, b, c, d] = face.nodes();
            let [ab, bc, cd, da] = midpoints;

            faces.push(QuadTopo([a, ab, m, da]));
            faces.push(QuadTopo([ab, b, bc, m]));
            faces.push(QuadTopo([m, bc, c, cd]));
            faces.push(QuadTopo([da, m, cd, d]));
        }

        // Update faces
        quad_msh.elems = faces
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