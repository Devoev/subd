use crate::cells::node::Node;
use crate::cells::quad::QuadNodes;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::cell_topology::CellTopology;
use crate::subd::catmull_clark::matrices::assemble_global_mat;
use nalgebra::{Point, RealField};
use nalgebra_sparse::CsrMatrix;
use crate::mesh::elem_vertex::ElemVec;

/// Catmull-Clark subdivision of a quad-vertex mesh.
pub struct CatmarkSubd<T: RealField, const M: usize> {
    /// The refined mesh.
    refined_mesh: QuadVertexMesh<T, M>,
}
// todo: make generic over T
impl <const M: usize> CatmarkSubd<f64, M> {
    /// Constructs a new [`CatmarkSubd`] from the given `quad_msh`.
    pub fn new(mut quad_msh: QuadVertexMesh<f64, M>) -> Self {
        CatmarkSubd::do_refine(&mut quad_msh);
        CatmarkSubd { refined_mesh: quad_msh }
    }

    /// Refines the given `quad_msh` using the global subdivision matrix.
    fn do_refine(quad_msh: &mut QuadVertexMesh<f64, M>) {
        // Refine coords
        let (s, face_midpoints, edge_midpoints) = assemble_global_mat(quad_msh);
        let s = CsrMatrix::from(&s);
        let c = quad_msh.coords_matrix();
        let c_subd = s * &c;

        // Update coords
        quad_msh.coords.clear();
        quad_msh.coords.reserve(c_subd.len());
        for point_coords in c_subd.row_iter() {
            quad_msh.coords.push(Point::from(point_coords.transpose()));
        }

        // Update connectivity
        let mut refined_faces = Vec::<QuadNodes>::with_capacity(quad_msh.cells.len() * 4);
        let mut add_face_nodes = |a: Node, b: Node, c: Node, d: Node| {
            refined_faces.push(QuadNodes([a, b, c, d]))
        };
        for face in quad_msh.cell_iter() {
            // Get corner nodes
            let [a, b, c, d] = face.nodes();

            // Get edge midpoints
            let [ab, bc, cd, da] = face.undirected_edges()
                .map(|edge| edge_midpoints[&edge]);

            // Get face midpoint
            let m = face_midpoints[face];

            // Add refined faces
            add_face_nodes(a, ab, m, da);
            add_face_nodes(ab, b, bc, m);
            add_face_nodes(m, bc, c, cd);
            add_face_nodes(da, m, cd, d);
        }

        // Update faces
        quad_msh.cells = ElemVec(refined_faces)
    }

    /// Retrieves the refined quad-vertex mesh.
    pub fn unpack(self) -> QuadVertexMesh<f64, M> {
        self.refined_mesh
    }

    /// Performs Catmull-Clark subdivision again.
    pub fn catmark_subd(self) -> CatmarkSubd<f64, M> {
        CatmarkSubd::new(self.refined_mesh)
    }
}

impl <const M: usize> QuadVertexMesh<f64, M> {
    /// Subdivides this mesh using Catmull-Clark subdivision.
    pub fn catmark_subd(self) -> CatmarkSubd<f64, M> {
        CatmarkSubd::new(self)
    }
}