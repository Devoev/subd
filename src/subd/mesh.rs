use crate::mesh::elem_vertex::ElementVertexMesh;
use crate::mesh::elem_vertex_topo::{ElementVertex, QuadVertex};
use crate::subd::patch::CatmarkPatchNodes;

/// Catmull-Clark mesh.
pub type CatmarkMesh<T, const M: usize> = ElementVertexMesh<2, T, CatmarkPatchNodes>;

/// Topology of a Catmull-Clark mesh.
pub type CatmarkMeshTopology = ElementVertex<2, CatmarkPatchNodes>;

impl CatmarkMeshTopology {
    /// Converts the given quad-vertex topology `msh` to a [`CatmarkMeshTopology`].
    ///
    /// This is done by finding the patch corresponding to every quadrilateral
    /// using [`CatmarkPatchNodes::find`].
    pub fn from_quad_mesh(msh: &QuadVertex) -> Self {
        let patches = msh.elems
            .iter()
            .map(|quad| CatmarkPatchNodes::find(msh, quad))
            .collect();
        CatmarkMeshTopology::new(patches, msh.num_nodes)
    }
}