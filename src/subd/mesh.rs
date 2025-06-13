use crate::mesh::elem_vertex_topo::{ElementVertex, QuadVertex};
use crate::subd::patch::CatmullClarkPatchTopology;

/// Topology of a Catmull-Clark mesh.
pub type CatmullClarkMeshTopology = ElementVertex<2, CatmullClarkPatchTopology>;

impl CatmullClarkMeshTopology {
    /// Converts the given quad-vertex topology `msh` to a [`CatmullClarkMeshTopology`].
    /// 
    /// This is done by finding the patch corresponding to every quadrilateral 
    /// using [`CatmullClarkPatchTopology::find`].
    pub fn from_quad_mesh(msh: &QuadVertex) -> Self {
        let patches = msh.elems
            .iter()
            .map(|quad| CatmullClarkPatchTopology::find(msh, quad))
            .collect();
        CatmullClarkMeshTopology::new(patches, msh.num_nodes)
    }
}