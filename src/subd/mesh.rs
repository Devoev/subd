use nalgebra::RealField;
use crate::mesh::elem_vertex::ElementVertexMesh;
use crate::mesh::elem_vertex_topo::{ElementVertex, QuadVertex};
use crate::mesh::geo::Mesh;
use crate::subd::patch::{CatmarkPatch, CatmarkPatchNodes};

/// Catmull-Clark mesh.
pub type CatmarkMesh<T, const M: usize> = ElementVertexMesh<T, CatmarkPatchNodes, 2, M>;

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

impl <'a, T: RealField + Copy, const M: usize> Mesh<'a, T, (T, T), 2, M, CatmarkPatch<T, M>> for CatmarkMesh<T, M> {
    type Elems = impl Iterator<Item = CatmarkPatch<T, M>>;

    fn elems(&'a self) -> Self::Elems {
        self.topology.elems
            .iter()
            .map(move |patch_to_nodes| CatmarkPatch::from_msh(&self, patch_to_nodes))
    }
}