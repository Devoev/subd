use nalgebra::RealField;
use crate::mesh::elem_vertex::ElementVertexMesh;
use crate::mesh::elem_vertex_topo;
use crate::mesh::elem_vertex_topo::{ElementVertex, QuadVertex};
use crate::mesh::topo::MeshTopology;
use crate::mesh::traits::Mesh;
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

impl <'a, T: RealField + Copy, const M: usize> Mesh<'a, T, (T, T), 2, M> for CatmarkMesh<T, M> {
    type Elem = &'a CatmarkPatchNodes;
    type GeoElem = CatmarkPatch<T, M>;
    type NodesIter = elem_vertex_topo::NodesIter;
    type ElemsIter = std::slice::Iter<'a, CatmarkPatchNodes>;

    fn num_nodes(&self) -> usize {
        self.topology.num_nodes
    }

    fn num_elems(&self) -> usize {
        self.topology.elems.len()
    }

    fn nodes(&'a self) -> Self::NodesIter {
        self.topology.nodes()
    }

    fn elems(&'a self) -> Self::ElemsIter {
        self.topology.elems.iter()
    }

    fn geo_elem(&'a self, elem: Self::Elem) -> Self::GeoElem {
        CatmarkPatch::from_msh(self, elem)
    }
}