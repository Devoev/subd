use crate::mesh::elem_vertex::ElemVertexMesh;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::Mesh;
use crate::subd::catmull_clark::patch::{CatmarkPatch, CatmarkPatchNodes};
use nalgebra::RealField;
use num_traits::ToPrimitive;

/// Catmull-Clark mesh.
pub type CatmarkMesh<T, const M: usize> = ElemVertexMesh<T, CatmarkPatchNodes, 2, M>;

impl <T: RealField, const M: usize> CatmarkMesh<T, M> {
    /// Converts the given quad-vertex `msh` to a [`CatmarkMesh`].
    ///
    /// This is done by finding the patch corresponding to every quadrilateral
    /// using [`CatmarkPatchNodes::find`].
    pub fn from_quad_mesh(msh: QuadVertexMesh<T, M>) -> Self {
        let patches = msh.elems
            .iter()
            .map(|quad| CatmarkPatchNodes::find(&msh, quad))
            .collect();
        CatmarkMesh::new(msh.coords, patches)
    }
}

impl <'a, T: RealField + Copy + ToPrimitive, const M: usize> Mesh<'a, T, 2, M> for CatmarkMesh<T, M> {
    type GeoElem = CatmarkPatch<T, M>;

    fn geo_elem(&'a self, elem: &Self::Elem) -> Self::GeoElem {
        CatmarkPatch::from_msh(self, elem)
    }
}