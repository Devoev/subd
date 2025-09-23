use crate::mesh::elem_vertex::ElemVertexMesh;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::Mesh;
use crate::subd::catmull_clark::patch::{CatmarkPatch, CatmarkPatchNodes};
use nalgebra::{Point, RealField};
use num_traits::ToPrimitive;
use crate::cells::quad::QuadNodes;

/// Catmull-Clark mesh.
pub type CatmarkMesh<T, const M: usize> = ElemVertexMesh<T, CatmarkPatchNodes, 2, M>;

impl <T: RealField, const M: usize> CatmarkMesh<T, M> {
    
}

impl <'a, T: RealField + Copy + ToPrimitive, const M: usize> Mesh<'a, T, 2, M> for CatmarkMesh<T, M> {
    type GeoElem = CatmarkPatch<T, M>;

    fn geo_elem(&'a self, elem: &Self::Elem) -> Self::GeoElem {
        CatmarkPatch::from_msh(self, elem)
    }

    fn vertex_iter(&'a self) -> impl Iterator<Item=Point<T, M>> {
        self.coords.iter().copied()
    }
}

impl <T: RealField, const M: usize> From<QuadVertexMesh<T, M>> for CatmarkMesh<T, M> {
    /// Turns a [`QuadVertexMesh<T,M>`] into a [`CatmarkMesh<T,M>`].
    ///
    /// The conversion moves the vector of vertex coordinates and does not re-allocate it.
    /// The [face one-rings](CatmarkPatchNodes) are computed for each `n` quadrilaterals and require reallocation.
    fn from(value: QuadVertexMesh<T, M>) -> Self {
        let face_one_rings = value.elems
            .iter()
            .map(|quad| CatmarkPatchNodes::find(&value, quad))
            .collect();
        CatmarkMesh::new(value.coords, face_one_rings)
    }
}

impl <T: RealField + Copy, const M: usize> From<CatmarkMesh<T, M>> for QuadVertexMesh<T, M> {
    /// Turns a [`CatmarkMesh<T,M>`] into a [`QuadVertexMesh<T,M>`].
    ///
    /// The conversion moves the vector of vertex coordinates and does not re-allocate it.
    /// The [quadrilateral faces](QuadNodes) require reallocation.
    fn from(value: CatmarkMesh<T, M>) -> Self {
        let faces: Vec<QuadNodes> = value.elems
            .iter()
            .map(|one_ring| one_ring.center_quad())
            .collect();
        QuadVertexMesh::new(value.coords, faces)
    }
}