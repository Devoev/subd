use itertools::Itertools;
use nalgebra::{DMatrix, DVector, Point, RealField};
use num_traits::ToPrimitive;
use crate::cells::node::NodeIdx;
use crate::mesh::elem_vertex::ElemVertexMesh;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::Mesh;
use crate::subd::basis::{permutation_matrix, permutation_vec};
use crate::subd::matrices::build_extended_mats;
use crate::subd::patch::{CatmarkPatch, CatmarkPatchNodes};

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

impl <'a, T: RealField + Copy + ToPrimitive, const M: usize> Mesh<'a, T, (T, T), 2, M> for CatmarkMesh<T, M> {
    type GeoElem = CatmarkPatch<T, M>;

    fn geo_elem(&'a self, elem: Self::Elem) -> Self::GeoElem {
        CatmarkPatch::from_msh(self, elem)
    }
}

impl <T: RealField + Copy, const M: usize> CatmarkMesh<T, M> {
    /// Refines this mesh by applying Catmull-Clark subdivision once.
    pub fn refine(&mut self) {
        // Empty coordinate and patch vectors
        let mut patches = Vec::<CatmarkPatchNodes>::new();
        let mut coords = Vec::<Point<T, M>>::new();

        // Refine each patch iteratively
        for patch in &self.elems {
            // Get control points
            let geo_patch = CatmarkPatch::from_msh(self, patch);
            let p = geo_patch.coords();

            // Get valence
            let n = match patch {
                CatmarkPatchNodes::Regular(_) => 4,
                CatmarkPatchNodes::Boundary(_) => continue,
                CatmarkPatchNodes::Corner(_) => continue,
                CatmarkPatchNodes::Irregular(_, n) => *n
            };

            // Build subdivision matrix
            let (_, a_bar) = build_extended_mats::<T>(n);

            // todo: this certainly doesn't work, because a lot of new control points
            //  get computed multiple times.
            //  This either has to be detected, or a different subdivision matrix is somehow used,
            //  that doesn't compute the same values again

            // Refine control points
            let p = a_bar * p;
            let idx = coords.len();
            let num_new = p.shape().0;
            let idx_vec = DVector::from_iterator(num_new, idx..idx+num_new);
            coords.extend(p.row_iter().map(|p| Point::from(p.transpose())));

            // todo: the 4th patch (i.e. the irregular one) is missing
            // Refine elements
            for k in 0..3 {
                let perm = permutation_vec(k, n);
                let perm = permutation_matrix(perm, n);
                let pk = perm * &idx_vec;
                let nodes: [NodeIdx; 16] = pk.iter().map(|n| NodeIdx(*n)).collect_array().unwrap();
                patches.push(CatmarkPatchNodes::Regular(nodes))
            }
        }

        // Update data
        self.coords = coords;
        self.elems = patches;
    }
}