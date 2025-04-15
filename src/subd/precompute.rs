use crate::subd::mesh::QuadMesh;
use crate::subd::patch::Patch;
use crate::subd::quad::GaussLegendrePatch;
use nalgebra::{DVector, RealField};
use num_traits::ToPrimitive;

/// Evaluated basis functions for each patch in a mesh.
#[derive(Debug, Clone)]
pub struct BasisEval<T: RealField> {
    /// Vector of evaluated basis functions for each patch.
    pub patch_to_eval: Vec<BasisEvalPatch<T>>
}

impl<T: RealField + Copy + ToPrimitive> BasisEval<T> {

    /// Constructs a new [`BasisEval`] for the given mesh using the quadrature rule `quad`.
    pub fn from(msh: &QuadMesh<T>, quad: GaussLegendrePatch) -> BasisEval<T> {
        BasisEval {
            patch_to_eval: msh.patches()
                .map(|patch| BasisEvalPatch::from(&patch, quad.clone()))
                .collect()
        }
    }
}

/// Evaluated basis functions for each quadrature point of a patch.
#[derive(Debug, Clone)]
pub struct BasisEvalPatch<T: RealField> {
    /// Vector of evaluated basis functions stored in a [`DVector`] for each quadrature point.
    pub quad_to_basis: Vec<DVector<T>>
}

impl<T: RealField + Copy + ToPrimitive> BasisEvalPatch<T> {

    /// Constructs a new [`BasisEvalPatch`] for the given `patch` using the quadrature rule `quad`,
    /// by evaluating each basis function at every quadrature point in `quad.nodes()`.
    pub fn from(patch: &Patch<T>, quad: GaussLegendrePatch) -> BasisEvalPatch<T> {
        Self {
            quad_to_basis: quad.nodes()
                .map(|(u, v)| {
                    let u = T::from_f64(u).unwrap();
                    let v = T::from_f64(v).unwrap();
                    patch.eval_basis(u, v)
                })
                .collect()
        }
    }
}