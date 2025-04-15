use crate::subd::mesh::QuadMesh;
use crate::subd::patch::Patch;
use crate::subd::quad::GaussLegendrePatch;
use nalgebra::{DVector, Matrix2, RealField};
use num_traits::ToPrimitive;

// todo: make generic struct like QuadEval<T>
//  and maybe differentiate between patch eval and global eval

/// Evaluated basis functions for each quadrature point of a patch.
#[derive(Debug, Clone)]
pub struct BasisEval<T: RealField> {
    /// Vector of evaluated basis functions stored in a [`DVector`] for each quadrature point.
    pub quad_to_basis: Vec<DVector<T>>,

    /// Quadrature rule used for integration.
    pub quad: GaussLegendrePatch
    // todo: remove this property
}

impl<T: RealField + Copy + ToPrimitive> BasisEval<T> {

    /// Constructs a new [`BasisEval`] for the given `patch` using the quadrature rule `quad`,
    /// by evaluating each basis function at every quadrature point in `quad.nodes()`.
    pub fn from(patch: &Patch<T>, quad: GaussLegendrePatch) -> BasisEval<T> {
        Self {
            quad_to_basis: quad.nodes()
                .map(|(u, v)| {
                    let u = T::from_f64(u).unwrap();
                    let v = T::from_f64(v).unwrap();
                    patch.eval_basis(u, v)
                })
                .collect(),
            quad
        }
    }

    // todo: add docs
    pub fn from_mesh(msh: &QuadMesh<T>, quad: GaussLegendrePatch) -> Vec<BasisEval<T>> {
        msh.patches()
            .map(|patch| BasisEval::from(&patch, quad.clone()))
            .collect()
    }
}

/// Evaluated Jacobian for each quadrature point of a patch.
#[derive(Debug, Clone)]
pub struct JacobianEval<T: RealField> {
    /// Vector of evaluated Jacobian matrices for each quadrature point.
    pub quad_to_jacobian: Vec<Matrix2<T>>
}

impl<T: RealField + Copy + ToPrimitive> JacobianEval<T> {
    /// Constructs a new [`JacobianEval`] for the given `patch` using the quadrature rule `quad`,
    /// by evaluating the Jacobian matrix at every quadrature point in `quad.nodes()`.
    pub fn from(patch: &Patch<T>, quad: GaussLegendrePatch) -> JacobianEval<T> {
        Self {
            quad_to_jacobian: quad.nodes()
                .map(|(u, v)| {
                    let u = T::from_f64(u).unwrap();
                    let v = T::from_f64(v).unwrap();
                    patch.eval_jacobian(u, v)
                })
                .collect(),
        }
    }

    // todo: add docs
    pub fn from_mesh(msh: &QuadMesh<T>, quad: GaussLegendrePatch) -> Vec<JacobianEval<T>> {
        msh.patches()
            .map(|patch| JacobianEval::from(&patch, quad.clone()))
            .collect()
    }

    /// Returns an iterator over all absolute values of the determinant of the Jacobian matrices.
    pub fn abs_det(&self) -> impl Iterator<Item=T> + '_ {
        self.quad_to_jacobian.iter().map(|d_phi| d_phi.determinant().abs())
    }
}