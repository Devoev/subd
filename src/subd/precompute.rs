use crate::subd::mesh::QuadMesh;
use crate::subd::patch::Patch;
use crate::subd::quad::GaussLegendrePatch;
use crate::subd::surface::{Basis, BasisGrad, Jacobian, ParametricMap, Parametrization};
use nalgebra::{DVector, Dyn, Matrix2, OMatrix, Point2, RealField, U2};
use num_traits::ToPrimitive;
use std::slice::Iter;

/// Evaluated parametric maps at each quadrature point.
pub struct QuadEval<T, F: ParametricMap<T>>(pub Vec<F::Eval>);

impl<T: RealField, F: ParametricMap<T>> QuadEval<T, F> {
    /// Constructs a new [`QuadEval`] using the quadrature rule `quad`,
    /// by evaluating the given `map` at every quadrature point in `quad.nodes()`.
    pub fn from(map: F, quad: &GaussLegendrePatch) -> QuadEval<T, F> {
        Self(
            quad.nodes()
                .map(|(u, v)| {
                    let u = T::from_f64(u).unwrap();
                    let v = T::from_f64(v).unwrap();
                    map.eval(u, v)
                })
                .collect()
        )
    }

    /// Constructs a vector of [`QuadEval`] for each map in the given iterator.
    pub fn from_iterator(maps: impl Iterator<Item=F>, quad: &GaussLegendrePatch) -> Vec<QuadEval<T, F>> {
        maps.map(|f| QuadEval::from(f, quad)).collect()
    }
}

/// Evaluated quantities on a single [`Patch`] at each quadrature point.
pub struct PatchEval<'a, T: RealField + Copy + ToPrimitive> {
    /// Quadrature rule on the patch.
    pub quad: GaussLegendrePatch,

    /// Evaluated basis functions.
    pub basis: QuadEval<T, Basis<'a, T>>,

    /// Evaluated gradients of basis functions.
    pub basis_grad: QuadEval<T, BasisGrad<'a, T>>,

    /// Evaluated parametrization.
    pub points: QuadEval<T, Parametrization<'a, T>>,

    /// Evaluated Jacobian.
    pub jacobian: QuadEval<T, Jacobian<'a, T>>,
}

impl<'a, T: RealField + Copy + ToPrimitive> PatchEval<'a, T> {
    /// Constructs a new [`PatchEval`] on `patch` using the given quadrature rule `quad`.
    pub fn from(patch: &'a Patch<T>, quad: GaussLegendrePatch) -> Self {
        PatchEval {
            basis: QuadEval::from(patch.basis(), &quad),
            basis_grad: QuadEval::from(patch.basis_grad(), &quad),
            points: QuadEval::from(patch.parametrization(), &quad),
            jacobian: QuadEval::from(patch.jacobian(), &quad),
            quad
        }
    }
}

/// Evaluated quantities on all patches of a [`QuadMesh`].
pub struct MeshEval<'a, T: RealField + Copy + ToPrimitive>(Vec<PatchEval<'a, T>>);

impl<'a, T: RealField + Copy + ToPrimitive> MeshEval<'a, T> {
    /// Constructs a new [`MeshEval`] on the `patches` using the given quadrature rule `quad`.
    pub fn from(patches: &'a Vec<Patch<T>>, quad: &GaussLegendrePatch) -> Self {
        // todo: add a PatchMesh struct or else as replacement for patches vec
        MeshEval(
            patches.iter().map(|patch| PatchEval::from(patch, quad.clone())).collect(),
        )
    }

    /// Iterates through the evaluated patches.
    pub fn patch_iter(&self) -> Iter<PatchEval<'a, T>> {
        self.0.iter()
    }
}

/// Evaluated basis functions for each quadrature point of a patch.
#[derive(Debug, Clone)]
pub struct BasisEval<T: RealField> {
    /// Vector of evaluated basis functions stored in a [`DVector`] for each quadrature point.
    pub quad_to_basis: Vec<DVector<T>>,
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
                    patch.basis().eval(u, v)
                })
                .collect()
        }
    }

    // todo: add docs
    pub fn from_mesh(msh: &QuadMesh<T>, quad: GaussLegendrePatch) -> Vec<BasisEval<T>> {
        msh.patches()
            .map(|patch| BasisEval::from(&patch, quad.clone()))
            .collect()
    }
}

/// Evaluated gradients for each quadrature point of a patch.
#[derive(Debug, Clone)]
pub struct GradEval<T: RealField> {
    /// Vector of evaluated gradients for each quadrature point.
    pub quad_to_grad: Vec<OMatrix<T, Dyn, U2>>
}

impl<T: RealField + Copy + ToPrimitive> GradEval<T> {
    /// Constructs a new [`GradEval`] for the given `patch` using the quadrature rule `quad`,
    /// by evaluating the gradients of the basis functions at every quadrature point in `quad.nodes()`.
    pub fn from(patch: &Patch<T>, quad: GaussLegendrePatch) -> GradEval<T> {
        Self {
            quad_to_grad: quad.nodes()
                .map(|(u, v)| {
                    let u = T::from_f64(u).unwrap();
                    let v = T::from_f64(v).unwrap();
                    patch.basis_grad().eval(u, v)
                })
                .collect(),
        }
    }

    // todo: add docs
    pub fn from_mesh(msh: &QuadMesh<T>, quad: GaussLegendrePatch) -> Vec<GradEval<T>> {
        msh.patches()
            .map(|patch| GradEval::from(&patch, quad.clone()))
            .collect()
    }
}

/// Evaluated Jacobian for each quadrature point of a patch.
#[derive(Debug, Clone)]
pub struct JacobianEval<T: RealField> {
    /// Vector of evaluated Jacobian matrices for each quadrature point.
    pub quad_to_jacobian: Vec<Matrix2<T>>,

    /// Quadrature rule used for integration.
    pub quad: GaussLegendrePatch
    // todo: remove this property
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
                    patch.jacobian().eval(u, v)
                })
                .collect(),
            quad
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

    /// Returns an iterator over the inverse Gram matrices, i.e. `(Jᐪ·J)⁻¹`.
    // todo: also precompute this
    pub fn gram_inv(&self) -> impl Iterator<Item=Matrix2<T>> + '_ {
        self.quad_to_jacobian.iter().map(|d_phi| (d_phi.transpose() * d_phi).try_inverse().unwrap())
    }
}

/// Evaluated patch points for each quadrature point of a patch.
#[derive(Debug, Clone)]
pub struct PointEval<T: RealField> {
    /// Vector of evaluated points for each quadrature point.
    pub quad_to_points: Vec<Point2<T>>,
}

impl<T: RealField + Copy + ToPrimitive> PointEval<T> {

    /// Constructs a new [`PointEval`] for the given `patch` using the quadrature rule `quad`,
    /// by evaluating the patch at every quadrature point in `quad.nodes()`.
    pub fn from(patch: &Patch<T>, quad: GaussLegendrePatch) -> PointEval<T> {
        Self {
            quad_to_points: quad.nodes()
                .map(|(u, v)| {
                    let u = T::from_f64(u).unwrap();
                    let v = T::from_f64(v).unwrap();
                    patch.parametrization().eval(u, v)
                })
                .collect()
        }
    }

    // todo: add docs
    pub fn from_mesh(msh: &QuadMesh<T>, quad: GaussLegendrePatch) -> Vec<PointEval<T>> {
        msh.patches()
            .map(|patch| PointEval::from(&patch, quad.clone()))
            .collect()
    }
}

