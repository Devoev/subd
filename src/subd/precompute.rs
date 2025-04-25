use crate::subd::patch::Patch;
use crate::subd::quad::GaussLegendrePatch;
use crate::subd::surface::{Basis, BasisGrad, Jacobian, ParametricMap, Parametrization};
use nalgebra::{Matrix2, RealField};
use num_traits::ToPrimitive;
use std::slice::Iter;
use std::vec::IntoIter;

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

    /// Iterates over the evaluated values.
    pub fn iter(&self) -> Iter<'_, F::Eval> {
        self.0.iter()
    }
}

impl<T, F: ParametricMap<T>> IntoIterator for QuadEval<T, F> {
    type Item = F::Eval;
    type IntoIter = IntoIter<F::Eval>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T, F: ParametricMap<T>> IntoIterator for &'a QuadEval<T, F> {
    type Item = &'a F::Eval;
    type IntoIter = Iter<'a, F::Eval>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// todo: move these functions to Jacobian struct or elsewhere
impl<T: RealField + Copy + ToPrimitive> QuadEval<T, Jacobian<'_, T>> {
    /// Returns an iterator over all absolute values of the determinant of the Jacobian matrices.
    pub fn abs_det(&self) -> impl Iterator<Item=T> + '_ {
        self.0.iter().map(|d_phi| d_phi.determinant().abs())
    }

    /// Returns an iterator over the inverse Gram matrices, i.e. `(Jᐪ·J)⁻¹`.
    // todo: also precompute this
    pub fn gram_inv(&self) -> impl Iterator<Item=Matrix2<T>> + '_ {
        self.0.iter().map(|d_phi| (d_phi.transpose() * d_phi).try_inverse().unwrap())
    }
}

/// Evaluated quantities on a single [`Patch`] at each quadrature point.
pub struct PatchEval<'a, T: RealField + Copy + ToPrimitive> {
    /// The patch.
    pub patch: &'a Patch<'a, T>,

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
            quad,
            patch
        }
    }
}

/// Evaluated quantities on all patches of a [`QuadMesh`].
pub struct MeshEval<'a, T: RealField + Copy + ToPrimitive>(Vec<PatchEval<'a, T>>);

impl<'a, T: RealField + Copy + ToPrimitive> MeshEval<'a, T> {
    /// Constructs a new [`MeshEval`] on the `patches` using the given quadrature rule `quad`.
    pub fn from(patches: &'a [Patch<T>], quad: &GaussLegendrePatch) -> Self {
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