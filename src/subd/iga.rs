//! Functions for isogeometric analysis on subdivision surfaces.

use std::iter::zip;
use itertools::Itertools;
use crate::subd::mesh::QuadMesh;
use crate::subd::patch::Patch;
use nalgebra::{DMatrix, DVector, Point2, RealField};
use nalgebra_sparse::CooMatrix;
use num_traits::ToPrimitive;
use crate::subd::precompute::{BasisEval, BasisEvalPatch};

/// A discrete scalar potential (i.e. a `0`-form) in IGA, represented by coefficients.
#[derive(Clone, Debug)]
pub struct IgaFn<'a, T: RealField> {
    /// The coefficient vector defining this function.
    pub coeffs: DVector<T>,
    /// The mesh this function is defined on.
    msh: &'a QuadMesh<T>
}

impl<T: RealField + Copy + ToPrimitive> IgaFn<'_, T> {

    /// Constructs a new [`IgaFn`] from the given `msh` and `coeffs`.
    pub fn new(msh: &QuadMesh<T>, coeffs: DVector<T>) -> IgaFn<T> {
        IgaFn { coeffs, msh }
    }

    /// Constructs a new [`IgaFn`] by interpolation of the given function `f: Ω ⟶ ℝ`.
    /// The coefficients are calculated by point-wise evaluation at the nodes of the given `msh`.
    pub fn from_fn(msh: &QuadMesh<T>, f: impl Fn(Point2<T>) -> T) -> IgaFn<T> {
        let coeffs = DVector::from_iterator(msh.num_nodes(), msh.nodes.iter().map(|&p| f(p)));
        IgaFn { coeffs, msh }
    }

    /// Constructs a new [`IgaFn`] by interpolation of the given boundary function `g: ∂Ω ⟶ ℝ`.
    /// The coefficients are calculated by point-wise evaluation at the boundary nodes of the given `msh`.
    pub fn from_bnd_fn(msh: &QuadMesh<T>, g: impl Fn(Point2<T>) -> T) -> IgaFn<T> {
        let nodes_bnd = msh.nodes.iter()
            .enumerate()
            .filter_map(|(i, node)| msh.is_boundary_node(i).then_some(node));

        let coeffs = nodes_bnd.map(|&p| g(p)).collect_vec();
        IgaFn { coeffs: DVector::from_vec(coeffs), msh }
    }

    /// Evaluates the pullback `f(phi)` at the parametric point `(u,v)`,
    /// where `phi` is the parametrization of the given `patch`.
    pub fn eval_pullback(&self, patch: &Patch<T>, u: T, v: T) -> T {
        // Get the indices of the control points corresponding to the patch
        let indices = patch.nodes();
        let c = DVector::from_iterator(indices.len(), indices.iter().map(|&i| self.coeffs[i]));
        let b = patch.eval_basis(u, v);
        c.dot(&b)
    }
}

/// Builds the discrete IGA operator `∫ fv dx` using `num_quad` quadrature points.
pub fn op_f_v<T: RealField + Copy + ToPrimitive>(msh: &QuadMesh<T>, f: impl Fn(Point2<T>) -> T + Clone, num_quad: usize) -> DVector<T> {
    let mut fi = DVector::<T>::zeros(msh.num_nodes());
    for patch in msh.patches() {
        let fi_local = op_f_v_local(&patch, f.clone(), num_quad);
        let indices = patch.nodes();
        for (idx_local, idx) in indices.into_iter().enumerate() {
            fi[idx] += fi_local[idx_local];
        }
    }
    fi
}

/// Builds the local discrete IGA operator `∫ fv dx` of the given `patch` using `num_quad` quadrature points.
fn op_f_v_local<T: RealField + Copy + ToPrimitive>(patch: &Patch<T>, f: impl Fn(Point2<T>) -> T, num_quad: usize) -> DVector<T> {
    // fixme: this is really expensive, because the whole basis gets evaluated multiple times. Change that
    let fv_pullback = |u: T, v: T| patch.eval_basis(u, v) * f(patch.eval(u, v));
    let num_basis = patch.nodes().len();
    let fi = (0..num_basis).map(|i| patch.integrate_pullback(|u, v| fv_pullback(u, v)[i], num_quad));
    DVector::from_iterator(num_basis, fi)
}

/// Builds the discrete IGA operator `∫ grad u · grad v dx` using `num_quad` quadrature points.
pub fn op_gradu_gradv<T: RealField + Copy + ToPrimitive>(msh: &QuadMesh<T>, num_quad: usize) -> CooMatrix<T> {
    let mut aij = CooMatrix::<T>::new(msh.num_nodes(), msh.num_nodes());

    for patch in msh.patches() {
        let aij_local = op_gradu_gradv_local(&patch, num_quad);
        let indices = patch.nodes().into_iter().enumerate();
        for ((i_local, i), (j_local, j)) in indices.clone().cartesian_product(indices) {
            aij.push(i, j, aij_local[(i_local, j_local)]);
        }
    }

    aij
}

/// Builds the local discrete IGA operator `∫ grad u · grad v dx`
/// of the given `patch` using `num_quad` quadrature points.
fn op_gradu_gradv_local<T: RealField + Copy + ToPrimitive>(patch: &Patch<T>, num_quad: usize) -> DMatrix<T> {
    // fixme: this is really expensive, because the whole basis gets evaluated multiple times. Change that
    let gradu_gradv_pullback = |u: T, v: T, i: usize, j: usize| {
        // Get gradients
        let grad_b = patch.eval_basis_grad(u, v);
        let grad_bi = grad_b.row(i);
        let grad_bj = grad_b.row(j);

        // Calculate inverse gram matrix
        let d_phi = patch.eval_jacobian(u, v);
        let g = d_phi.transpose() * d_phi;
        let g_inv = g.try_inverse().unwrap();

        // Calculate integrand
        (grad_bi * g_inv * grad_bj.transpose()).x
    };

    let num_basis = patch.nodes().len();
    let aij = (0..num_basis).cartesian_product(0..num_basis)
        .map(|(i, j)| patch.integrate_pullback(|u, v| gradu_gradv_pullback(u, v, i, j), num_quad));

    DMatrix::from_iterator(num_basis, num_basis, aij)
}

/// Builds the discrete IGA operator `∫ uv dx` using the precomputed basis functions `basis_eval`.
pub fn op_u_v<T: RealField + Copy + ToPrimitive>(msh: &QuadMesh<T>, basis_eval: &BasisEval<T>) -> CooMatrix<T> {
    let mut bij = CooMatrix::<T>::zeros(msh.num_nodes(), msh.num_nodes());

    for (patch, b_eval) in zip(msh.patches(), &basis_eval.patch_to_eval) {
        let bij_local = op_u_v_local(&patch, b_eval);
        let indices = patch.nodes().into_iter().enumerate();
        for ((i_local, i), (j_local, j)) in indices.clone().cartesian_product(indices) {
            bij.push(i, j, bij_local[(i_local, j_local)]);
        }
    }

    bij
}

/// Builds the local discrete IGA operator `∫ uv dx`
/// of the given `patch` using the precomputed basis functions `basis_eval`.
fn op_u_v_local<T: RealField + Copy + ToPrimitive>(patch: &Patch<T>, basis_eval: &BasisEvalPatch<T>) -> DMatrix<T>  {
    let uv_pullback = |b: &DVector<T>, i: usize, j: usize| {
        // Eval basis
        let bi = b[i];
        let bj = b[j];

        // Calculate integrand
        bi * bj
    };

    let num_basis = patch.nodes().len();
    let bij = (0..num_basis).cartesian_product(0..num_basis)
        .map(|(i, j)| basis_eval.integrate_pullback(|b| uv_pullback(b, i, j)));

    DMatrix::from_iterator(num_basis, num_basis, bij)
}
