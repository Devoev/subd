//! Functions for isogeometric analysis on subdivision surfaces.

use std::iter::zip;
use crate::subd::mesh::QuadMesh;
use crate::subd::patch::Patch;
use crate::subd::precompute::{BasisEval, GradEval, JacobianEval, PointEval};
use itertools::{izip, Itertools};
use nalgebra::{DMatrix, DVector, Dyn, Matrix2, OMatrix, Point2, RealField, U2};
use nalgebra_sparse::CooMatrix;
use num_traits::ToPrimitive;
use crate::subd::surface::ParametricMap;

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
        let indices = patch.connectivity.as_slice();
        let c = DVector::from_iterator(indices.len(), indices.iter().map(|&i| self.coeffs[i]));
        let b = patch.basis().eval(u, v);
        c.dot(&b)
    }
}

/// Builds the discrete IGA operator `∫ fv dx` using `num_quad` quadrature points.
pub fn op_f_v<T: RealField + Copy + ToPrimitive>(
    msh: &QuadMesh<T>, f: impl Fn(Point2<T>) -> T + Clone,
    basis_eval: &Vec<BasisEval<T>>,
    point_eval: &Vec<PointEval<T>>,
    jacobian_eval: &Vec<JacobianEval<T>>
) -> DVector<T> {
    let mut fi = DVector::<T>::zeros(msh.num_nodes());
    for (patch, b_eval, p_eval, j_eval) in izip!(msh.patches(), basis_eval, point_eval, jacobian_eval) {
        let fi_local = op_f_v_local(&patch, f.clone(), b_eval, p_eval, j_eval);
        let indices = patch.connectivity.as_slice();
        for (idx_local, &idx) in indices.iter().enumerate() {
            fi[idx] += fi_local[idx_local];
        }
    }
    fi
}

/// Builds the local discrete IGA operator `∫ fv dx` of the given `patch` using `num_quad` quadrature points.
fn op_f_v_local<T: RealField + Copy + ToPrimitive>(
    patch: &Patch<T>, f: impl Fn(Point2<T>) -> T,
    basis_eval: &BasisEval<T>,
    point_eval: &PointEval<T>,
    jacobian_eval: &JacobianEval<T>
) -> DVector<T> {

    let fv_pullback = |b: &DVector<T>, p: Point2<T>, j: usize| b[j] * f(p);

    let num_basis = patch.connectivity.as_slice().len();
    let fj = (0..num_basis).map(|j| {
        // Calculate integrand f * bj
        let fj = zip(&basis_eval.quad_to_basis, &point_eval.quad_to_points)
            .map(|(b, &p)| fv_pullback(b, p, j)).collect();
        
        // Evaluate integral
        jacobian_eval.quad.integrate_pullback(fj, jacobian_eval)
    });
    DVector::from_iterator(num_basis, fj)
}

/// Builds the discrete IGA operator `∫ grad u · grad v dx` using `num_quad` quadrature points.
pub fn op_gradu_gradv<T: RealField + Copy + ToPrimitive>(msh: &QuadMesh<T>, grad_eval: &Vec<GradEval<T>>, jacobian_eval: &Vec<JacobianEval<T>>) -> CooMatrix<T> {
    let mut kij = CooMatrix::<T>::new(msh.num_nodes(), msh.num_nodes());

    for (patch, grad_b_eval, j_eval) in izip!(msh.patches(), grad_eval, jacobian_eval) {
        let kij_local = op_gradu_gradv_local(&patch, grad_b_eval, j_eval);
        let indices = patch.connectivity.as_slice().iter().enumerate();
        for ((i_local, &i), (j_local, &j)) in indices.clone().cartesian_product(indices) {
            kij.push(i, j, kij_local[(i_local, j_local)]);
        }
    }

    kij
}

/// Builds the local discrete IGA operator `∫ grad u · grad v dx`
/// of the given `patch` using `num_quad` quadrature points.
fn op_gradu_gradv_local<T: RealField + Copy + ToPrimitive>(patch: &Patch<T>, grad_eval: &GradEval<T>, jacobian_eval: &JacobianEval<T>) -> DMatrix<T> {
    let gradu_gradv_pullback = |grad_b: &OMatrix<T, Dyn, U2>, g_inv: &Matrix2<T>, i: usize, j: usize| {
        // Get gradients
        let grad_bi = grad_b.row(i);
        let grad_bj = grad_b.row(j);

        // Calculate integrand
        (grad_bi * g_inv * grad_bj.transpose()).x
    };

    let num_basis = patch.connectivity.as_slice().len();
    let kij = (0..num_basis).cartesian_product(0..num_basis)
        .map(|(i, j)| {
            // Calculate integrand bi * bj
            let fij = zip(&grad_eval.quad_to_grad, jacobian_eval.gram_inv())
                .map(|(grad_b, g_inv)| gradu_gradv_pullback(grad_b, &g_inv, i, j)).collect();

            // Evaluate integral
            jacobian_eval.quad.integrate_pullback(fij, jacobian_eval)
        });

    DMatrix::from_iterator(num_basis, num_basis, kij)
}

/// Builds the discrete IGA operator `∫ uv dx` using the precomputed basis functions `basis_eval`.
pub fn op_u_v<T: RealField + Copy + ToPrimitive>(msh: &QuadMesh<T>, basis_eval: &Vec<BasisEval<T>>, jacobian_eval: &Vec<JacobianEval<T>>) -> CooMatrix<T> {
    let mut mij = CooMatrix::<T>::zeros(msh.num_nodes(), msh.num_nodes());

    for (patch, b_eval, j_eval) in izip!(msh.patches(), basis_eval, jacobian_eval) {
        let mij_local = op_u_v_local(&patch, b_eval, j_eval);
        let indices = patch.connectivity.as_slice().iter().enumerate();
        for ((i_local, &i), (j_local, &j)) in indices.clone().cartesian_product(indices) {
            mij.push(i, j, mij_local[(i_local, j_local)]);
        }
    }

    mij
}

/// Builds the local discrete IGA operator `∫ uv dx`
/// of the given `patch` using the precomputed basis functions `basis_eval`.
fn op_u_v_local<T: RealField + Copy + ToPrimitive>(patch: &Patch<T>, basis_eval: &BasisEval<T>, jacobian_eval: &JacobianEval<T>) -> DMatrix<T>  {
    let uv_pullback = |b: &DVector<T>, i: usize, j: usize| {
        // Eval basis
        let bi = b[i];
        let bj = b[j];

        // Calculate integrand
        bi * bj
    };

    let num_basis = patch.connectivity.as_slice().len();
    let mij = (0..num_basis).cartesian_product(0..num_basis)
        .map(|(i, j)| {
            // Calculate integrand bi * bj
            let fij = basis_eval.quad_to_basis.iter().map(|b| uv_pullback(b, i, j)).collect();

            // Evaluate integral
            jacobian_eval.quad.integrate_pullback(fij, jacobian_eval)
        });
    DMatrix::from_iterator(num_basis, num_basis, mij)
}
