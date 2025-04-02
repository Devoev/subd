//! Functions for isogeometric analysis on subdivision surfaces.

use crate::subd::mesh::QuadMesh;
use crate::subd::patch::Patch;
use nalgebra::{DVector, Point2, RealField};
use num_traits::ToPrimitive;

/// A discrete scalar potential (i.e. a `0`-form) in IGA, represented by coefficients.
#[derive(Clone, Debug)]
pub struct IgaFn<'a, T: RealField> {
    /// The coefficient vector defining this function.
    coeffs: DVector<T>,
    /// The mesh this function is defined on.
    msh: &'a QuadMesh<T>
}

impl<T: RealField + Copy + ToPrimitive> IgaFn<'_, T> {

    /// Constructs a new [`IgaFn`] from the given `msh` and `coeffs`.
    pub fn new(msh: &QuadMesh<T>, coeffs: DVector<T>) -> IgaFn<T> {
        IgaFn { coeffs, msh }
    }

    /// Constructs a new [`IgaFn`] by interpolation of the given function `f`.
    /// The coefficients are calculated by point-wise evaluation at the nodes of the given `msh`.
    pub fn from_fn(msh: &QuadMesh<T>, f: impl Fn(Point2<T>) -> T) -> IgaFn<T> {
        let coeffs = DVector::from_iterator(msh.num_nodes(), msh.nodes.iter().map(|&p| f(p)));
        IgaFn { coeffs, msh }
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