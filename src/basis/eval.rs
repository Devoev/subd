use crate::basis::traits::Basis;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, DimNameAdd, DimNameSum, Dyn, OMatrix, RealField, U1};

/// Allocator for the [`B::NumComponents`] ✕ [`B::NumBasis`] matrix of basis evaluations.
pub trait EvalBasisAllocator<B: Basis>: Allocator<B::NumComponents, B::NumBasis> {}

impl <B: Basis> EvalBasisAllocator<B> for DefaultAllocator
    where DefaultAllocator: Allocator<B::NumComponents, B::NumBasis> {}

/// Set of basis functions which can be evaluated at arbitrary points using [`Self::eval`].
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
pub trait EvalBasis<T: RealField, X>: Basis + Sized
    where DefaultAllocator: EvalBasisAllocator<Self>
{
    /// Evaluates all basis functions at the parametric point `x`
    /// as the column-wise matrix `(b[1],...,b[n])`.
    fn eval(&self, x: X) -> OMatrix<T, Self::NumComponents, Self::NumBasis>;
}

/// Scalar, differentiable basis functions.
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
pub trait EvalDerivs<T: RealField, X>: EvalBasis<T, X, NumComponents = U1>
    where DefaultAllocator: EvalBasisAllocator<Self>
{
    /// Evaluates the value and the first [`K`] derivatives of all basis functions
    /// at the parametric point `x` as the matrix
    /// ```text
    ///   ┌                     ┐
    ///   │  b[1]   ...   b[n]  │
    ///   │  db[1]  ...  db[n]  │
    ///   │ db[1]^2 ... db[n]^2 │
    ///   │ db[1]^3 ... db[n]^3 │
    ///   │                     │
    ///   └                     ┘
    /// ```
    /// where each row corresponds to the `i-1`-th derivative of basis functions.
    fn eval_derivs<const K: usize>(&self, x: X) -> OMatrix<T, DimNameSum<Const<K>, U1>, Self::NumBasis>
        where Const<K>: DimNameAdd<U1>,
              DefaultAllocator: Allocator<<Const<K> as DimNameAdd<U1>>::Output, Self::NumBasis>;
}

/// Allocator for basis evaluations and the [`D`] ✕ [`B::NumBasis`] matrix of gradient evaluations.
pub trait EvalGradAllocator<B: Basis, const D: usize>:
    EvalBasisAllocator<B>
    + Allocator<Const<D>, B::NumBasis> {}

impl <B: Basis, const D: usize> EvalGradAllocator<B, D> for DefaultAllocator
    where DefaultAllocator: EvalBasisAllocator<B> + Allocator<Const<D>, B::NumBasis> {}

/// Basis functions for `H(grad)`-conforming spaces (i.e. nodal functions).
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
/// - [`D`] : Dimension of the reference domain.
pub trait EvalGrad<T: RealField, X, const D: usize> : EvalBasis<T, X, NumComponents = U1>
    where DefaultAllocator: EvalGradAllocator<Self, D>
{
    /// Evaluates the gradients of all basis functions at the parametric point `x`
    /// as the column-wise matrix `(grad b[1],...,grad b[n])`.
    fn eval_grad(&self, x: X) -> OMatrix<T, Const<D>, Self::NumBasis>;
}