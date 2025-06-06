use nalgebra::{Const, DefaultAllocator, Dim, DimName, DimNameAdd, DimNameSum, Dyn, OMatrix, RealField, U1};
use nalgebra::allocator::Allocator;

/// Set of [`Self::num_basis`] basis functions.
pub trait NumBasis {
    /// Returns the number of basis functions in this set.
    fn num_basis(&self) -> usize;
}

/// Set of basis functions which can be evaluated at arbitrary points using [`Self::eval`].
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
pub trait Basis<T: RealField, X>: NumBasis
    where DefaultAllocator: Allocator<Self::NumComponents, Self::NumBasis>
{
    /// Number of basis functions.
    type NumBasis: Dim;

    /// Number of components for each basis function.
    /// For scalar valued functions equal to `1`,
    /// for vector valued functions equal to the dimension of the parametric domain.
    type NumComponents: DimName;

    /// Evaluates all basis functions at the parametric point `x`
    /// as the column-wise matrix `(b[1],...,b[n])`.
    fn eval(&self, x: X) -> OMatrix<T, Self::NumComponents, Self::NumBasis>;
}

/// Scalar, differentiable basis functions.
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
pub trait DiffBasis<T: RealField, X>: Basis<T, X, NumComponents = U1>
    where DefaultAllocator: Allocator<U1, Self::NumBasis>
{
    /// Evaluates the value and the first [`K`] derivatives of all basis functions
    /// at the parametric point `x` as the matrix
    /// ```text
    ///   ┌                   ┐
    ///   │ b[1]   ...   b[n] │
    ///   │ b'[1]  ...  b'[n] │
    ///   │ b''[1] ... b''[n] │
    ///   │                   │
    ///   └                   ┘
    /// ```
    /// where each row corresponds to the `i-1`-th derivative of basis functions.
    fn eval_derivs<const K: usize>(&self, x: X) -> OMatrix<T, DimNameSum<Const<K>, U1>, Self::NumBasis>
        where Const<K>: DimNameAdd<U1>,
              DefaultAllocator: Allocator<<Const<K> as DimNameAdd<U1>>::Output, Self::NumBasis>;
}

/// Basis functions for `H(grad)`-conforming spaces (i.e. nodal functions).
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
/// - [`D`] : Dimension of the reference domain.
pub trait HgradBasis<T: RealField, X, const D: usize> : Basis<T, X, NumComponents = U1>
    where DefaultAllocator: Allocator<U1, Self::NumBasis>
{
    /// Evaluates the gradients of all basis functions at the parametric point `x` 
    /// as the column-wise matrix `(grad b[1],...,grad b[n])`.
    fn eval_grad(&self, x: X) -> OMatrix<T, Const<D>, Dyn>;
}