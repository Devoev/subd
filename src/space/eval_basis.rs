use crate::space::basis::BasisFunctions;
use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, DimName, DimNameAdd, DimNameSum, Dyn, OMatrix, RealField, RowOVector, RowVector, Scalar, U1, U2};

/// Allocator for the [`Basis::NumComponents`] ✕ [`Basis::NumBasis`] matrix of basis evaluations.
pub trait EvalBasisAllocator<Basis: BasisFunctions>: Allocator<Basis::NumComponents, Basis::NumBasis> {}

impl <Basis: BasisFunctions> EvalBasisAllocator<Basis> for DefaultAllocator
    where DefaultAllocator: Allocator<Basis::NumComponents, Basis::NumBasis> {}

/// Pointwise evaluation of basis functions.
///
/// The set of all [basis functions](BasisFunctions) can be evaluated at arbitrary parametric points of type [`Self::Coord<T>`]
/// using [`Self::eval`].
pub trait EvalBasis<T: Scalar>: BasisFunctions + Sized
    where DefaultAllocator: EvalBasisAllocator<Self>
{
    /// Evaluates all basis functions at the parametric point `x`
    /// as the column-wise matrix `(b[1],...,b[n])`.
    fn eval(&self, x: Self::Coord<T>) -> OMatrix<T, Self::NumComponents, Self::NumBasis>;
}

/// Evaluation of arbitrary derivatives of basis functions.
///
/// The derivatives up to a given order `K` can be evaluated using [`Self::eval_derivs`].
pub trait EvalDerivs<T: RealField>: EvalBasis<T, NumComponents = U1>
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
    fn eval_derivs<const K: usize>(&self, x: Self::Coord<T>) -> OMatrix<T, DimNameSum<Const<K>, U1>, Self::NumBasis>
        where Const<K>: DimNameAdd<U1>,
              DefaultAllocator: Allocator<<Const<K> as DimNameAdd<U1>>::Output, Self::NumBasis>;
}

/// Allocator for basis evaluations and the [`Basis::ParametricDim`] ✕ [`Basis::NumBasis`] matrix of gradient evaluations.
pub trait EvalGradAllocator<Basis: BasisFunctions>:
    EvalBasisAllocator<Basis>
    + Allocator<Basis::ParametricDim, Basis::NumBasis> {}

impl <Basis: BasisFunctions> EvalGradAllocator<Basis> for DefaultAllocator
    where DefaultAllocator: EvalBasisAllocator<Basis> + Allocator<Basis::ParametricDim, Basis::NumBasis> {}

/// Evaluation of gradients of basis functions.
///
/// The gradient of all basis functions can be evaluated using [`Self::eval_grad`],
/// where each gradient is represented as a column vector.
pub trait EvalGrad<T: RealField> : EvalBasis<T, NumComponents = U1>
    where DefaultAllocator: EvalGradAllocator<Self>
{
    /// Evaluates the gradients of all basis functions at the parametric point `x`
    /// as the column-wise matrix `(grad b[1],...,grad b[n])`.
    fn eval_grad(&self, x: Self::Coord<T>) -> OMatrix<T, Self::ParametricDim, Self::NumBasis>;
}

/// Allocator for basis evaluations and the row vector of [`B::NumBasis`] scalar curl evaluations.
pub trait EvalScalarCurlAllocator<B: BasisFunctions>: EvalBasisAllocator<B> + Allocator<U1, B::NumBasis> {}

impl <B: BasisFunctions> EvalScalarCurlAllocator<B> for DefaultAllocator
    where DefaultAllocator: EvalBasisAllocator<B> + Allocator<U1, B::NumBasis> {}

/// Evaluation of scalar curls of basis functions.
///
/// The scalar curls of all basis functions can be evaluated using [`Self::eval_scalar_curl`].
/// The scalar curls are scalar valued, hence the result is stored in a row-vector.
pub trait EvalScalarCurl<T: RealField>: EvalBasis<T, NumComponents = U2>
    where DefaultAllocator: EvalScalarCurlAllocator<Self>
{
    /// Evaluates the scalar curl of all basis functions at the parametric point `x`
    /// as the column-wise matrix `(curl b[1],...,curl b[n])`.
    fn eval_scalar_curl(&self, x: Self::Coord<T>) -> RowOVector<T, Self::NumBasis>;
}