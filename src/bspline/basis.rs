use nalgebra::{Const, DimNameAdd, DimNameSum, Dyn, OMatrix, RealField, U1};

// todo: possibly add number of nonzero basis functions as generic parameter
//  and make NonzeroIndices also of the SAME size
// todo: possibly transpose return value of eval_nonzero

/// Set of B-Spline basis functions.
///
/// The nonzero basis functions can be evaluated at a parametric point
/// using [`BsplineBasis::eval_nonzero`].
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
/// - [`N`] : Number of components of the basis functions.
///   For scalar valued functions (e.g. B-Splines) equal to `1`,
///   for vector valued functions equal to the dimension of the embedding space.
pub trait BsplineBasis<T: RealField, X, const N: usize> {
    /// Iterator over (linear) global indices corresponding to nonzero basis functions.
    type NonzeroIndices: Iterator<Item = usize>;
    
    /// Returns the number of basis functions in this set.
    fn num_basis(&self) -> usize;

    /// Evaluates the nonzero basis functions of this basis at the parametric point `x`.
    /// Returns the evaluated functions as well as the global nonzero indices.
    fn eval_nonzero(&self, x: X) -> (OMatrix<T, Dyn, Const<N>>, Self::NonzeroIndices);
}

/// Set of scalar valued B-Spline basis functions.
///
/// The nonzero basis functions can be evaluated at a parametric point
/// using [`BsplineBasis::eval_nonzero`].
/// The derivatives of nonzero basis functions
/// can be evaluated using [`ScalarBasis::eval_derivs_nonzero`].
///
/// # Type parameters
/// - [`T`] : Real scalar type.
/// - [`X`] : Type of parametric values in the reference domain.
pub trait ScalarBasis<T: RealField, X>: BsplineBasis<T, X, 1> {
    /// Evaluates the first [`K`] derivatives and functions values of nonzero basis functions
    /// of this scalar basis at the parametric point `x`.
    /// Returns a matrix where each row corresponds to the `i`-th derivative of the basis functions
    /// as well as the global nonzero indices.
    fn eval_derivs_nonzero<const K: usize>(&self, x: X) -> (OMatrix<T, DimNameSum<Const<K>, U1>, Dyn>, Self::NonzeroIndices)
        where Const<K>: DimNameAdd<U1>;
}