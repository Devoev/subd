use nalgebra::{Const, Dyn, OMatrix, RealField};

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
/// For scalar valued functions (e.g. B-Splines) equal to `1`,
/// for vector valued functions equal to the dimension of the embedding space.
pub trait BsplineBasis<T: RealField, X, const N: usize> {
    /// Iterator over (linear) global indices corresponding to nonzero basis functions.
    type NonzeroIndices: Iterator<Item = usize>;
    
    /// Returns the length of this set, i.e. the number of basis functions.
    fn len(&self) -> usize;

    /// Evaluates the nonzero basis functions of this basis at the parametric point `x`.
    /// Returns the evaluated functions as well as the global nonzero indices.
    fn eval_nonzero(&self, x: X) -> (OMatrix<T, Dyn, Const<N>>, Self::NonzeroIndices);
}

/// Conversion into breakpoints.
pub trait IntoBreaks<'a> {
    /// An iterator that yields the knot breakpoints.
    type Breaks: Iterator;

    /// An iterator that yields the breakpoints with multiplicity.
    type BreaksWithMultiplicity: Iterator;

    /// Returns an iterator over the breaks, i.e. unique knot values.
    fn breaks(&'a self) -> Self::Breaks;

    /// Returns an iterator over `(multiplicity, break)` pairs.
    fn breaks_with_multiplicity(&'a self) -> Self::BreaksWithMultiplicity;
}