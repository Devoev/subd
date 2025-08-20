use nalgebra::{Dim, DimName};

/// Set of [`Self::NumBasis`] basis functions with [`Self::NumComponents`].
pub trait Basis {
    /// Number of basis functions.
    type NumBasis: Dim;

    /// Number of components for each basis function.
    /// For scalar valued functions equal to `1`,
    /// for vector valued functions equal to the dimension of the parametric domain.
    type NumComponents: DimName;

    /// Coordinate of a parametric point over type [`T`].
    type Coord<T>;

    /// Returns the number of basis functions wrapped into [`Self::NumBasis`] (`Const` or `Dyn`).
    fn num_basis_generic(&self) -> Self::NumBasis;

    /// Returns the number of basis functions in this set.
    fn num_basis(&self) -> usize {
        self.num_basis_generic().value()
    }

    /// Returns the number of components wrapped into [`Self::NumComponents`] (`Const`).
    fn num_components_generic(&self) -> Self::NumComponents {
        Self::NumComponents::name()
    }

    /// Returns the number of components for each basis function.
    fn num_components(&self) -> usize {
        Self::NumComponents::dim()
    }
}