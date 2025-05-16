use nalgebra::DimName;

/// A [`K`]-dimensional cell with geometric information.
pub trait Cell<K: DimName> {
    /// Parametrization of this cell.
    type Parametrization;
    /// Returns the parametrization of this cell.
    fn parametrization(&self) -> Self::Parametrization;
    
    // todo: change return type of parametrization
    //  maybe add methods for jacobian? 
}