use nalgebra::DimName;

/// A [`K`]-dimensional cell with geometric information.
pub trait Cell<K: DimName> {
    /// Returns the parametrization of this cell.
    fn parametrization(&self);
    
    // todo: change return type of parametrization
    //  maybe add methods for jacobian? 
}