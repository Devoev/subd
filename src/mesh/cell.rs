use nalgebra::{DimName, DimNameDiff, DimNameSub, U1};

/// Topology of a `K`-cell inside a mesh.
pub trait CellTopo<K: DimName + DimNameSub<U1>> {
    
    /// Topology of the `K-1`-dimensional boundary of this cell.
    type Boundary<L: DimName>;
    
    /// Returns the [boundary topology](Self::Boundary) of this cell.
    fn boundary(&self) -> Self::Boundary<DimNameDiff<K, U1>>;
}

/// Geometric `K`-cell.
pub trait Cell {
    
    /// `K-1`-dimensional boundary of this cell. 
    type Boundary;
    
    /// Returns the [boundary](Self::Boundary) of this cell.
    fn boundary(&self) -> Self::Boundary;
}