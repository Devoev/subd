// todo:
//  1. possibly replace with MultiProd<UnitInterval> and UnitInterval as Unit<Interval>
//  2. store calculated coords directly in struct instead of computation in coords()

/// The [`D`]-dimensional unit hyper-cube `[0,1]^D`.
/// In 2D the domain is given by
/// ```text
///      v ^
///        |
///     1 -+------+
///        |      |
///        |      |
///     0 -+------+--->
///        0      1   u
/// ```
#[derive(Debug, Copy, Clone, Default)]
pub struct UnitCube<const D: usize>;

impl <const D: usize> UnitCube<D> {
    /// Returns the `D` unit intervals.
    pub fn intervals(&self) -> [UnitCube<1>; D] {
        [UnitCube; D]
    }
}

/// The [`D`]-dimensional symmetric normalized hyper-cube `[-1,1]^D`.
/// In 2D the domain is given by
/// ```text
///            v ^
///  (-1,1)      |      (1,1)
///       +------+------+
///       |      |      |
///       |      |      |
///    ---+------+------+--->
///       |      |      |   u
///       |      |      |
///       +------+------+ 
/// (-1,-1)      |      (1,-1)
/// ```
#[derive(Debug, Copy, Clone, Default)]
pub struct SymmetricUnitCube<const D: usize>;

impl <const D: usize> SymmetricUnitCube<D> {
    /// Returns the `D` symmetric unit intervals.
    pub fn intervals(&self) -> [SymmetricUnitCube<1>; D] {
        [SymmetricUnitCube; D]
    }
}