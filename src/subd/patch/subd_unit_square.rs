
/// Subdivision form of the unit square `[0,1] × [0,1]`.
///
/// The unit square is the standard parametric domain of a subdivision spline.
/// Around irregular vertices, the domain is infinitely partitioned into so called *segments*,
/// depicted as the L-shapes in the graphic
/// ```text
///      v ^
///        |
///     1  +---------------+
///        |               |
///        |               |
///        +-------+       |
///        |       |       |
///        +---+   |       |
///        |   |   |       |
///     0 -+---+---+-------+---->
///        0               1   u
/// ```
/// The irregular vertex corresponds to the parametric point `(0,0)`.
#[derive(Debug, Copy, Clone)]
pub enum SubdUnitSquare {
    /// Equivalent to the standard unit square.
    Regular,

    /// Infinite sequence of segments covering the unit square, except the irregular vertex.
    Irregular
}