use nalgebra::RealField;
use num_traits::ToPrimitive;
use numeric_literals::replace_float_literals;

/// Subdivision form of the unit square `(0,1) × (0,1)`.
///
/// The unit square is the standard parametric domain of a subdivision spline.
/// Around irregular vertices, the domain is infinitely partitioned into so called *segments*,
/// depicted as the L-shapes in the graphic
/// ```text
///      v ^
///        |
///     1  +---------------+
///        |               |
///        | (1,2)   (1,1) |
///        |               |
///        +-------+       |
///        |       |       |
///        +---+   | (1,0) |
///        |   |   |       |
///     0 -+---+---+-------+---->
///        0               1   u
/// ```
/// The three [subcells](SubdCell) of the first regular segment
/// are labeled `(1,0)`, `(1,1)` and `(1,2)`,
/// where the first number `n=1` is the level of subdivision
/// and the second one `k=0,1,2` is the index of the subcell.
/// The irregular vertex corresponds to the parametric point `(0,0)`.
#[derive(Debug, Copy, Clone)]
pub enum SubdUnitSquare {
    /// Equivalent to the standard unit square.
    Regular,

    /// Infinite sequence of segments covering the unit square, except the irregular vertex.
    Irregular
}

impl SubdUnitSquare {
    /// Evaluates the parametrization.
    ///
    /// Transforms the given parametric values `(u,v) ∈ (0,1)²` (from the unit square)
    /// to the `k`-th *regular* subcell `(n,k)` of the `n`-th subdivided segment.
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    pub fn transform<T: RealField + Copy + ToPrimitive>(mut u: T, mut v: T) -> (T, T, usize, SubdCell) {
        // Determine number of required subdivisions
        // For u,v = 1, set the value to 1 still
        let n = (-u.log2()).min(-v.log2()).ceil().to_usize()
            .map(|n| if n == 0 { 1 } else { n }).unwrap();

        // Calculate 2^(n-1) using left bit shifts
        let pow = T::from_i32(1 << (n - 1)).unwrap();

        // Transform (u,v) to regular sub-cell
        u *= pow;
        v *= pow;
        let k = SubdCell::from_parameters_in_segment(u, v);
        let (u, v) = k.transform_inv(u, v);
        (u, v, n, k)
    }
}

/// Subdivided cell (halved unit square) inside a segment of a [`SubdUnitSquare`].
/// One of the following three cells `(0)`, `(1)` or `(2)`.
/// ```text
///      v ^
///        |
///     1  +---------------+
///        |               |
///        |  (2)     (1)  |
///        |               |
///        +-------+       |
///        |       |       |
///        |       |  (0)  |
///        |       |       |
///     0 -+---+---+-------+---->
///        0               1   u
/// ```
#[derive(Debug, Copy, Clone)]
pub enum SubdCell {
    /// First cell (`k = 0`), corresponding to the parametric range `(u,v) ∈ (0.5, 1) × (0,0.5)`
    First = 0,

    /// Second cell (`k = 1`), corresponding to the parametric range `(u,v) ∈ (0.5, 1) × (0.5,1)`
    Second = 1,

    /// Third cell (`k = 3`), corresponding to the parametric range `(u,v) ∈ (0,0.5) × (0.5, 1)`
    Third = 2,
}

impl SubdCell {
    /// Gets the subdivided cell the given parameters `(u,v)` of a regular segment are in.
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    pub fn from_parameters_in_segment<T: RealField>(u: T, v: T) -> SubdCell {
        if v < 0.5 { SubdCell::First }
        else if u < 0.5 { SubdCell::Third }
        else { SubdCell::Second }
    }

    /// Evaluates the inverse parametrization aka. the chart.
    ///
    /// Transforms the parameters `(u,v)` from this regular subdivided cell to the unit square.
    #[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
    pub fn transform_inv<T: RealField>(&self, u:T, v: T) -> (T, T) {
        match self {
            SubdCell::First => (u*2.0 - 1.0, v*2.0),
            SubdCell::Second => (u*2.0 - 1.0, v*2.0 - 1.0),
            SubdCell::Third => (u*2.0, v*2.0 - 1.0),
        }
    }
}