use std::iter::zip;
use nalgebra::{DMatrix, RealField, RowDVector, RowSVector, Scalar};
use num_traits::{ToPrimitive, Zero};
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
///        |       |       |
///        |  (2)     (1)  |
///        |       |       |
///        +-------+--   --|
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

// todo: replace with Index vector or dedicated Permutation vector newtype

/// A permutation vector to map control points from an irregular patch to a sub-patch.
type PermutationVec = [usize; 16];

impl SubdCell {
    /// Builds the permutation vector mapping the control points of the irregular patch of valence `n`
    /// to the control points of this regular subdivided patch.
    pub fn permutation_vec(&self, n: usize) -> PermutationVec {
        let m = 2 * n;

        match self {
            SubdCell::First => [
                7, 6, m + 4, m + 12,
                0, 5, m + 3, m + 11,
                3, 4, m + 2, m + 10,
                m + 6, m + 5, m + 1, m + 9
            ],
            SubdCell::Second => [
                0, 5, m + 3, m + 11,
                3, 4, m + 2, m + 10,
                m + 6, m + 5, m + 1, m + 9,
                m + 15, m + 14, m + 13, m + 8
            ],
            SubdCell::Third => [
                1, 0, 5, m + 3,
                2, 3, 4, m + 2,
                m + 7, m + 6, m + 5, m + 1,
                m + 16, m + 15, m + 14, m + 13
            ]
        }
    }

    /// Maps the evaluated local basis functions `b`
    /// from this regular sub-patch to the irregular patch of valence `n`.
    pub fn permute_basis<T: Scalar + Zero + Copy>(&self, n: usize, b: RowSVector<T, 16>) -> RowDVector<T> {
        let mut res = RowDVector::zeros(2*n + 17);
        for (bi, idx_global) in zip(b.iter(), self.permutation_vec(n)) {
            res[idx_global] = *bi;
        }
        res
    }
}

// todo: move this to possible permutation vec file

/// Applies the given permutation `p` to the evaluated basis functions `b`,
/// mapping them from a regular sub-patch to the irregular patch of valence `n`.
pub fn apply_permutation<T: RealField + Copy>(n: usize, b: RowSVector<T, 16>, p: PermutationVec) -> RowDVector<T> {
    let mut res = RowDVector::zeros(2*n + 17);
    for (bi, pi) in zip(b.iter(), p) {
        res[pi] = *bi;
    }
    res
}

/// Constructs the permutation matrix from the given permutation vector `p` and valence `n`.
pub fn permutation_matrix(p: PermutationVec, n: usize) -> DMatrix<usize> {
    let mut mat = DMatrix::<usize>::zeros(16, 2*n + 17);
    for (i, pi) in p.into_iter().enumerate() {
        // Set i-th row and pi-th column to 1
        mat[(i, pi)] = 1;
    }
    mat
}