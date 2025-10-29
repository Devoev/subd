use crate::cells::quad::Quad;
use nalgebra::{Point2, RealField, Vector2};

// todo: actually use this

/// A quadrilateral element in a `ρ-z`-section of an axial symmetric geometry.
///
/// Embedded in the `ρ-z`-section the quadrilateral can be represented
/// by the coordinates `ρ` (radius) and `z` (height) alone.
/// For embedding into the 3D cylindrical geometry,
/// the additional coordinate `𝜑` (angle) is required.
#[derive(Debug, Clone, Copy)]
pub struct QuadAxi<T: RealField>(Quad<T, 2>);

impl<T: RealField> QuadAxi<T> {
    /// Constructs a new `QuadAxi` from the given `vertices`.
    ///
    /// In contrast to a [`Quad`] the vertices are not points in Euclidean space,
    /// but rather `(ρ,z)` coordinate pairs in cylindrical space.
    pub fn new(vertices: [Vector2<T>; 4]) -> Self {
        Self(Quad::new(vertices.map(Point2::from)))
    }
}