use std::error::Error;
use std::fmt::{Display, Formatter};

/// An error returned when the number of coefficients does not match the dimension of the space.
#[derive(Debug)]
pub struct CoeffsSpaceDimError {
    pub num_coeffs: usize,
    pub dim_space: usize
}

impl Display for CoeffsSpaceDimError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "number of coefficients (is {}) must match the dimension of the space (is {})",
            self.num_coeffs, self.dim_space)
    }
}

impl Error for CoeffsSpaceDimError {}