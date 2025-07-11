use nalgebra::RealField;
use crate::knots::breaks_with_multiplicity::BreaksWithMultiplicity;
use crate::knots::knot_span::KnotSpan;
use crate::knots::knot_vec::KnotVec;

/// Vector of index *increments* between successive unique knots in the knot vector.
#[derive(Debug, Clone)]
pub struct Increments(pub Vec<usize>);

impl Increments {
    /// Constructs a new [`Increments`] vector from the given `breaks`.
    pub fn from_multiplicities<T>(breaks: BreaksWithMultiplicity<T>) -> Self {
        let num_spans = breaks.len() - 1;
        let mut increments = Vec::with_capacity(num_spans);

        // Calculate increments as cumulative sum of multiplicities
        let mut acc = 0;
        for (k, _) in breaks.into_iter().take(num_spans) {
            acc += k - 1;
            increments.push(acc);
        }
        Increments(increments)
    }

    /// Constructs a new [`Increments`] vector from the given `knots`.
    pub fn from_knots<T: RealField>(knots: KnotVec<T>) -> Self {
        let breaks = BreaksWithMultiplicity::from_knots(knots);
        Increments::from_multiplicities(breaks)
    }
    
    /// Returns an iterator over all span indices defined by the increments.
    pub fn span_indices(&self) -> impl Iterator<Item=KnotSpan> + Clone + '_ {
        self.0.iter().enumerate().map(|(break_idx, increment)| KnotSpan(break_idx + increment))
    }
}