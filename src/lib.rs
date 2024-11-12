mod knots;

#[cfg(test)]
mod tests {
    use crate::knots::knot_vec::KnotVec;

    #[test]
    fn knots() {
        let Xi1 = KnotVec::from_sorted(vec![0.0, 0.0, 0.5, 1.0, 1.0]);
        let Xi2 = KnotVec::<f64>::open(6, 2);
        let (Z, m) = Xi1.breaks_with_multiplicity();
        println!("Z: {:?}", Z);
        println!("m: {:?}", m);
        println!("{}", Xi1);
        println!("{}", Xi2);

        let t = 0.7;
        let idx = Xi2.find_span(t);
        println!("value {} in interval [{}, {})", t, Xi2[idx], Xi2[idx+1]);
    }
}
