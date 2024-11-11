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
    }
}
