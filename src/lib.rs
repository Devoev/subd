mod knots;
mod bspline;

#[cfg(test)]
mod tests {
    use nalgebra::point;
    use crate::bspline::spline_basis::SplineBasis;
    use crate::bspline::spline_curve::SplineCurve;
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

    #[test]
    fn splines() {
        let n = 4;
        let p = 2;
        let knots = KnotVec::<f64>::open(n, p);
        let splines = SplineBasis::new(knots.clone(), n, p);

        let t = 0.6;
        let idx = splines.find_span(t).unwrap();
        println!("{}", knots);
        println!("index {} in interval [{}, {})", idx, knots[idx], knots[idx+1]);
        
        println!("{:?}", splines.eval(t));
    }

    #[test]
    fn spline_curves() {
        let n = 3;
        let p = 1;
        let knots = KnotVec::<f64>::open(n, p);
        let splines = SplineBasis::new(knots.clone(), n, p);
        let curve = SplineCurve::new(
            vec![point![1.0, 0.0], point![1.0, 1.0], point![0.0, 1.0]],
            splines
        );

        let t = 0.25;
        println!("{:?}", curve.eval(t));
    }
}
