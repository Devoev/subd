#![feature(impl_trait_in_assoc_type)]

mod knots;
mod bspline;
mod mesh;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use crate::bspline::spline_basis::SplineBasis;
    use crate::bspline::spline_curve::SplineCurve;
    use crate::knots::knot_vec::KnotVec;
    use nalgebra::{matrix, point};
    use plotters::backend::BitMapBackend;
    use plotters::chart::ChartBuilder;
    use plotters::prelude::{IntoDrawingArea, LineSeries, RED, WHITE};
    use crate::bspline::multivariate_spline_basis::MultivariateSplineBasis;
    use crate::knots::multivariate_knot_vec::MultivariateKnotVec;
    use crate::mesh::Mesh;

    #[test]
    fn knots() {
        let Xi1 = KnotVec::from_sorted(vec![0.0, 0.0, 0.5, 1.0, 1.0]);
        let Xi2 = KnotVec::<f64>::open(6, 2);
        let (m, Z): (Vec<_>, Vec<&f64>) = Xi1.breaks_with_multiplicity().unzip();
        let Xi3 = MultivariateKnotVec::new([Xi1.clone(), Xi2.clone()]);
        let Xi4 = MultivariateKnotVec::<f64, 2>::open([5, 3], [1, 2]);
        
        println!("Z: {:?}", Z);
        println!("m: {:?}", m);
        println!("{}", Xi1);
        println!("{}", Xi2);
        println!("{:?}", Xi1.elems().collect_vec());
        println!("{:?}", Xi2.elems().collect_vec());
        println!("{}", Xi3);
        println!("{}", Xi4);
        println!("{:?}", (&Xi4).into_iter().collect_vec());
        println!("{:?}", Xi4.breaks().collect_vec());
        println!("{:?}", Xi4.nodes().collect_vec());

        println!("Multivariate Bezier elements of Xi = {Xi4}:");
        for elem in Xi4.elems() {
            println!("Element {elem} of size {}", elem.elem_size());
        }

        println!("Multivariate knot vector has {} nodes and {} elems.", Xi4.num_nodes(), Xi4.num_elems())

    }

    #[test]
    fn splines() {
        let n = 4;
        let p = 2;
        let knots = KnotVec::<f64>::open(n, p);
        let splines = SplineBasis::new(knots.clone(), n, p);
        let splines_2d = MultivariateSplineBasis::<f64, 2>::open([5, 5], [1, 1]);
        let splines_3d = MultivariateSplineBasis::<f64, 3>::open([5, 5, 5], [1, 1, 1]);

        let t = 0.6;
        let idx = splines.find_span(t).unwrap();
        println!("{}", knots);
        println!("index {} in interval [{}, {})", idx, knots[idx], knots[idx+1]);
        
        println!("{}", splines.eval(t));
        println!("{}", splines_2d.eval_surf([t, t]));
        println!("{}", splines_2d.eval([t, t]));
        println!("{}", splines_3d.eval([t, t, t]));
    }

    #[test]
    fn spline_curves() {
        let n = 5;
        let p = 2;
        let knots = KnotVec::<f64>::open(n, p);
        let splines = SplineBasis::new(knots, n, p);
        let curve = SplineCurve::new(
            vec![point![-1.0, 0.0], point![-0.5, 0.7], point![0.0, 0.0], point![0.5, -0.7], point![1.0, 0.0]],
            splines
        ).unwrap();

        println!("{:?}", curve);

        let N = 1000;
        let mesh = curve.mesh(N);
        let data = mesh.nodes();

        let root_area = BitMapBackend::new("spline_curve.png", (800, 800))
            .into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        let mut ctx = ChartBuilder::on(&root_area)
            .build_cartesian_2d(-1.5..1.5, -1.5..1.5)
            .unwrap();

        ctx.configure_mesh().draw().unwrap();
        ctx.draw_series(LineSeries::new(data.map(|x| (x[0], x[1])), RED)).unwrap();
    }
}
