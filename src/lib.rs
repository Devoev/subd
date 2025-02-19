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
    use nalgebra::{dmatrix, matrix, point, Const, Dyn, Matrix, Matrix3, Matrix3xX, Matrix5, MatrixView, OMatrix, SMatrix, U2, U7};
    use plotters::backend::BitMapBackend;
    use plotters::chart::ChartBuilder;
    use plotters::prelude::{IntoDrawingArea, LineSeries, RED, WHITE};
    use crate::bspline::control_points::ControlPoints;
    use crate::bspline::multivariate_spline_basis::MultivariateSplineBasis;
    use crate::bspline::spline::Spline;
    use crate::knots::multi_knot_vec::MultiKnotVec;
    use crate::mesh::Mesh;

    #[test]
    fn knots() {
        let Xi1 = KnotVec::from_sorted(vec![0.0, 0.0, 0.5, 1.0, 1.0]);
        let Xi2 = KnotVec::<f64>::open(6, 2);
        let (m, Z): (Vec<_>, Vec<&f64>) = Xi1.breaks_with_multiplicity().unzip();
        let Xi3 = MultiKnotVec::new([Xi1.clone(), Xi2.clone()]);
        let Xi4 = MultiKnotVec::<f64, 2>::open([5, 3], [1, 2]);
        
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

        println!("Multivariate knot vector has {} nodes and {} elems.", Xi4.num_nodes(), Xi4.num_elems());

        let t = 0.6;
        let span1 = Xi2.find_span(t, 6).unwrap();
        let span2 = Xi4.find_span([t, t], [5, 3]).unwrap();
        
        println!("Span for univariate knot vec {:?}", span1.nonzero_indices(2).collect_vec());
        println!("Span for multivariate knot vec {:?}", span2.nonzero_indices([1, 2]).collect_vec());
        
        let lin_indices = span2.nonzero_indices([1, 2])
            .map(|idx| MultiKnotVec::<f64, 2>::linear_index(idx.into_iter().collect_array().unwrap(), [5, 3]))
            .collect_vec();
        
        println!("Span for multivariate knot vec {:?} (linear)", lin_indices);
    }

    #[test]
    fn multi_index() {
        const N: usize = 6;
        let p = 2;
        let t = 0.5;

        let knots = MultiKnotVec::<f64, 2>::open([N, N], [p, p]);
        let span = knots.find_span([t, t], [N, N]).unwrap();
        let idx = span.nonzero_indices([p, p]).collect_vec();
        let lin_idx = idx.clone().into_iter()
            .map(|i| MultiKnotVec::<f64, 2>::linear_index(i.into_iter().collect_array().unwrap(), [N, N]))
            .collect_vec();
        let lin_idx_2 = span.nonzero_lin_indices([N, N], [p, p]).collect_vec();
        let mat_idx = lin_idx.iter()
            .map(|i| OMatrix::<f64, Const<N>, Const<N>>::zeros().vector_to_matrix_index(*i))
            .collect_vec();

        println!("{}", knots);
        println!("{:?}", idx);
        println!("{:?}", lin_idx);
        println!("{:?}", lin_idx_2);
        println!("{:?}", mat_idx);
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
        println!("{}", knots);
        
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
        let splines_2d = MultivariateSplineBasis::new([splines.clone(), splines.clone()]);
        let coords = matrix![
            -1.0, -0.5, 0.0, 0.5, 1.0;
            0.0, 0.7, 0.0, -0.7, 0.0;
        ];
        let coords2 = SMatrix::<f64, 2, 25>::new_random();
        let control_points = ControlPoints::new(coords);
        let control_points2 = ControlPoints::new(coords2);
        
        let curve = SplineCurve::new(
            control_points.clone().point_iter().collect_vec(),
            splines.clone()
        ).unwrap();

        let curve2 = Spline::new(control_points, MultivariateSplineBasis::new([splines])).unwrap();
        let surf = Spline::new(control_points2, splines_2d).unwrap();
        
        dbg!(curve.eval(0.0));
        dbg!(curve2.eval_curve(0.0));
        dbg!(surf.eval([0.0, 0.0]));

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
