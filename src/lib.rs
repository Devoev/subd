#![feature(impl_trait_in_assoc_type)]
#![feature(cmp_minmax)]
extern crate core;

mod knots;
mod bspline;
mod subd;
mod mesh;

#[cfg(test)]
mod tests {
    use crate::bspline::basis::Basis;
    use crate::bspline::control_points::ControlPoints;
    use crate::bspline::multi_spline_basis::MultiSplineBasis;
    use crate::bspline::spline::{Spline, SplineCurve, SplineSurf};
    use crate::bspline::spline_basis::SplineBasis;
    use crate::knots::index::{IntoLinear, Linearize, MultiIndex, Strides};
    use crate::knots::knot_vec::KnotVec;
    use itertools::Itertools;
    use nalgebra::{matrix, vector, Const, OMatrix, SMatrix};
    use plotters::backend::BitMapBackend;
    use plotters::chart::ChartBuilder;
    use plotters::prelude::{IntoDrawingArea, LineSeries, RED, WHITE};

    #[test]
    fn knots() {
        let xi1 = SplineBasis::new(KnotVec(vec![0.0, 0.0, 0.5, 1.0, 1.0]), 3, 1).unwrap();
        let xi2 = SplineBasis::<f64>::open_uniform(6, 2);
        let (m, z): (Vec<_>, Vec<&f64>) = xi1.knots.breaks_with_multiplicity().unzip();
        let xi3 = MultiSplineBasis::new([xi1.clone(), xi2.clone()]);
        let xi4 = MultiSplineBasis::<f64, 2>::open_uniform([5, 3], [1, 2]);
        
        println!("Z: {:?}", z);
        println!("m: {:?}", m);
        println!("{:?}", xi1);
        println!("{:?}", xi2);
        println!("{:?}", xi3);
        println!("{:?}", xi4);

        let t = 0.6;
        let span1 = xi2.find_span(t).unwrap();
        let span2 = xi4.find_span(vector![t, t]).unwrap();
        
        println!("Span for univariate knot vec {:?}", span1.nonzero_indices(xi2.p).collect_vec());
        println!("Span for multivariate knot vec {:?}", span2.nonzero_indices(xi4.p()).collect_vec());
    }

    #[test]
    fn multi_index() {
        const N: usize = 6;
        let p = 2;
        let t = 0.5;
        
        let dims = [3, 3, 3];
        let strides = Strides::from_dims(dims);
        let multi_idx = MultiIndex([2, 2, 2]);
        println!("{:?}", multi_idx);
        println!("{:?}", multi_idx.into_lin(strides));

        let space = MultiSplineBasis::<f64, 2>::open_uniform([N, N], [p, p]);
        let strides = Strides::from_dims(space.n());
        let span = space.find_span(vector![t, t]).unwrap();
        let idx = span.nonzero_indices(space.p()).collect_vec();
        let lin_idx = idx.clone().into_iter()
            .map(|i| i.into_lin(strides))
            .collect_vec();
        let lin_idx_2 = span.nonzero_indices(space.p()).linearize(strides).collect_vec();
        let mat_idx = lin_idx.iter()
            .map(|i| OMatrix::<f64, Const<N>, Const<N>>::zeros().vector_to_matrix_index(*i))
            .collect_vec();

        println!("{:?}", space);
        println!("{:?}", idx);
        println!("{:?}", lin_idx);
        println!("{:?}", lin_idx_2);
        println!("{:?}", mat_idx);
    }

    #[test]
    fn splines() {
        let n = 4;
        let p = 2;
        let splines = SplineBasis::<f64>::open_uniform(n, p);
        let splines_2d = MultiSplineBasis::<f64, 2>::open_uniform([5, 5], [1, 1]);
        let splines_3d = MultiSplineBasis::<f64, 3>::open_uniform([5, 5, 5], [1, 1, 1]);

        let t = 0.6;
        println!("{}", splines.eval(t, &splines.find_span(t).unwrap()));
        println!("{}", splines_2d.eval(vector![t, t], &splines_2d.find_span(vector![t, t]).unwrap()));
        println!("{}", splines_3d.eval(vector![t, t, t], &splines_3d.find_span(vector![t, t, t]).unwrap()));
    }

    #[test]
    fn spline_curves() {
        let n = 5;
        let p = 2;
        let space = SplineBasis::<f64>::open_uniform(n, p);
        let coords = matrix![
            -1.0, -0.5, 0.0, 0.5, 1.0;
            0.0, 0.7, 0.0, -0.7, 0.0;
        ];
        let control_points = ControlPoints::new(coords);

        let curve = SplineCurve::new(control_points, space).unwrap();
        dbg!(curve.eval(0.0));
    }

    #[test]
    fn spline_surf() {
        let n = 3;
        let p = 2;

        let splines_1d = SplineBasis::<f64>::open_uniform(n, p);
        let splines_2d = MultiSplineBasis::new([splines_1d.clone(), splines_1d.clone()]);

        let control_points = ControlPoints::new(matrix![
            0.0, 0.3, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.8;
            0.1, 0.2, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.8
        ]);

        let coords_rand = SMatrix::<f64, 2, 25>::new_random();
        let control_points_rand = ControlPoints::new(coords_rand);

        let surf = SplineSurf::new(control_points, splines_2d).unwrap();

        const N: i32 = 100;
        let mut points: Vec<(f64, f64)> = vec![];

        for i in 0..N {
            for j in 0..N {
                let tx = i as f64 / N as f64;
                let ty = j as f64 / N as f64;
                let p = surf.eval(vector![tx, ty]);
                points.push((p.x, p.y));
            }
        }

        let data = points.into_iter();
        let root_area = BitMapBackend::new("spline_surf.png", (800, 800))
            .into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        let mut ctx = ChartBuilder::on(&root_area)
            .build_cartesian_2d(-0.1..1.1, -0.1..1.1)
            .unwrap();

        ctx.configure_mesh().draw().unwrap();
        ctx.draw_series(LineSeries::new(data, RED)).unwrap();
    }
}
