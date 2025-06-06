#![feature(impl_trait_in_assoc_type)]
#![feature(cmp_minmax)]
extern crate core;

pub mod basis;
pub mod bspline;
pub mod cells;
pub mod index;
pub mod knots;
pub mod mesh;
pub mod operator;
pub mod quadrature;
pub mod subd;
mod diffgeo;

#[cfg(test)]
mod tests {
    use crate::basis::local::LocalBasis;
    use crate::basis::traits::{Basis, DiffBasis, NumBasis};
    use crate::bspline::basis::{BsplineBasis, ScalarBasis};
    use crate::bspline::de_boor::DeBoorMulti;
    use crate::bspline::de_boor::{DeBoor, DeBoorBi};
    use crate::bspline::global_basis::MultiBsplineBasis;
    use crate::bspline::grad::BasisGrad;
    use crate::bspline::space::SplineSpace;
    use crate::bspline::spline_geo::{Jacobian, SplineCurve, SplineGeo};
    use crate::bspline::{cart_prod, global_basis, tensor_prod};
    use crate::cells::quad::QuadTopo;
    use crate::cells::topo::Cell;
    use crate::cells::vertex::VertexTopo;
    use crate::index::dimensioned::{DimShape, Strides};
    use crate::index::multi_index::MultiIndex;
    use crate::knots::knot_span::KnotSpan;
    use crate::knots::knot_vec::KnotVec;
    use crate::mesh::bezier::BezierMesh;
    use crate::mesh::cartesian::CartMesh;
    use crate::mesh::geo::Mesh;
    use crate::mesh::topo::MeshTopology;
    use crate::operator::hodge::assemble_hodge;
    use crate::quadrature::tensor_prod_gauss_legendre::TensorProdGaussLegendre;
    use crate::subd::basis;
    use gauss_quad::GaussLegendre;
    use iter_num_tools::lin_space;
    use itertools::Itertools;
    use nalgebra::{matrix, vector, DMatrix, DVector, Dyn, OMatrix, SMatrix, SVector, Vector, U2};
    use plotters::backend::BitMapBackend;
    use plotters::chart::ChartBuilder;
    use plotters::prelude::{IntoDrawingArea, LineSeries, RED, WHITE};
    use std::hint::black_box;
    use std::iter::zip;
    use std::time::Instant;
    use crate::basis::space::Space;
    use crate::cells::hyper_rectangle::HyperRectangle;
    use crate::knots::breaks::Breaks;
    use crate::quadrature::bezier::BezierQuad;
    use crate::quadrature::tensor_prod::GaussLegendreMulti;
    use crate::quadrature::traits::{Quadrature, RefQuadrature};

    #[test]
    fn knots() {
        let n = 5;
        let p = 1;
        let uniform = KnotVec::<f64>::new_uniform(n-p+1);
        let open = KnotVec::<f64>::new_open(uniform.clone(), p-1);
        let open_uniform = KnotVec::<f64>::new_open_uniform(n, p);

        println!("{uniform:?}");
        println!("{open:?}");
        println!("{open_uniform:?}");
    }

    #[test]
    fn multi_index() {
        const N: usize = 6;
        let p = 2;
        let t = 0.5;

        let dims = DimShape([3, 3, 3]);
        let strides = Strides::from(dims);
        let multi_idx = [2, 2, 2];
        println!("{:?}", multi_idx);
        println!("{:?}", multi_idx.into_lin(&strides));

        let space = DeBoorMulti::<f64, 2>::open_uniform([N, N], [p, p]);
        let strides = Strides::from(space.num_basis());
    }

    #[test]
    fn splines() {
        let n = 4;
        let p = 2;
        let splines = DeBoor::<f64>::open_uniform(n, p);
        let splines_2d = DeBoorMulti::<f64, 2>::open_uniform([5, 5], [1, 1]);
        let splines_3d = DeBoorMulti::<f64, 3>::open_uniform([5, 5, 5], [1, 1, 1]);

        let t = 0.6;
        println!("{}", splines.eval_nonzero(t).0);
        println!("{}", splines_2d.eval_nonzero([t, t]).0);
        println!("{}", splines_3d.eval_nonzero([t, t, t]).0);
    }

    #[test]
    fn spline_curves() {
        let n = 5;
        let p = 2;
        let basis = DeBoor::<f64>::open_uniform(n, p);
        let space = SplineSpace::new(basis);
        let coords = matrix![
            -1.0, -0.5, 0.0, 0.5, 1.0;
            0.0, 0.7, 0.0, -0.7, 0.0;
        ];

        let curve = SplineCurve::new(coords, &space);
        dbg!(curve.eval(0.0));
    }

    #[test]
    fn vector_basis() {
        let n = 20;
        let p = 3;
        let basis_p = DeBoor::<f64>::open_uniform(n, p);
        let basis_q = DeBoor::<f64>::open_uniform(n, p - 1);
        let basis_x = DeBoorBi::new(basis_p.clone(), basis_q.clone());
        let basis_y = tensor_prod::Prod::new(basis_q, basis_p);

        let sp_vec_2d = cart_prod::Prod::new(basis_x.clone(), basis_y);
        let sp_vec_3d = cart_prod::TriProd::new(basis_x.clone(), basis_x.clone(), basis_x.clone());
        println!("{}", sp_vec_2d.eval_nonzero([0.5, 0.2]).0);
        println!("{}", sp_vec_3d.eval_nonzero([0.5, 0.2]).0);
    }

    #[test]
    fn spline_surf() {
        let n = 3;
        let p = 2;

        let splines_1d = DeBoor::<f64>::open_uniform(n, p);
        let splines_2d = DeBoorMulti::new([splines_1d.clone(), splines_1d.clone()]);
        let space = SplineSpace::new(splines_2d);

        let control_points = matrix![
            0.0, 0.3, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.8;
            0.1, 0.2, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.8
        ];

        let coords_rand = SMatrix::<f64, 2, 9>::new_random();

        let surf = space.linear_combination(coords_rand);

        const N: i32 = 100;
        let mut points: Vec<(f64, f64)> = vec![];

        for i in 0..N {
            for j in 0..N {
                let tx = i as f64 / N as f64;
                let ty = j as f64 / N as f64;
                let p = surf.eval([tx, ty]);
                points.push((p.x, p.y));
            }
        }

        let data = points.into_iter();
        let root_area = BitMapBackend::new("spline_surf.png", (800, 800)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        let mut ctx = ChartBuilder::on(&root_area)
            .build_cartesian_2d(-0.1..1.1, -0.1..1.1)
            .unwrap();

        ctx.configure_mesh().draw().unwrap();
        ctx.draw_series(LineSeries::new(data, RED)).unwrap();
    }

    #[test]
    fn spline_derivs() {
        let n = 3;
        let p = 1;

        // Univariate derivatives
        let knots = KnotVec::<f64>::new_open_uniform(n, p);
        let basis = global_basis::BsplineBasis::new(knots, n, p);
        let basis_3d = MultiBsplineBasis::<f64, 3>::repeat(basis.clone());
        // let space = Space::<f64, f64, _>::new(basis_3d);

        // todo: implement these "find" functions directly in LocalBasis
        let find_local_basis_3d = |x: [f64; 3]| {
            let vertex = Vector::from(x);
            let elem = HyperRectangle::new(vertex, vertex);
            basis_3d.elem_basis(&elem)
        };

        let find_local_basis = |x: f64| {
            let vertex = Vector::from([x]);
            let elem = HyperRectangle::new(vertex, vertex);
            basis.elem_basis(&elem)
        };

        let de_boor = DeBoor::<f64>::open_uniform(n, p);
        let de_boor_multi = DeBoorMulti::new([de_boor.clone(), de_boor.clone(), de_boor.clone()]);
        let space = SplineSpace::new(de_boor_multi.clone());

        println!(
            "Derivatives of basis: {}",
            find_local_basis(0.8).eval_derivs::<3>(0.8)
        );

        // Jacobian
        let control_points = SMatrix::<f64, 3, 27>::new_random();
        let surf = space.linear_combination(control_points);
        let d_phi = Jacobian { geo_map: &surf };
        let j = d_phi.eval([0.0, 0.2, 0.5]);
        println!("Jacobian matrix: {j}");

        // Function values
        let x = [0.1, 0.0, 0.5];
        println!(
            "Function values of basis: {}",
            find_local_basis_3d(x).eval([0.1, 0.0, 0.5])
        );

        // todo: implement Hgrad for other bases
        // Gradients
        let grad_b = BasisGrad::new(de_boor_multi);
        println!("Gradients of basis: {}", grad_b.eval([0.1, 0.0, 0.5]));
    }

    #[test]
    fn bezier_elems() {
        let knots =
            KnotVec::new(vec![0.0, 0.0, 0.0, 0.2, 0.4, 0.4, 0.4, 0.8, 1.0, 1.0, 1.0]).unwrap();
        let basis = DeBoor::new(knots.clone(), 7, 3).unwrap();
        // let quad = GaussLegendre::new(5).unwrap();

        let breaks = Breaks::from_knots(knots.clone());
        let ref_mesh = CartMesh::from_breaks([breaks]);

        for (elem, topo) in zip(ref_mesh.elems(), ref_mesh.topology.elems()) {
            let span = basis.find_span(elem.a.x).unwrap();
            println!(
                "Bezier element = [{:.3}, {:.3}] (index = {})",
                elem.a.x, elem.b.x, topo.0[0]
            );
            println!(
                "Knot span = [{:.3}, {:.3}] (index = {})",
                knots[span.0],
                knots[span.0 + 1],
                span.0
            );
        }
    }

    #[test]
    fn quadrature() {
        let quad_1d = GaussLegendre::new(2).unwrap();
        let quad_multi = GaussLegendreMulti::<f64, 2>::new([quad_1d.clone(), quad_1d.clone()]);

        println!("Quadrature nodes = {:?} (in [-1,1]]", quad_multi.nodes_ref().collect_vec());
        println!("Quadrature weights = {:?} (in [-1,1])", quad_multi.weights_ref().collect_vec());

        let elem = HyperRectangle::new(vector![0.2, 0.2], vector![0.4, 0.4]);
        println!("Quadrature nodes = {:?} in {:?}", quad_multi.nodes_elem(&elem).collect_vec(), elem);
        println!("Quadrature weights = {:?} in {:?}", quad_multi.weights_elem(&elem).collect_vec(), elem);
    }

    #[test]
    fn mesh() {
        let quad = QuadTopo([VertexTopo(0), VertexTopo(1), VertexTopo(2), VertexTopo(3)]);
        quad.is_connected::<2>(&quad);
    }

    #[test]
    fn cart_mesh() {
        let breaks = Breaks(vec![0.0, 1.0, 2.0, 3.0]);
        let msh = CartMesh::from_breaks([breaks.clone(), breaks]);

        for elem in msh.elems() {
            println!("Nodes of rectangle {:?}", elem.points().collect_vec());
            println!("Ranges of rectangle {:?}", elem.ranges());
        }
    }

    #[test]
    fn iga_assembly() {
        let n = 3;
        let p = 1;
        let knots = DeBoor::<f64>::open_uniform(n, p).knots;

        let basis_geo = DeBoorMulti::<f64, 2>::open_uniform([2, 2], [1, 1]);
        let space_geo = SplineSpace::new(basis_geo);
        let c = OMatrix::<f64, U2, Dyn>::from_column_slice(&[
            0.0, 0.0,
            0.0, 1.0,
            1.0, 0.0,
            1.0, 1.0]
        );
        let geo_map = SplineGeo::new(c, &space_geo);

        let breaks = Breaks::from_knots(knots.clone());
        let cart_mesh = CartMesh::from_breaks([breaks.clone(), breaks]);
        let msh = BezierMesh::new(cart_mesh, geo_map);
        let basis = global_basis::BsplineBasis::new(knots, n, p);
        let basis = MultiBsplineBasis::new([basis.clone(), basis]);
        let space = Space::new(basis);

        let ref_quad = GaussLegendreMulti::with_degrees([5, 2]);
        let quad = BezierQuad::new(ref_quad);
        let mat = assemble_hodge(&msh, &space, quad);

        // Print
        let mut dense = DMatrix::<f64>::zeros(space.dim(), space.dim());
        for (i, j, &v) in mat.triplet_iter() {
            dense[(i, j)] = v;
        }
        println!("{}", dense);
        println!(
            "Eigenvalues = {} (should be positive)",
            dense.eigenvalues().unwrap()
        );
        println!(
            "||M - M^T|| = {} (should be zero)",
            (dense.clone() - dense.transpose()).norm()
        );
    }

    #[test]
    fn benchmark_de_boor_vs_mat_mat() {
        let num_eval = 1000;
        let grid = lin_space(0.0..=1.0, num_eval);
        let grid = grid.clone().cartesian_product(grid);

        // de Boor algorithm
        let start = Instant::now();
        // todo: using uniform(8) throws errors, because xi is not open!
        // let xi = KnotVec::uniform(8);
        // let basis_uni = SplineBasis::new(xi, 4, 3).unwrap();

        let basis_uni = DeBoor::open_uniform(4, 3);
        let basis = DeBoorMulti::<f64, 2>::new([basis_uni.clone(), basis_uni]);
        for (u, v) in grid.clone() {
            // let span = KnotSpan(MultiIndex([3, 3])); // hardcoding span for open uniform knot vector of maximal regularity (=global polynomials)
            let eval = basis.eval_nonzero([u, v]);
            // println!("{}", eval.norm());
        }
        let time_de_boor = start.elapsed();

        // matrix-matrix (for catmull clark)
        let start = Instant::now();
        for (u, v) in grid.clone() {
            let eval = basis::eval_regular(u, v);
            // println!("{}", eval.norm());
        }
        let time_mat_mat = start.elapsed();

        println!(
            "Took {:?} for {num_eval} basis evaluations (de Boor).",
            time_de_boor
        );
        println!(
            "Took {:?} for {num_eval} basis evaluations (matrix-matrix).",
            time_mat_mat
        );
        println!(
            "De Boors algorithm is {} % faster than matrix-matrix algorithm",
            (time_mat_mat.as_secs_f64() - time_de_boor.as_secs_f64()) / time_mat_mat.as_secs_f64()
                * 100.0
        )
    }

    #[test]
    fn benchmark_uni_vs_tp() {
        let num_eval = 10_000_000;
        let grid = lin_space(0.0..=1.0, num_eval);

        // univariate algorithm
        let start = Instant::now();
        let basis = DeBoor::open_uniform(30, 3);
        for t in grid.clone() {
            let _ = black_box(basis.eval_nonzero(t));
        }
        let time_uni = start.elapsed();

        // tensor product algorithm
        let start = Instant::now();
        let basis = DeBoorMulti::open_uniform([30], [3]);
        for t in grid.clone() {
            let _ = black_box(basis.eval_nonzero(t));
        }
        let time_tp = start.elapsed();

        println!(
            "Took {:?} for {num_eval} basis evaluations (univariate algorithm).",
            time_uni
        );
        println!(
            "Took {:?} for {num_eval} basis evaluations (tensor product algorithm).",
            time_tp
        );
        println!(
            "Univariate algorithm is {} % faster than tensor product algorithm",
            (time_tp.as_secs_f64() - time_uni.as_secs_f64()) / time_tp.as_secs_f64() * 100.0
        )
    }

    #[test]
    fn benchmark_bi_vs_tp() {
        let num_eval = 5_000;
        let n = 100;
        let p = 3;
        let grid = lin_space(0.0..=1.0, num_eval);
        let grid = grid.clone().cartesian_product(grid);

        // univariate algorithm
        let start = Instant::now();
        let basis = DeBoor::open_uniform(n, p);
        let basis = DeBoorBi::new(basis.clone(), basis);
        for (u, v) in grid.clone() {
            let _ = black_box(basis.eval_nonzero([u, v]));
        }
        let time_bi = start.elapsed();

        // tensor product algorithm
        let start = Instant::now();
        let basis = DeBoorMulti::open_uniform([n, n], [p, p]);
        for t in grid.clone() {
            let _ = black_box(basis.eval_nonzero(t));
        }
        let time_tp = start.elapsed();

        println!(
            "Took {:?} for {num_eval} basis evaluations (bivariate algorithm).",
            time_bi
        );
        println!(
            "Took {:?} for {num_eval} basis evaluations (tensor product algorithm).",
            time_tp
        );
        println!(
            "Bivariate algorithm is {} % faster than tensor product algorithm",
            (time_tp.as_secs_f64() - time_bi.as_secs_f64()) / time_tp.as_secs_f64() * 100.0
        )
    }

    #[test]
    fn benchmark_pows() {
        let num_eval = 10_000_000;
        let range = lin_space(0f64..=1.0, num_eval);

        // Using powi
        let start = Instant::now();
        for x in range {
            black_box(x.powi(1));
        }
        let time_powi = start.elapsed();

        let range = lin_space(0f64..=1.0, num_eval);

        // Using muls
        let start = Instant::now();
        for x in range {
            black_box(x);
        }
        let time_muls = start.elapsed();

        println!(
            "Took {:?} for {num_eval} power calculations (using powi).",
            time_powi
        );
        println!(
            "Took {:?} for {num_eval} power calculations (manually optimized).",
            time_muls
        );
        println!(
            "powi is {} % slower than optimized algorithm",
            (time_powi.as_secs_f64() - time_muls.as_secs_f64()) / time_muls.as_secs_f64() * 100.0
        )
    }

    #[test]
    fn benchmark_dyn_vs_static() {
        let num_eval = 1_000_000;
        const N: usize = 20;

        let start = Instant::now();
        let coords = DMatrix::<f64>::new_random(N, N);
        let vec = DVector::<f64>::new_random(N);
        for _ in 0..num_eval {
            black_box(&coords * &vec);
        }
        let time_dyn = start.elapsed();

        let start = Instant::now();
        let coords = SMatrix::<f64, N, N>::new_random();
        let vec = SVector::<f64, N>::new_random();
        for _ in 0..num_eval {
            black_box(coords * vec);
        }
        let time_static = start.elapsed();

        println!(
            "Took {:?} for {num_eval:e} {N}x{N} matrix-vector multiplications (dynamic storage).",
            time_dyn
        );
        println!(
            "Took {:?} for {num_eval:e} {N}x{N} matrix-vector multiplications (static storage).",
            time_static
        );
        println!(
            "dynamic is {} % slower than static storage",
            (time_dyn.as_secs_f64() - time_static.as_secs_f64()) / time_static.as_secs_f64()
                * 100.0
        )
    }

    #[test]
    fn benchmark_derivs() {
        let num_eval = 10;
        let grid = lin_space(0.0..=1.0, num_eval);
        let basis = DeBoor::<f64>::open_uniform(200, 3);

        // Algorithm from the nurbs package
        let start = Instant::now();
        for t in grid.clone() {
            let span = KnotSpan::find(&basis.knots, basis.n, t).unwrap();
            let _ = black_box(basis.eval_derivs_with_span::<5>(t, span));
        }
        let time_algo_1 = start.elapsed();

        // Implement algorithm using Curry-Schoenberg basis
        let start = Instant::now();
        for t in grid.clone() {
            // let span = KnotSpan::find(&basis.knots, basis.n, t).unwrap();
            // let _  = black_box(basis.eval_deriv_2(t, span));
        }
        let time_algo_2 = start.elapsed();

        println!(
            "Took {:?} for {num_eval:e} derivative evaluations (algorithm #1).",
            time_algo_1
        );
        println!(
            "Took {:?} for {num_eval:e} derivative evaluations (algorithm #2).",
            time_algo_2
        );
    }

    #[test]
    fn benchmark_vec_dot() {
        let num_eval = 100_000_000;
        const N: usize = 20;

        let nodes = DVector::<f64>::new_random(N);
        let f = |x: f64| x * 2.0;
        let quad = GaussLegendre::new(N).unwrap();
        let w = DVector::from_iterator(N, quad.weights().copied());

        // Vector dot vector
        let start = Instant::now();
        for _ in 0..num_eval {
            // let f_it = nodes.iter().copied().map(f);
            // let f = DVector::from_iterator(N, f_it);
            let f = nodes.map(f);
            let _ = black_box(f.dot(&w));
        }
        let time_dot = start.elapsed();

        // Iter map
        let start = Instant::now();
        for _ in 0..num_eval {
            let f_it = nodes.iter().copied().map(f);
            let _ = black_box(zip(f_it, &w).map(|(fi, wi)| fi * wi).sum::<f64>());
        }
        let time_iter = start.elapsed();

        println!(
            "Took {:?} for {num_eval:e} vector-vector muls (dot product).",
            time_dot
        );

        println!(
            "Took {:?} for {num_eval:e} vector-vector muls (iter map).",
            time_iter
        );

        println!(
            "dot product is {} % slower than using iterators",
            (time_dot.as_secs_f64() - time_iter.as_secs_f64()) / time_iter.as_secs_f64()
                * 100.0
        )
    }
}
