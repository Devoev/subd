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
pub mod diffgeo;
mod subd_legacy;
mod cg;

#[cfg(test)]
mod tests {
    use crate::basis::cart_prod;
    use crate::basis::eval::EvalDerivs;
    use crate::basis::local::LocalBasis;
    use crate::basis::space::Space;
    use crate::bspline::de_boor;
    use crate::bspline::de_boor::MultiDeBoor;
    use crate::bspline::space::{BsplineSpace, BsplineSpaceVec2d};
    use crate::bspline::spline_geo::{SplineCurve, SplineGeo};
    use crate::cells::cartesian::CartCell;
    use crate::cells::geo::Cell as GeoCell;
    use crate::cells::quad::QuadTopo;
    use crate::cg::cg;
    use crate::diffgeo::chart::Chart;
    use crate::index::dimensioned::{DimShape, Dimensioned, Strides};
    use crate::index::multi_index::MultiIndex;
    use crate::knots::breaks::Breaks;
    use crate::knots::breaks_with_multiplicity::BreaksWithMultiplicity;
    use crate::knots::knot_vec::KnotVec;
    use crate::mesh::bezier::BezierMesh;
    use crate::mesh::cartesian::CartMesh;
    use crate::mesh::face_vertex::QuadVertexMesh;
    use crate::mesh::traits::{Mesh, MeshTopology};
    use crate::operator::function::assemble_function;
    use crate::operator::hodge::assemble_hodge;
    use crate::operator::laplace::assemble_laplace;
    use crate::quadrature::pullback::{BezierQuad, PullbackQuad};
    use crate::quadrature::tensor_prod::GaussLegendreMulti;
    use crate::quadrature::traits::Quadrature;
    use crate::subd::basis::CatmarkBasis;
    use crate::subd::lin_subd::LinSubd;
    use crate::subd::mesh::CatmarkMesh;
    use crate::subd::patch::CatmarkPatch;
    use crate::subd::space::CatmarkSpace;
    use crate::subd_legacy;
    use gauss_quad::GaussLegendre;
    use iter_num_tools::lin_space;
    use itertools::{iproduct, Itertools};
    use nalgebra::{center, matrix, point, DMatrix, DVector, Dyn, Matrix1, OMatrix, Point, Point2, RealField, RowSVector, SMatrix, SVector, Vector1, U2};
    use nalgebra_sparse::CsrMatrix;
    use num_traits::real::Real;
    use plotters::backend::BitMapBackend;
    use plotters::chart::ChartBuilder;
    use plotters::prelude::{IntoDrawingArea, LineSeries, RED, WHITE};
    use std::collections::BTreeSet;
    use std::f64::consts::PI;
    use std::hint::black_box;
    use std::iter::zip;
    use std::time::Instant;
    use crate::mesh::incidence::{edge_to_node_incidence, face_to_edge_incidence};
    use crate::subd::edge_basis::CatmarkEdgeBasis;

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

        let space = BsplineSpace::<f64, (f64, f64), 2>::new_open_uniform([N, N], [p, p]);
        let strides = space.basis.strides();
    }

    #[test]
    fn splines() {
        let n = 4;
        let p = 2;
        let knots = KnotVec::<f64>::new_open_uniform(n, p);
        let basis = de_boor::DeBoor::new(knots, n, p);
        let basis_2d = MultiDeBoor::new([basis.clone(), basis.clone()]);
        let basis_3d = MultiDeBoor::new([basis.clone(), basis.clone(), basis.clone()]);

        let space_1d = BsplineSpace::new(MultiDeBoor::new([basis.clone()]));
        let space_2d = BsplineSpace::new(basis_2d);
        let space_3d = BsplineSpace::new(basis_3d);

        let t = 0.6;
        println!("{}", space_1d.eval_local(t));
        println!("{}", space_2d.eval_local([t, t]));
        println!("{}", space_3d.eval_local([t, t, t]));
    }

    #[test]
    fn spline_curves() {
        let n = 5;
        let p = 2;
        let space = BsplineSpace::new_open_uniform([n], [p]);
        let coords = matrix![
            -1.0, -0.5, 0.0, 0.5, 1.0;
            0.0, 0.7, 0.0, -0.7, 0.0;
        ];

        let curve = match SplineCurve::from_matrix(coords.transpose(), &space) {
            Ok(curve) => { curve }
            Err(error) => { panic!("{}", error) }
        };
        dbg!(curve.eval(0.0));
    }

    #[test]
    fn vector_basis() {
        // Parameters
        let n = 5;
        let p = 2;

        // Knot vectors
        let knots_p = KnotVec::<f64>::new_open_uniform(n, p);
        let knots_q = KnotVec::<f64>::new_open_uniform(n, p - 1);

        // Basis for components
        let basis_x = MultiDeBoor::from_knots([knots_p.clone(), knots_q.clone()], [n, n], [p, p-1]);
        let basis_y = MultiDeBoor::from_knots([knots_q.clone(), knots_p.clone()], [n, n], [p-1, p]);

        // Vector basis & space
        let basis_2d = cart_prod::Prod::new((basis_x, basis_y));
        let space = BsplineSpaceVec2d::new(basis_2d);

        let x = (0.5, 0.2);
        println!("{}", space.eval_local(x));
    }

    #[test]
    fn spline_surf() {
        let n = 3;
        let p = 2;
        let space = BsplineSpace::new_open_uniform([n, n], [p, p]);

        let control_points = matrix![
            0.0, 0.3, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 0.8;
            0.1, 0.2, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 0.8
        ];

        let coords_rand = OMatrix::<f64, Dyn, U2>::new_random(9);

        let surf = SplineGeo::new(coords_rand, &space).unwrap();

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
        let root_area = BitMapBackend::new("out/spline_surf.png", (800, 800)).into_drawing_area();
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
        let basis = de_boor::DeBoor::new(knots, n, p);
        let basis_3d = MultiDeBoor::<f64, 3>::repeat(basis.clone());
        let space = Space::new(basis_3d.clone());

        let x = 0.8;
        let elem = basis.find_elem(x);
        println!(
            "Derivatives of basis: {}",
            basis.elem_basis(&elem).eval_derivs::<3>(0.8)
        );

        // Jacobian
        let control_points = SMatrix::<f64, 3, 27>::new_random();
        let solid = SplineGeo::from_matrix(control_points.transpose(), &space).unwrap();
        let x = [0.0, 0.2, 0.5];
        let j = solid.eval_diff(x);
        println!("Jacobian matrix: {j} and determinant {}", j.determinant());

        // Function values
        let x = [0.1, 0.0, 0.5];
        println!(
            "Function values of basis: {}",
            space.eval_local([0.1, 0.0, 0.5])
        );

        println!("Gradients of basis: {}", space.eval_grad_local([0.1, 0.0, 0.5]));
    }

    #[test]
    fn bezier_elems() {
        let knots = KnotVec::new(vec![
            0.0, 0.0, 0.0, 0.2, 0.4, 0.4, 0.4, 0.8, 1.0, 1.0, 1.0]
        ).unwrap();

        let basis = de_boor::DeBoor::new(knots.clone(), 7, 3);
        // let quad = GaussLegendre::new(5).unwrap();

        let breaks = Breaks::from_knots(knots.clone());
        let msh = CartMesh::from_breaks([breaks.clone()]);

        // find_span
        println!("--- Finding span indices with `breaks` and `find_span` ---");
        for idx in msh.elems() {
            let elem_idx = idx.0[0];
            let elem = msh.geo_elem(idx);
            let span = basis.find_span(elem.a.x).unwrap();
            let span_idx = span.0;

            println!(
                "Bezier element = [{:.3}, {:.3}] (index = {})",
                elem.a.x, elem.b.x, elem_idx
            );
            println!(
                "Knot span = [{:.3}, {:.3}] (index = {})",
                knots[span_idx],
                knots[span_idx + 1],
                span_idx
            );
        }

        // multiplicity
        println!("--- Find span indices with `breaks_with_multiplicity_iter` ---");
        let (multiplicities, _): (Vec<usize>, Vec<f64>) = knots.breaks_with_multiplicity_iter().unzip();
        let mut k = 0;
        for elem_idx in 0..multiplicities.len() - 1 {
            k += multiplicities[elem_idx] - 1;
            let span_idx = elem_idx + k;

            println!(
                "Bezier element = [{:.3}, {:.3}] (index = {})",
                breaks[elem_idx], breaks[elem_idx + 1], elem_idx
            );
            println!(
                "Knot span = [{:.3}, {:.3}] (index = {})",
                knots[span_idx],
                knots[span_idx + 1],
                span_idx
            );
        }
    }

    #[test]
    fn quadrature() {
        let quad_1d = GaussLegendre::new(2).unwrap();
        let quad_multi = GaussLegendreMulti::<f64, 2>::new([quad_1d.clone(), quad_1d.clone()]);

        // println!("Quadrature nodes = {:?} (in [-1,1]]", quad_multi.nodes_ref().collect_vec());
        // println!("Quadrature weights = {:?} (in [-1,1])", quad_multi.weights_ref().collect_vec());

        let elem = CartCell::new(point![0.2, 0.2], point![0.4, 0.4]);
        println!("Quadrature nodes = {:?} in {:?}", quad_multi.nodes_elem(&elem).collect_vec(), elem);
        println!("Quadrature weights = {:?} in {:?}", quad_multi.weights_elem(&elem).collect_vec(), elem);
    }

    #[test]
    fn mesh() {
        // Define quads
        let quads_regular = vec![
            QuadTopo::from_indices(0, 1, 5, 4),
            QuadTopo::from_indices(1, 2, 6, 5),
            QuadTopo::from_indices(2, 3, 7, 6),
            QuadTopo::from_indices(4, 5, 9, 8),
            QuadTopo::from_indices(5, 6, 10, 9),
            QuadTopo::from_indices(6, 7, 11, 10),
            QuadTopo::from_indices(8, 9, 13, 12),
            QuadTopo::from_indices(9, 10, 14, 13),
            QuadTopo::from_indices(10, 11, 15, 14),
        ];
        let quads_irregular = vec![
            QuadTopo::from_indices(0, 5, 4, 3),
            QuadTopo::from_indices(1, 0, 3, 2),
            QuadTopo::from_indices(2, 3, 16, 17),
            QuadTopo::from_indices(3, 4, 15, 16),
            QuadTopo::from_indices(4, 12, 11, 15),
            QuadTopo::from_indices(5, 13, 12, 4),
            QuadTopo::from_indices(6, 14, 13, 5),
            QuadTopo::from_indices(7, 6, 5, 0),
            QuadTopo::from_indices(8, 7, 0, 9),
            QuadTopo::from_indices(9, 0, 1, 10),
        ];

        // Define coords
        let coords_regular = matrix![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0;
            0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0
        ].transpose();

        // Constructs quad mesh and catmark patch mesh
        let quad_msh = QuadVertexMesh::from_matrix(coords_regular, quads_regular);
        let msh = CatmarkMesh::from_quad_mesh(quad_msh);

        // Print patches
        for elem in msh.elem_iter() {
            let patch = CatmarkPatch::from_msh(&msh, elem);
            let map = patch.geo_map();
            println!("{:?}", elem.as_slice().iter().map(|v| v.0).collect_vec());
            println!("{}", map.eval((0.5, 0.1)));
            println!("{}", map.eval_diff((0.5, 0.1)));
        }
    }

    #[test]
    fn cart_mesh() {
        let breaks = Breaks(vec![0.0, 1.0, 2.0, 3.0]);
        let msh = CartMesh::from_breaks([breaks.clone(), breaks]);

        for idx in msh.elems() {
            let elem = msh.geo_elem(idx);
            println!("Nodes of rectangle {:?}", elem.points().collect_vec());
            println!("Ranges of rectangle {:?}", elem.ranges());
        }
    }

    #[test]
    fn incidence_mats() {
        // Define geo
        let quads_regular = vec![
            QuadTopo::from_indices(0, 1, 2, 3),
        ];

        let coords_regular = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

        // Constructs quad mesh and catmark patch mesh (topological)
        let quad_msh = QuadVertexMesh::from_matrix(coords_regular, quads_regular);
        let mut lin_subd = LinSubd(quad_msh);
        lin_subd.refine();
        // lin_subd.refine();
        let msh = lin_subd.0;

        let g = edge_to_node_incidence(&msh);
        let c = face_to_edge_incidence(&msh);
        let mut div = DMatrix::zeros(g.nrows(), g.ncols());
        for (i, j, &v) in g.triplet_iter() {
            div[(i, j)] = v;
        }
        let mut curl_dual = DMatrix::zeros(c.nrows(), c.ncols());
        for (i, j, &v) in c.triplet_iter() {
            curl_dual[(i, j)] = v;
        }
        let grad = div.transpose();
        let curl = curl_dual.transpose();

        println!("Edge-to-node incidence (div): {}", div);
        println!("Face-to-edge incidence (dual curl): {}", curl_dual);
        println!("Product d*c = {}", div * curl_dual);
        println!("Product c*g = {}", curl * grad);
    }

    #[test]
    fn iga_assembly() {
        // Parameters
        let n_geo = 2;
        let p_geo = 1;
        let n = 6;
        let p = 3;

        // Build knots and space (geometry)
        let knots = KnotVec::new_open_uniform(n_geo, p_geo);
        let basis_uni = de_boor::DeBoor::new(knots.clone(), n_geo, p_geo);
        let basis_geo = MultiDeBoor::new([basis_uni.clone(), basis_uni.clone()]);
        let space_geo = Space::new(basis_geo);

        // Build mapping
        let grid = lin_space(0.0..=1.0, n_geo);
        let c = grid.clone().cartesian_product(grid)
            .flat_map(|(x, y)| [x, y]);
        let c = OMatrix::<f64, U2, Dyn>::from_iterator(n_geo*n_geo, c).transpose();
        let geo_map = SplineGeo::new(c, &space_geo)
            .unwrap_or_else(|e| panic!("{e}"));

        // Build knots and space (basis functions)
        let knots = KnotVec::new_open_uniform(n, p);
        let breaks = Breaks::from_knots(knots.clone());
        let cart_mesh = CartMesh::from_breaks([breaks.clone(), breaks]);
        let msh = BezierMesh::new(cart_mesh, geo_map);
        let basis = de_boor::DeBoor::new(knots, n, p);
        let basis = MultiDeBoor::new([basis.clone(), basis]);
        let space = Space::new(basis);

        // Build quadrature
        let ref_quad = GaussLegendreMulti::with_degrees([6, 6]);
        let quad = BezierQuad::new(ref_quad);
        let mat = assemble_hodge(&msh, &space, quad, |elem| {
            // todo: this should DIRECTLY be implemented in the spline spaces
            let x = msh.geo_elem(*elem).ref_elem.a;
            space.basis.find_elem(x.into_arr())
        });

        // Print
        let mut dense = DMatrix::<f64>::zeros(space.dim(), space.dim());
        for (i, j, &v) in mat.triplet_iter() {
            dense[(i, j)] = v;
        }
        println!("{}", dense);
        println!("Norm ||M|| = {}", dense.norm());
        println!("Rank rk(M) = {} (size is {}x{})", dense.rank(1e-10), dense.nrows(), dense.ncols());
        println!(
            "||M - M^T|| = {} (should be zero)",
            (dense.clone() - dense.transpose()).norm()
        );
        println!(
            "Eigenvalues = {} (should be positive)",
            dense.eigenvalues().unwrap()
        );
    }
    #[test]
    fn subd_assembly() {
        // Define quads
        let quads_regular = vec![
            QuadTopo::from_indices(0, 1, 5, 4),
            QuadTopo::from_indices(1, 2, 6, 5),
            QuadTopo::from_indices(2, 3, 7, 6),
            QuadTopo::from_indices(4, 5, 9, 8),
            QuadTopo::from_indices(5, 6, 10, 9),
            QuadTopo::from_indices(6, 7, 11, 10),
            QuadTopo::from_indices(8, 9, 13, 12),
            QuadTopo::from_indices(9, 10, 14, 13),
            QuadTopo::from_indices(10, 11, 15, 14),
        ];

        let quads_regular = vec![
            QuadTopo::from_indices(0, 1, 2, 3),
        ];

        // Define coords
        let coords_regular = matrix![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0;
            0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0
        ].transpose();

        let coords_regular = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

        // Constructs quad mesh and catmark patch mesh (topological)
        let quad_msh = QuadVertexMesh::from_matrix(coords_regular, quads_regular);
        let mut lin_subd = LinSubd(quad_msh);
        lin_subd.refine();
        lin_subd.refine();
        let msh = CatmarkMesh::from_quad_mesh(lin_subd.0);

        // Construct basis and space
        let basis = CatmarkBasis(&msh);
        let space = Space::new(basis);

        let ref_quad = GaussLegendreMulti::with_degrees([3, 3]);
        let quad = PullbackQuad::new(ref_quad);

        // Load function
        let f = |p: Point<f64, 2>| Matrix1::new(p.coords.norm_squared());

        // Assembly
        let mass = assemble_hodge(&msh, &space, quad.clone(), |&elem| elem.clone());
        let stiffness = assemble_laplace(&msh, &space, quad.clone(), |&elem| elem.clone());
        let load = assemble_function(&msh, &space, quad, f, |&elem| elem.clone());

        // Mass matrix checks
        let mut mass_dense = DMatrix::<f64>::zeros(space.dim(), space.dim());
        for (i, j, &v) in mass.triplet_iter() {
            mass_dense[(i, j)] = v;
        }
        println!("{}", mass_dense);
        println!(
            "Eigenvalues = {} (should be positive)",
            mass_dense.eigenvalues().unwrap()
        );
        println!(
            "||M - M^T|| = {} (should be zero)",
            (mass_dense.clone() - mass_dense.transpose()).norm()
        );
        println!("Norm ||M|| = {}", mass_dense.norm());
        println!("Rank rk(M) = {} (size is {}x{})", mass_dense.rank(1e-10), mass_dense.nrows(), mass_dense.ncols());

        // Stiffness matrix checks
        let mut stiffness_dense = DMatrix::<f64>::zeros(space.dim(), space.dim());
        for (i, j, &v) in stiffness.triplet_iter() {
            stiffness_dense[(i, j)] = v;
        }
        println!("{}", stiffness_dense);
        println!(
            "Eigenvalues = {} (should be positive)",
            stiffness_dense.eigenvalues().unwrap()
        );
        println!(
            "||M - M^T|| = {} (should be zero)",
            (stiffness_dense.clone() - stiffness_dense.transpose()).norm()
        );
        println!("Norm ||M|| = {}", stiffness_dense.norm());
        println!("Rank rk(M) = {} (size is {}x{})", stiffness_dense.rank(1e-10), stiffness_dense.nrows(), stiffness_dense.ncols());

        // Load vector checks
        println!("{}", load);
        println!("Norm ||f|| = {}", load.norm());
    }

    #[test]
    fn subd_edge_assembly() {
        // Define geo
        let quads_regular = vec![
            QuadTopo::from_indices(0, 1, 2, 3),
        ];

        let coords_regular = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

        // Constructs quad mesh and catmark patch mesh (topological)
        let quad_msh = QuadVertexMesh::from_matrix(coords_regular, quads_regular);
        let mut lin_subd = LinSubd(quad_msh);
        lin_subd.refine();
        lin_subd.refine();
        let msh = CatmarkMesh::from_quad_mesh(lin_subd.0);

        // Construct basis and space
        let basis = CatmarkEdgeBasis(&msh);
        let space = Space::<f64, (f64, f64), _, 2>::new(basis);

        let ref_quad = GaussLegendreMulti::with_degrees([3, 3]);
        let quad = PullbackQuad::new(ref_quad);

        // Assembly
        let mass = assemble_hodge(&msh, &space, quad.clone(), |&elem| elem.clone());

        // Mass matrix checks
        let mut mass_dense = DMatrix::<f64>::zeros(space.dim(), space.dim());
        for (i, j, &v) in mass.triplet_iter() {
            mass_dense[(i, j)] = v;
        }
        println!("{}", mass_dense);
        println!(
            "Eigenvalues = {} (should be positive)",
            mass_dense.eigenvalues().unwrap()
        );
        println!(
            "||M - M^T|| = {} (should be zero)",
            (mass_dense.clone() - mass_dense.transpose()).norm()
        );
        println!("Norm ||M|| = {}", mass_dense.norm());
        println!("Rank rk(M) = {} (size is {}x{})", mass_dense.rank(1e-10), mass_dense.nrows(), mass_dense.ncols());
    }

    #[test]
    fn subd_dr_neu_square() {
        // Define problem
        let u = |p: Point2<f64>| Vector1::new((p.x * PI).cos() * (p.y * PI).cos());
        let f = |p: Point2<f64>| (2.0 * PI.powi(2) + 1.0) * u(p);

        let coords_square = matrix![
            0.0, 0.0, 1.0, 1.0;
            0.0, 1.0, 1.0, 0.0
        ].transpose();

        // Define mesh
        let quads = vec![QuadTopo::from_indices(0, 1, 2, 3)];
        let quad_msh = QuadVertexMesh::from_matrix(coords_square, quads);
        let mut lin_msh = LinSubd(quad_msh);
        lin_msh.refine();
        lin_msh.refine();
        // lin_msh.refine();
        // lin_msh.refine();
        let msh = CatmarkMesh::from_quad_mesh(lin_msh.0);

        // Define space
        let basis = CatmarkBasis(&msh);
        let space = CatmarkSpace::new(basis);

        // Define quadrature
        let p = 2;
        let ref_quad = GaussLegendreMulti::with_degrees([p, p]);
        let quad = PullbackQuad::new(ref_quad);

        // Assemble system
        let f = assemble_function(&msh, &space, quad.clone(), f, |&elem| elem.clone());
        let m_coo = assemble_hodge(&msh, &space, quad.clone(), |&elem| elem.clone());
        let k_coo = assemble_laplace(&msh, &space, quad.clone(), |&elem| elem.clone());
        let m = CsrMatrix::from(&m_coo);
        let k = CsrMatrix::from(&k_coo);

        // Solve system
        let uh = cg(&(k + &m), &f, f.clone(), f.len(), 1e-10);

        // Calculate error
        let u = DVector::from_iterator(msh.num_nodes(), msh.coords.iter().map(|&p| u(p).x));
        let du = &u - uh;
        let err_l2 = (&m * &du).dot(&du);
        let norm_l2 = (&m * &u).dot(&u);

        println!("Absolute L2 error ||u - u_h||_2 = {:.7}", err_l2);
        println!("Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.5}%", err_l2 / norm_l2 * 100.0);
    }

    #[test]
    pub fn subd_poisson_dir_pentagon() {
        // Define geometry
        let r = 1;
        let n = 5;
        let phi = 2.0*PI / n as f64;

        let mut coords = vec![point![0.0, 0.0]];

        for i in 0..n {
            let phi_i = phi * i as f64;
            let phi_j = phi * (i + 1) as f64;
            let pi = point![phi_i.cos(), phi_i.sin()];
            let pj = point![phi_j.cos(), phi_j.sin()];
            coords.push(pi);
            coords.push(center(&pi, &pj));
        }

        // Define problem
        let points = coords.iter().skip(1).step_by(2).collect_vec();
        let xs = points.iter().map(|p| p.x).collect_vec();
        let ys = points.iter().map(|p| p.y).collect_vec();
        let a = ys.iter().circular_tuple_windows().map(|(yi, yj)| yi - yj).collect_vec();
        let b = xs.iter().circular_tuple_windows().map(|(xi, xj)| xi - xj).collect_vec();
        let c = points.iter().circular_tuple_windows().map(|(pi, pj)| pi.x * pj.y - pj.x * pi.y).collect_vec();

        // Define solution
        let eval_factor = |i: usize, x: f64, y: f64| a[i]*x + b[i]*y + c[i];
        let eval_product = |k: usize, j: usize, x: f64, y: f64| {
            (0..5).filter(|&i| i != k && i != j)
                .map(|i| eval_factor(i, x, y))
                .product::<f64>()
        };
        let eval_deriv = |coeffs: &Vec<f64>, x: f64, y: f64| {
            (0..5).cartesian_product(0..5)
                .filter(|(k, j)| k != j)
                .map(|(k, j)| eval_product(k, j, x, y) * coeffs[k] * coeffs[j])
                .sum::<f64>()
        };

        let u = |p: Point2<f64>| {
            (0..5).map(|i| eval_factor(i, p.x, p.y)).product::<f64>()
        };

        let u_dxx = |p: Point2<f64>| eval_deriv(&a, p.x, p.y);
        let u_dyy = |p: Point2<f64>| eval_deriv(&b, p.x, p.y);
        let f = |p: Point2<f64>| Vector1::new(-u_dxx(p) - u_dyy(p));

        // Define mesh
        let faces = vec![
            QuadTopo::from_indices(0, 10, 1, 2),
            QuadTopo::from_indices(0, 2, 3, 4),
            QuadTopo::from_indices(0, 4, 5, 6),
            QuadTopo::from_indices(0, 6, 7, 8),
            QuadTopo::from_indices(0, 8, 9, 10),
        ];
        let quad_msh = QuadVertexMesh::new(coords, faces);

        let mut lin_msh = LinSubd(quad_msh.clone());
        lin_msh.refine();
        lin_msh.refine();
        let msh = CatmarkMesh::from_quad_mesh(lin_msh.0);

        // Define space
        let basis = CatmarkBasis(&msh);
        let space = CatmarkSpace::new(basis);

        // Define quadrature
        let p = 2;
        let ref_quad = GaussLegendreMulti::with_degrees([p, p]);
        let quad = PullbackQuad::new(ref_quad);

        // Assemble system
        let f = assemble_function(&msh, &space, quad.clone(), f, |&elem| elem.clone());
        let m_coo = assemble_hodge(&msh, &space, quad.clone(), |&elem| elem.clone());
        let k_coo = assemble_laplace(&msh, &space, quad.clone(), |&elem| elem.clone());
        let m = CsrMatrix::from(&m_coo);
        let k = CsrMatrix::from(&k_coo);

        // Deflate system (homogeneous BC)
        let idx = (0..msh.num_nodes()).collect::<BTreeSet<_>>();
        let idx_bc = msh.boundary_nodes().map(|n| n.0).collect::<BTreeSet<_>>();
        let idx_dof = idx.difference(&idx_bc).collect::<BTreeSet<_>>();

        // todo: using get_entry is expensive. Implement BC differently
        let f_dof = DVector::from_iterator(idx_dof.len(), idx_dof.iter().map(|&&i| f[i]));
        let k_dof_dof = DMatrix::from_iterator(idx_dof.len(), idx_dof.len(), iproduct!(idx_dof.iter(), idx_dof.iter())
            .map(|(&&i, &&j)| k.get_entry(i, j).unwrap().into_value())
        );
        let k_dof_bc = DMatrix::from_iterator(idx_dof.len(), idx_bc.len(), iproduct!(idx_dof.iter(), idx_bc.iter())
            .map(|(&&i, &j)| k.get_entry(i, j).unwrap().into_value())
        );

        let f = f_dof;
        let k = CsrMatrix::from(&k_dof_dof);

        // Solve system
        let mut uh = DVector::zeros(msh.num_nodes());
        let uh_dof = cg(&k, &f, f.clone(), f.len(), 1e-10);

        // Inflate system
        for (i_local, &&i) in idx_dof.iter().enumerate() {
            uh[i] = uh_dof[i_local];
        }

        // Calculate error
        let u = DVector::from_iterator(msh.num_nodes(), msh.coords.iter().map(|&p| u(p)));
        let du = &u - uh;
        let err_l2 = (&m * &du).dot(&du).sqrt();
        let norm_l2 = (&m * &u).dot(&u).sqrt();

        println!("Absolute L2 error ||u - u_h||_2 = {:.7}", err_l2);
        println!("Relative L2 error ||u - u_h||_2 / ||u||_2 = {:.5}%", err_l2 / norm_l2 * 100.0);
    }

    #[test]
    fn benchmark_de_boor_vs_mat_mat() {
        let num_eval = 1000;
        let grid = lin_space(0.0..=1.0, num_eval);
        let grid = grid.clone().cartesian_product(grid);

        // de Boor algorithm
        let start = Instant::now();
        let space = BsplineSpace::new_open_uniform([4, 4], [3, 3]);
        for (u, v) in grid.clone() {
            let _ = black_box(space.eval_local([u, v]));
            // println!("{}", eval.norm());
        }
        let time_de_boor = start.elapsed();

        // matrix-matrix (for catmull clark)
        let start = Instant::now();
        for (u, v) in grid.clone() {
            let eval = black_box(subd_legacy::basis::eval_regular(u, v));
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
        let n = 30;
        let p = 3;

        // univariate algorithm
        let start = Instant::now();
        let knots = KnotVec::new_open_uniform(n, p);
        let basis = de_boor::DeBoor::new(knots, n, p);
        let space = Space::<_,_,_,1>::new(basis);
        for t in grid.clone() {
            let _ = black_box(space.eval_local(t));
        }
        let time_uni = start.elapsed();

        // tensor product algorithm
        let start = Instant::now();
        let space = BsplineSpace::new_open_uniform([n], [p]);
        for t in grid.clone() {
            let _ = black_box(space.eval_local(t));
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

    #[test]
    fn benchmark_cast() {
        let num_eval = 1_000_000_000;
        const N: usize = 8;

        type NC = f64;
        type C = f32;
        type Mat<T> = SMatrix::<T, N, N>;
        type Vec<T> = RowSVector<T, N>;

        let mat = Mat::<C>::new_random();
        let vec = Vec::<NC>::new_random();

        fn cast<T: RealField>(m: Mat<C>, x: Vec<T>) -> Vec<T> {
            x * m.cast()
        }

        fn no_cast<T: RealField>(m: Mat<T>, x: Vec<T>) -> Vec<T> {
            x * m
        }

        let start = Instant::now();
        for _ in 0..num_eval {
            let _ = black_box(cast(mat, vec));
            // println!("{}", eval.norm());
        }
        let time_cast = start.elapsed();

        let start = Instant::now();
        let mat = mat.cast();
        for _ in 0..num_eval {
            let _ = black_box(no_cast(mat, vec));
            // println!("{}", eval.norm());
        }
        let time_no_cast = start.elapsed();

        println!(
            "Took {:?} for {num_eval:e} {N}x{N} matrix-vector muls (with cast).",
            time_cast
        );

        println!(
            "Took {:?} for {num_eval:e} {N}x{N} matrix-vector muls (without map).",
            time_no_cast
        );

        println!(
            "Casting is {} % slower than no casting",
            (time_cast.as_secs_f64() - time_no_cast.as_secs_f64()) / time_no_cast.as_secs_f64() * 100.0
        )
    }

    #[test]
    fn benchmark_find_span() {
        // Parameters
        const NUM_BASIS: usize = 10_000_000;
        const DEGREE: usize = 6;
        const NUM_SPANS: f64 = 1_000_000.0;

        // Knot vector
        let random = DVector::<f64>::new_random(NUM_BASIS - DEGREE - 1).map(|v| (v * NUM_SPANS).round() / NUM_SPANS);
        let internal = KnotVec::from_unsorted(random.as_slice().to_vec());
        let knots = KnotVec::new_open(internal, DEGREE);

        let basis = de_boor::DeBoor::new(knots.clone(), NUM_BASIS, DEGREE);

        // find_span
        let start = Instant::now();
        let breaks = Breaks::from_knots(knots.clone());
        let msh = CartMesh::from_breaks([breaks.clone()]);
        let mut spans_1 = vec![0; breaks.len() - 1];
        for idx in msh.elems() {
            let elem = msh.geo_elem(idx);
            let span = black_box(basis.find_span(elem.a.x).unwrap());
            let span_idx = span.0;

            spans_1[idx.0[0]] = span_idx;
        };

        let time_find_span = start.elapsed();

        // multiplicity
        let start = Instant::now();
        let breaks_with_multi = BreaksWithMultiplicity::from_knots(knots);
        let spans_2 = breaks_with_multi.knot_spans()
            .iter().map(|span| span.0)
            .collect_vec();

        let time_multi = start.elapsed();

        // Check if both span indices are actually equal
        assert_eq!(spans_1, spans_2);

        // Compare runtimes
        println!(
            "Took {:?} for finding {:e} spans with {NUM_BASIS:e} basis functions of degree {DEGREE} (with find_span).",
            time_find_span,
            spans_1.len()
        );

        println!(
            "Took {:?} for finding {:e} spans with {NUM_BASIS:e} basis functions of degree {DEGREE} (with breaks_with_multiplicity_iter).",
            time_multi,
            spans_2.len()
        );

        println!(
            "Using `find_span` is {} % slower than using `breaks_with_multiplicity_iter`",
            (time_find_span.as_secs_f64() - time_multi.as_secs_f64()) / time_multi.as_secs_f64() * 100.0
        )
    }
}
