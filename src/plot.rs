use std::collections::BTreeMap;
use std::fmt::Display;
use std::fs;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::iter::zip;
use approx::relative_eq;
use crate::cells::geo::{Cell, CellAllocator, HasBasisCoord};
use crate::cells::edge::LineSegment;
use crate::cells::node::Node;
use crate::cells::quad::{Quad, QuadNodes};
use crate::diffgeo::chart::Chart;
use crate::index::dimensioned::Dimensioned;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::Mesh;
use itertools::Itertools;
use nalgebra::{ComplexField, Const, DefaultAllocator, Dim, DimName, DimNameSub, Dyn, Point, RealField, Scalar, U2};
use numeric_literals::replace_float_literals;
use plotly::common::{ColorScale, ColorScalePalette};
use plotly::layout::Annotation;
use plotly::{Layout, Plot, Scatter, Surface};
use crate::basis::eval::EvalBasisAllocator;
use crate::basis::lin_combination::{EvalFunctionAllocator, LinCombination, SelectCoeffsAllocator};
use crate::basis::local::MeshBasis;
use crate::basis::traits::Basis;
use crate::cells::traits;
use crate::cells::traits::ToElement;

/// Plots the given `faces` of a 2D quad-vertex `msh`.
pub fn plot_faces(msh: &QuadVertexMesh<f64, 2>, faces: impl Iterator<Item=QuadNodes>) -> Plot {
    let mut plot = Plot::new();
    let mut layout = Layout::new();

    for (num, face) in faces.enumerate() {
        for edge in face.edges() {
            let line = LineSegment::from_msh(edge, msh);
            let [pi, pj] = line.vertices;
            let edge = Scatter::new(vec![pi.x, pj.x], vec![pi.y, pj.y]);
            plot.add_trace(edge)
        }
        let quad = face.to_element(&msh.coords);
        let center = quad.centroid();
        let text = Annotation::new()
            .text(num.to_string())
            .show_arrow(false)
            .x(center.x)
            .y(center.y);
        layout.add_annotation(text);
    }

    plot.set_layout(layout);
    plot
}

/// Plots the given `nodes` of a 2D quad-vertex `msh`.
pub fn plot_nodes(msh: &QuadVertexMesh<f64, 2>, nodes: impl Iterator<Item=Node>) -> Plot {
    let mut plot = Plot::new();
    let mut layout = Layout::new();

    for (num, node) in nodes.enumerate() {
        let pos = msh.coords(node);
        let pos_trace = Scatter::new(vec![pos.x], vec![pos.y]);
        plot.add_trace(pos_trace);

        let text = Annotation::new()
            .text(num.to_string())
            .show_arrow(false)
            .x(pos.x)
            .y(pos.y + 0.05);
        layout.add_annotation(text);
    }

    plot.set_layout(layout);
    plot
}

/// Plots the function `f` on the given `elem`/`patch`
/// using `num` evaluation points per parametric direction.
pub fn plot_fn_elem<X, Patch, Elem, F, D>(cell: &Patch, elem: &Elem, f: &F, num: usize, mesh_grid: &D) -> Plot
    where X: Dimensioned<f64, 2> + From<(f64, f64)>,
          Patch: Cell<f64>,
          Patch::GeoMap: Chart<f64, Coord = X, ParametricDim = U2, GeometryDim = U2>,
          F: Fn(&Elem, X) -> f64,
          D: Fn(&Patch, usize) -> (Vec<f64>, Vec<f64>)
{
    let mut plot = Plot::new();
    let phi = cell.geo_map();
    let (u_range, v_range) = mesh_grid(cell, num);

    let mut x = vec![vec![0.0; num]; num];
    let mut y = vec![vec![0.0; num]; num];
    let mut z = vec![vec![0.0; num]; num];

    for (i, &u) in u_range.iter().enumerate() {
        for (j, &v) in v_range.iter().enumerate() {
            // Evaluate patch
            let pos = phi.eval((u, v).into());

            // Set coordinates
            x[i][j] = pos.x;
            y[i][j] = pos.y;
            z[i][j] = f(elem, (u, v).into());
        }
    }

    let surface = Surface::new(z).x(x).y(y)
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis))
        .show_scale(false);
    plot.add_trace(surface.clone());
    plot
}

/// Plots the function `f` on the entire `msh`
/// using `num` evaluation points per parametric direction per element.
pub fn plot_fn_msh<'a, X, Msh, F, D>(msh: &'a Msh, f: &F, num: usize, mesh_grid: D) -> Plot
    where X: Dimensioned<f64, 2> + From<(f64, f64)>,
          Msh: Mesh<'a, f64, 2, 2>,
          <Msh::GeoElem as Cell<f64>>::GeoMap: Chart<f64, Coord = X, ParametricDim = U2, GeometryDim = U2>,
          F: Fn(&Msh::Elem, X) -> f64,
          D: Fn(&Msh::GeoElem, usize) -> (Vec<f64>, Vec<f64>)
{
    let mut plot = Plot::new();
    let elems = msh.elem_iter().collect_vec();

    for elem in elems {
        let elem_plt = plot_fn_elem(&msh.geo_elem(&elem), &elem, f, num, &mesh_grid);
        plot.add_traces(elem_plt.data().iter().cloned().collect_vec());
    }

    plot
}

#[replace_float_literals(T::from_f64(literal).expect("Literal must fit in T"))]
pub fn eval_at_vertices<'a, T, Msh, B>(msh: &'a Msh, fh: LinCombination<T, B, 2>)
    where T: RealField,
          Msh: Mesh<'a, T, 2, 2>,
          Msh::GeoElem: HasBasisCoord<T, B>,
          B: MeshBasis<T, Cell= Msh::Elem, Coord<T> = (T, T)>,
          // B::ElemBasis: Basis<Coord<T> = (T, T)>,
          DefaultAllocator: CellAllocator<T, Msh::GeoElem> + EvalBasisAllocator<B::LocalBasis> + EvalFunctionAllocator<B> + SelectCoeffsAllocator<B::LocalBasis>
{
    let vertices_to_err = msh.elem_iter()
        .flat_map(|elem| {
            let patch = msh.geo_elem(&elem);
            let phi = patch.geo_map();
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)].map(|uv| {
                let p = phi.eval(uv.clone());
                let err = fh.eval_on_elem(&elem, uv).norm();
                (p, err)
            })
        })
        .collect_vec();

    todo!("")
}

/// Writes the coordinates of control points `coords` into a `file`.
pub fn write_coords<T: Scalar + Display, const D: usize>(coords: impl Iterator<Item = Point<T, D>>, file: &mut File) -> Result<(), Box<dyn std::error::Error>> {
    for p in coords {
        let str = p.coords.iter().map(|n| n.to_string()).collect_vec().join(" ");
        writeln!(file, "{str}")?;
    }
    Ok(())
}

/// Writes the element connectivity of `elems` into a `file`.
pub fn write_connectivity<C: traits::CellConnectivity>(elems: impl Iterator<Item = C>, file: &mut File) -> Result<(), Box<dyn std::error::Error>> {
    for elem in elems {
        let str = elem.nodes().iter().map(|n| n.to_string()).collect_vec().join(" ");
        writeln!(file, "{str}")?;
    }
    Ok(())
}

/// Writes all `(x,f(x))` pairs into a `file`,
/// where `x` are the coordinates `coords` and `f` a function.
pub fn write_coords_with_fn<T: Scalar + Display, const D: usize>(coords: impl Iterator<Item = Point<T, D>>, f: impl Iterator<Item = T>, file: &mut File) -> Result<(), Box<dyn std::error::Error>> {
    for (p, f) in zip(coords, f) {
        let coord = p.coords.iter().map(|n| n.to_string()).collect_vec().join(" ");
        writeln!(file, "{coord} {f}")?;
    }
    Ok(())
}