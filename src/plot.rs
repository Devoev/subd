use iter_num_tools::lin_space;
use itertools::Itertools;
use nalgebra::{DefaultAllocator, U1};
use nalgebra::allocator::Allocator;
use plotly::{Layout, Plot, Scatter, Surface};
use plotly::common::{ColorScale, ColorScalePalette};
use plotly::layout::Annotation;
use crate::basis::eval::EvalBasis;
use crate::basis::lin_combination::LinCombination;
use crate::basis::local::LocalBasis;
use crate::basis::traits::Basis;
use crate::cells::geo::Cell;
use crate::cells::line_segment::LineSegment;
use crate::cells::node::NodeIdx;
use crate::cells::quad::{Quad, QuadNodes};
use crate::diffgeo::chart::Chart;
use crate::mesh::face_vertex::QuadVertexMesh;
use crate::mesh::traits::Mesh;
use crate::subd::catmull_clark::basis::CatmarkBasis;
use crate::subd::catmull_clark::mesh::CatmarkMesh;
use crate::subd::catmull_clark::patch::{CatmarkPatch, CatmarkPatchNodes};

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
        let quad = Quad::from_msh(face, msh);
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
pub fn plot_nodes(msh: &QuadVertexMesh<f64, 2>, nodes: impl Iterator<Item=NodeIdx>) -> Plot {
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

/// Plots the function `f` on the given `elem`  of the chart `phi`
/// using `num` evaluation points per parametric direction.
pub fn plot_fn_elem<Map, Elem, F>(phi: &Map, elem: &Elem, f: &F, num: usize) -> Plot
    where Map: Chart<f64, (f64, f64), 2, 2>,
          F: Fn(&Elem, (f64, f64)) -> f64,
{
    let mut plot = Plot::new();
    let u_range = lin_space(0.0..=1.0, num);
    let v_range = u_range.clone();

    let mut x = vec![vec![0.0; num]; num];
    let mut y = vec![vec![0.0; num]; num];
    let mut z = vec![vec![0.0; num]; num];

    for (i, u) in u_range.clone().enumerate() {
        for (j, v) in v_range.clone().enumerate() {
            // Evaluate patch
            let pos = phi.eval((u, v));

            // Set coordinates
            x[i][j] = pos.x;
            y[i][j] = pos.y;
            z[i][j] = f(elem, (u, v));
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
pub fn plot_fn_msh<'a, M, F>(msh: &'a M, f: &F, num: usize) -> Plot
    where M: Mesh<'a, f64, (f64, f64), 2, 2>,
          F: Fn(&M::Elem, (f64, f64)) -> f64,
{
    let mut plot = Plot::new();
    let elems = msh.elem_iter().collect_vec();

    for elem in elems {
        let phi = msh.geo_elem(&elem).geo_map();
        let elem_plt = plot_fn_elem(&phi, &elem, f, num);
        plot.add_traces(elem_plt.data().iter().cloned().collect_vec());
    }

    plot
}