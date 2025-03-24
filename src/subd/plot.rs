use iter_num_tools::lin_space;
use itertools::Itertools;
use crate::subd::mesh::{Face, Node, QuadMesh};
use plotly::layout::Annotation;
use plotly::{Layout, Plot, Scatter, Surface};
use crate::subd::face::edges_of_face;

/// Plots the given `faces` of a `msh`.
pub fn plot_faces(msh: &QuadMesh<f64>, faces: impl Iterator<Item=Face>) -> Plot {
    let mut plot = Plot::new();
    let mut layout = Layout::new();

    for (num, face) in faces.enumerate() {
        for edge in edges_of_face(face) {
            let [pi, pj] = msh.nodes_of_edge(&edge);
            let edge = Scatter::new(vec![pi.x, pj.x], vec![pi.y, pj.y]);
            plot.add_trace(edge)
        }
        let center = msh.centroid(face);
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

/// Plots the given `nodes` of the `msh`.
pub fn plot_nodes(msh: &QuadMesh<f64>, nodes: impl Iterator<Item=Node>) -> Plot {
    let mut plot = Plot::new();
    let mut layout = Layout::new();

    for (num, node) in nodes.enumerate() {
        let pos = msh.node(node);
        let pos_trace = Scatter::new(vec![pos.x], vec![pos.y]);
        plot.add_trace(pos_trace);

        let text = Annotation::new()
            .text(num.to_string())
            .show_arrow(false)
            .x(pos.x)
            .y(pos.y + 0.2);
        layout.add_annotation(text);
    }

    plot.set_layout(layout);
    plot
}

/// Plots the scalar function `b: (0,1)² ⟶ ℝ` on the parametric domain.
pub fn plot_fn(b: fn(f64, f64) -> f64, num: usize) -> Plot {
    let mut plot = Plot::new();
    let min = 1e-5;
    let u_range = lin_space(min..=1.0, num);
    let v_range = u_range.clone();
    
    // Calculate data
    let mut z = vec![vec![0.0; num]; num];
    for (i, u) in u_range.clone().enumerate() {
        for (j, v) in v_range.clone().enumerate() {
            z[i][j] = b(u, v);
        }
    }
    
    let trace = Surface::new(z).x(u_range.collect()).y(v_range.collect());
    plot.add_trace(trace);
    
    plot
}