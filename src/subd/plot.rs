use crate::subd::mesh::{Face, Node, QuadMesh};
use plotly::layout::Annotation;
use plotly::{Layout, Plot, Scatter};
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