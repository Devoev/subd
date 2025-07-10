use plotly::{Layout, Plot, Scatter};
use plotly::layout::Annotation;
use crate::cells::line_segment::LineSegment;
use crate::cells::quad::{Quad, QuadTopo};
use crate::mesh::face_vertex::QuadVertexMesh;

/// Plots the given `faces` of a 2D quad-vertex `msh`.
pub fn plot_faces(msh: &QuadVertexMesh<f64, 2>, faces: impl Iterator<Item=QuadTopo>) -> Plot {
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