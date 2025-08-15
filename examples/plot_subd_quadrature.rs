//! This example visualizes the quadrature rule in the irregular subdivision setting,
//! by plotting the quadrature nodes in a truncated series of segments (see [SubdUnitSquare]).

use itertools::Itertools;
use plotly::layout::{Annotation, Axis, Shape, ShapeType};
use plotly::{Layout, Plot, Scatter};
use subd::quadrature::tensor_prod::GaussLegendreMulti;
use subd::quadrature::traits::Quadrature;
use subd::subd::catmull_clark::quadrature::SubdUnitSquareQuad;
use subd::subd::patch::subd_unit_square::SubdUnitSquare;

fn main() {
    // Define quadrature
    let p = 2;
    let m_max = 5;
    let gauss_quad = GaussLegendreMulti::<f64, 2>::with_degrees([p, p]);
    let subd_quad = SubdUnitSquareQuad::new(gauss_quad.clone(), m_max);

    // Get irregular nodes
    let nodes_irr = subd_quad.nodes_elem(&SubdUnitSquare::Irregular);
    let mut plot = Plot::new();
    let mut layout = Layout::new()
        .x_axis(Axis::new().range(vec![0, 1]))
        .y_axis(Axis::new().range(vec![0, 1]));

    // Iteration over all quadrature nodes, chunked in into segments
    for (m, nodes_segment) in nodes_irr.chunks(3*p*p).into_iter().enumerate() {

        // Iteration of nodes inside the 3 quads inside one segment
        for (k, nodes_quad) in nodes_segment.chunks(p*p).into_iter().enumerate() {

            // Iteration over nodes inside on sub-quad
            for (i,(u, v)) in nodes_quad.enumerate() {
                // Add parameter point (u,v)
                let pos_trace = Scatter::new(vec![u], vec![v]);
                plot.add_trace(pos_trace);

                // Add node label. Only draw for first 2 sublevels,
                // otherwise, labels are way too large.
                if m < 2 {
                    let text = Annotation::new()
                        .text(format!("n^({k},{m})_{i}"))
                        .show_arrow(false)
                        .x(u)
                        .y(v + 0.05);
                    layout.add_annotation(text);
                }
            }
        }
    }

    // Add segment lines
    for m in 0..=m_max {
        let z0 = 2f32.powi(-(m as i32) + 1);
        let z1 = 2f32.powi(-(m as i32));
        let vert_line = Shape::new().shape_type(ShapeType::Line)
            .x0(z1).y0(0.0)
            .x1(z1).y1(z0);
        let hor_line = Shape::new().shape_type(ShapeType::Line)
            .x0(0.0).y0(z1)
            .x1(z0).y1(z1);


        layout.add_shape(vert_line);
        layout.add_shape(hor_line);
    }

    plot.set_layout(layout);
    plot.show();
}