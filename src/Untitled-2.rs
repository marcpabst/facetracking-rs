}

fn decode_boxes(
    raw_boxes: & [f32; 4],
    anchors: &[[f32; 4]],
    x_scale: f32,
    y_scale: f32,
    w_scale: f32,
    h_scale: f32,
) -> [[f32; 4]; 6] {
    let mut boxes: [[f32; 4]; 6] = Default::default();


        let x_center = raw_boxes[0] / x_scale * anchors[2] + anchors[0];
        let y_center = raw_boxes[1] / y_scale * anchors[3] + anchors[1];

        let w = raw_boxes[2] / w_scale * anchors[2];
        let h = raw_boxes[3] / h_scale * anchors[3];

        boxes[0] = y_center - h / 2.0; // ymin
        boxes[1] = x_center - w / 2.0; // xmin
        boxes[2] = y_center + h / 2.0; // ymax
        boxes[3] = x_center + w / 2.0; // xmax

        for k in 0..6 {
            let offset = 4 + k * 2;
            let keypoint_x =
                raw_boxes[offset] / x_scale * anchors[2] + anchors[0];
            let keypoint_y =
                raw_boxes[offset + 1] / y_scale * anchors[3] + anchors[1];
            boxes[offset] = keypoint_x;
            boxes[offset + 1] = keypoint_y;
    }

    boxes
}