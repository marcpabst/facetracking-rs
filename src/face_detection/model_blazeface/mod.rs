use std::cmp::min;
use std::io::Cursor;
use std::ops::Deref;
use std::sync::Arc;

use image::DynamicImage;
use ndarray::prelude::*;
use ndarray_npy::ReadNpyExt;
use ort::tensor::OrtOwnedTensor;
use ort::{Environment, ExecutionProvider, InMemorySession, SessionBuilder, Value};

use crate::face_detection::{FaceBoundingBox, FaceDetectionModel};

pub struct BlazefaceFaceResult {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    image_width: u32,
    image_height: u32,
    cropped_image_box: (u32, u32, u32, u32),
    score: f32,
}

pub struct BlazefaceModel<'a> {
    pub enviroment: Arc<Environment>,
    pub session: Arc<InMemorySession<'a>>,
    pub anchors: Array2<f32>,
}

impl FaceBoundingBox for BlazefaceFaceResult {
    fn image_size(&self) -> (u32, u32) {
        (self.image_width, self.image_height)
    }

    fn origin(&self) -> (u32, u32) {
        // return origin of face bounding box, taking into account the cropped image
        (
            (self.x * self.cropped_image_box.2 as f32) as u32 + self.cropped_image_box.0,
            (self.y * self.cropped_image_box.3 as f32) as u32 + self.cropped_image_box.1,
        )
    }

    fn width(&self) -> u32 {
        // take into account the cropped image
        (self.width * self.cropped_image_box.2 as f32) as u32
    }

    fn height(&self) -> u32 {
        // take into account the cropped image
        (self.height * self.cropped_image_box.3 as f32) as u32
    }

    fn score(&self) -> f32 {
        self.score
    }

    fn to_tuple(&self) -> (u32, u32, u32, u32) {
        (
            self.origin().0,
            self.origin().1,
            self.width(),
            self.height(),
        )
    }
}

impl BlazefaceModel<'_> {
    pub fn new() -> BlazefaceModel<'static> {
        let enviroment = Environment::builder()
            .with_execution_providers([ExecutionProvider::CoreML(Default::default())])
            .build()
            .unwrap()
            .into_arc();

        let model = include_bytes!("assets/face_detection_back_256x256_float32_opt.onnx");

        let session = SessionBuilder::new(&enviroment)
            .unwrap()
            .with_intra_threads(5)
            .unwrap()
            .with_model_from_memory(model)
            .unwrap();

        // read anchors from file anc.npy
        let anchors_bytes = include_bytes!("assets/anchors.npy");
        // Create fake "file"
        let c = Cursor::new(anchors_bytes);
        //let anchors: Array2<f32> = read_npy("./anc2.npy")?;
        let anchors: Array2<f64> = ReadNpyExt::read_npy(c).unwrap();

        // convert to f32
        let anchors = anchors.map(|x| *x as f32);

        BlazefaceModel {
            enviroment: enviroment,
            session: Arc::new(session),
            anchors,
        }
    }

    fn decode_boxes(
        &self,
        raw_boxes: &Array3<f32>,
        x_scale: f32,
        y_scale: f32,
        w_scale: f32,
        h_scale: f32,
    ) -> Array3<f32> {
        let anchors = &self.anchors;

        let shape = raw_boxes.shape();
        let num_boxes = shape[1];

        let mut boxes = Array3::zeros(Ix3(shape[0], shape[1], shape[2]));

        for i in 0..num_boxes {
            let x_center = &raw_boxes[[0, i, 0]] / x_scale * &anchors[[i, 2]] + &anchors[[i, 0]];
            let y_center = &raw_boxes[[0, i, 1]] / y_scale * &anchors[[i, 3]] + &anchors[[i, 1]];

            let w = &raw_boxes[[0, i, 2]] / w_scale * &anchors[[i, 2]];
            let h = &raw_boxes[[0, i, 3]] / h_scale * &anchors[[i, 3]];

            boxes[[0, i, 0]] = y_center - h / 2.0; // ymin
            boxes[[0, i, 1]] = x_center - w / 2.0; // xmin
            boxes[[0, i, 2]] = y_center + h / 2.0; // ymax
            boxes[[0, i, 3]] = x_center + w / 2.0; // xmax

            for k in 0..6 {
                let offset = 4 + k * 2;
                let keypoint_x =
                    &raw_boxes[[0, i, offset]] / x_scale * &anchors[[i, 2]] + &anchors[[i, 0]];
                let keypoint_y =
                    &raw_boxes[[0, i, offset + 1]] / y_scale * &anchors[[i, 3]] + &anchors[[i, 1]];

                boxes[[0, i, offset]] = keypoint_x;
                boxes[[0, i, offset + 1]] = keypoint_y;
            }
        }

        boxes
    }
}

impl FaceDetectionModel for BlazefaceModel<'_> {
    fn run(&self, image: &DynamicImage) -> Box<dyn FaceBoundingBox> {
        // cut square from image
        // find the smallest side
        let smallest_side = min(image.width(), image.height());

        // cut square from image, centered
        let cropped_image = image.crop_imm(
            (image.width() - smallest_side) / 2,
            (image.height() - smallest_side) / 2,
            smallest_side,
            smallest_side,
        );

        // resize to 256x256
        let cropped_image =
            cropped_image.resize_exact(256, 256, image::imageops::FilterType::Nearest);

        // convert to RGB
        let input = cropped_image.to_rgb8();

        // convert to vector
        let input_: Vec<f32> = input
            .pixels()
            .map(|p| p.0)
            .flatten()
            .map(|p| p as f32 / 255.0)
            .collect();

        let array: CowArray<_, _> = Array::from_shape_vec((1, 256, 256, 3), input_)
            .unwrap()
            .into_dyn()
            .into();

        let inputs = vec![Value::from_array(self.session.allocator(), &array).unwrap()];

        let outputs: Vec<Value> = self.session.run(inputs).unwrap();

        let results01: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
        let results01 = results01.view().deref().clone();

        let results02: OrtOwnedTensor<f32, _> = outputs[1].try_extract().unwrap();
        let results02 = results02.view().deref().clone();

        // concatenate results along existing axis (2)
        let result_array0 = ndarray::concatenate(Axis(1), &[results01, results02]).unwrap();

        // covert to 3d array
        let result_array0 = result_array0.into_dimensionality::<Ix3>().unwrap();

        let results1: OrtOwnedTensor<f32, _> = outputs[2].try_extract().unwrap();
        let results1 = results1.view().deref().clone();

        let results2: OrtOwnedTensor<f32, _> = outputs[3].try_extract().unwrap();
        let results2 = results2.view().deref().clone();

        // concatenate results along existing axis (2)
        let result_array = ndarray::concatenate(Axis(1), &[results1, results2]).unwrap();
        // covert to 3d array
        let result_array = result_array.into_dimensionality::<Ix3>().unwrap();

        // find argmax of flattend result_array0
        let argmax = argmax(result_array0.iter().cloned().collect());
        let score = result_array0[[0, argmax, 0]];

        // get boxes
        let boxes = self.decode_boxes(&result_array, 265.0, 265.0, 265.0, 265.0);

        // extract bounding box coordinates
        let x1 = boxes[[0, argmax, 1]];
        let y1 = boxes[[0, argmax, 0]];
        let x2 = boxes[[0, argmax, 3]];
        let y2 = boxes[[0, argmax, 2]];

        let width = x2 - x1;
        let height = y2 - y1;

        let face_bounding_box = BlazefaceFaceResult {
            x: x1,
            y: y1,
            width,
            height,
            image_width: image.width(),
            image_height: image.height(),
            cropped_image_box: (
                (image.width() - smallest_side) / 2,
                (image.height() - smallest_side) / 2,
                smallest_side,
                smallest_side,
            ),
            score,
        };

        Box::new(face_bounding_box)
    }
}

fn argmax(input: Vec<f32>) -> usize {
    let mut max = 0.0;
    let mut max_index = 0;
    for (i, &item) in input.iter().enumerate() {
        if item > max {
            max = item;
            max_index = i;
        }
    }
    max_index
}
