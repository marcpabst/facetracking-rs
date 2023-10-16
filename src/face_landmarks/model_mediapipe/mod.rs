mod canonical_face_model;
use canonical_face_model::{
    CANONICAL_FACE_MODEL_POINTS, CANONICAL_FACE_MODEL_WEIGHTS,
    CANONICAL_FACE_MODEL_WEIGHTS_LANDMARK_IDS,
};
use nalgebra::DVector;

use crate::face_landmarks::{FaceLandmarksModel, MetricFaceLandmarks, ScreenFaceLandmarks};

mod procrustes_solver;
use std::ops::Deref;
use std::sync::Arc;

use image::DynamicImage;
use nalgebra::{Dyn, Matrix, Matrix3xX, VecStorage, Vector3, U3};
use ndarray::{Array, CowArray};
use ort::tensor::OrtOwnedTensor;
use ort::{Environment, ExecutionProvider, InMemorySession, SessionBuilder, Value};
use procrustes_solver::FloatPrecisionProcrustesSolver;

use self::procrustes_solver::ProcrustesSolver;

type Point3 = nalgebra::Point3<f32>;
type Point2 = nalgebra::Point2<f32>;

type UnitVector3 = nalgebra::Unit<Vector3<f32>>;
type TransformationMatrix = nalgebra::Matrix3<f32>;

type Matrix3X<T> = Matrix<T, U3, Dyn, VecStorage<T, U3, Dyn>>;

// struct that stores face landmarks (478 3D points)
pub struct MediapipeFaceLandmarks {
    // the 3D points of the face mesh
    points: Vec<Point3>,
    //metric_points: Option<Vec<Point3>>,
    face_bbox: (u32, u32, u32, u32),
    face_confidence: f32,
    image_width: u32,
    image_height: u32,
}

pub struct MediapipeMetricFaceLandmarks {
    // the 3D points of the face mesh
    points: Vec<Point3>,
}

pub enum LandmarkPoint {
    LeftPupil,
    RightPupil,
    LeftEyeTop,
    LeftEyeBottom,
    RightEyeTop,
    RightEyeBottom,
    LeftCheek,
    RightCheek,
    NoseTip,
}

pub enum LandmarkMesh {
    LeftEye,
    RightEye,
    LeftPupil,
    RightPupil,
}

// construct from 1D vector of 478 3D points (flattened 3D points)

impl MediapipeFaceLandmarks {
    pub fn from_vec(
        points: Vec<f32>,
        face_bbox: (u32, u32, u32, u32),
        face_confidence: f32,
        image_width: u32,
        image_height: u32,
    ) -> MediapipeFaceLandmarks {
        let mut landmarks = MediapipeFaceLandmarks {
            points: Vec::new(),
            //metric_points: None,
            face_bbox: face_bbox,
            face_confidence: face_confidence,
            image_width: image_width,
            image_height: image_height,
        };
        for i in 0..points.len() / 3 {
            landmarks.points.push(Point3::new(
                points[i * 3],
                points[i * 3 + 1],
                points[i * 3 + 2],
            ));
        }
        landmarks
    }

    pub fn get_point(&self, point: LandmarkPoint) -> Point3 {
        match point {
            LandmarkPoint::LeftPupil => self.points[468],
            LandmarkPoint::RightPupil => self.points[473],
            LandmarkPoint::LeftEyeTop => self.points[159],
            LandmarkPoint::LeftEyeBottom => self.points[145],
            LandmarkPoint::RightEyeTop => self.points[386],
            LandmarkPoint::RightEyeBottom => self.points[374],
            LandmarkPoint::LeftCheek => self.points[127],
            LandmarkPoint::RightCheek => self.points[356],
            LandmarkPoint::NoseTip => self.points[6],
        }
    }

    pub fn get_mesh(&self, mesh: LandmarkMesh) -> Vec<Point3> {
        match mesh {
            // [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
            LandmarkMesh::LeftEye => [
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33,
            ]
            .iter()
            .map(|&i| self.points[i])
            .collect(),
            // [362, 382, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 362]
            LandmarkMesh::RightEye => [
                362, 382, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 362,
            ]
            .iter()
            .map(|&i| self.points[i])
            .collect(),
            // [478 - 9, 478 - 8, 478 - 7, 478 - 6]
            LandmarkMesh::LeftPupil => [478 - 9, 478 - 8, 478 - 7, 478 - 6]
                .iter()
                .map(|&i| self.points[i])
                .collect(),
            //[478 - 4, 478 - 3, 478 - 2, 478 - 1]
            LandmarkMesh::RightPupil => [478 - 4, 478 - 3, 478 - 2, 478 - 1]
                .iter()
                .map(|&i| self.points[i])
                .collect(),
        }
    }

    // returns the 2d location of the iris (left and right)
    // the coordinatee system is spanned by the direction vectors of the face
    pub fn get_iris_locations(&self) -> (Point2, Point2) {
        let left_pupil = self.get_point(LandmarkPoint::LeftPupil);
        let right_pupil = self.get_point(LandmarkPoint::RightPupil);

        // get direction vectors
        let (vx, vy, _vz) = self.get_direction_vectors();

        // project the pupil points onto the plane spanned by the direction vectors (vx, vy)
        let left_pupil_2d = Point2::new(left_pupil.coords.dot(&vx), left_pupil.coords.dot(&vy));
        let right_pupil_2d = Point2::new(right_pupil.coords.dot(&vx), right_pupil.coords.dot(&vy));

        (left_pupil_2d, right_pupil_2d)
    }

    pub fn get_direction_vectors(
        &self,
    ) -> (
        UnitVector3, // x
        UnitVector3, // y
        UnitVector3, // z
    ) {
        // These points lie *almost* on the same line in uv coordinates
        let p1 = self.get_point(LandmarkPoint::LeftCheek);
        let p2 = self.get_point(LandmarkPoint::RightCheek);
        let p3 = self.get_point(LandmarkPoint::NoseTip);

        let vhelp = p3 - p1;
        let vx_d = p2 - p1;
        let vy_d = vx_d.cross(&vhelp);
        let vx = vx_d.normalize();
        let vy = vy_d.normalize();
        let vz = vx_d.cross(&vy_d);

        (
            UnitVector3::new_normalize(vx),
            UnitVector3::new_normalize(vy),
            UnitVector3::new_normalize(vz),
        )
    }

    pub fn get_rotation_matrix(&self) -> TransformationMatrix {
        let (vx, vy, vz) = self.get_direction_vectors();

        TransformationMatrix::from_columns(&[vx.into_inner(), vy.into_inner(), vz.into_inner()])
    }

    pub fn get_distance(&self, point1: LandmarkPoint, point2: LandmarkPoint) -> f32 {
        let p1 = self.get_point(point1);
        let p2 = self.get_point(point2);
        (p1 - p2).norm()
    }

    fn get_face_bbox_from_landmarks(&self) -> (u32, u32, u32, u32) {
        let mut x_min = std::f32::MAX;
        let mut y_min = std::f32::MAX;
        let mut x_max = 0.0;
        let mut y_max = 0.0;

        for p in self.points.iter() {
            let x = p.x;
            let y = p.y;

            if x < x_min {
                x_min = x;
            }
            if y < y_min {
                y_min = y;
            }
            if x > x_max {
                x_max = x;
            }
            if y > y_max {
                y_max = y;
            }
        }

        // current coordinates are in relation tot the current face bounding box

        x_min = x_min * self.face_bbox.2 as f32 + self.face_bbox.0 as f32;
        y_min = y_min * self.face_bbox.3 as f32 + self.face_bbox.1 as f32;

        x_max = x_max * self.face_bbox.2 as f32 + self.face_bbox.0 as f32;
        y_max = y_max * self.face_bbox.3 as f32 + self.face_bbox.1 as f32;

        let w = x_max - x_min;
        let h = y_max - y_min;

        // find center of box
        let center = (x_min + w / 2.0, y_min + h / 2.0);

        let sze = w.max(h);

        // re-center the box
        let x_min = center.0 - sze / 2.0;
        let y_min = center.1 - sze / 2.0;

        (x_min as u32, y_min as u32, sze as u32, sze as u32)
    }

    fn landmark_list_as_eigen_matrix(&self, points: Vec<Point3>) -> Matrix3X<f32> {
        let mut eigen_matrix = Matrix3X::zeros(points.len());

        for i in 0..points.len() {
            let landmark = &points[i as usize];
            eigen_matrix[(0, i)] = landmark.x;
            eigen_matrix[(1, i)] = landmark.y;
            eigen_matrix[(2, i)] = landmark.z;
        }

        eigen_matrix
    }

    fn get_weights_vector() -> DVector<f32> {
        let mut weights = DVector::zeros(468);

        for i in 0..CANONICAL_FACE_MODEL_WEIGHTS_LANDMARK_IDS.len() {
            let j = CANONICAL_FACE_MODEL_WEIGHTS_LANDMARK_IDS[i];
            weights[j as usize] = CANONICAL_FACE_MODEL_WEIGHTS[i as usize];
        }

        weights
    }

    fn project_xy(
        &self,
        pcf: &PerspectiveCameraFrustum,
        mut landmarks: Matrix3xX<f32>,
    ) -> Matrix3xX<f32> {
        let x_scale = pcf.right - pcf.left;
        let y_scale = pcf.top - pcf.bottom;
        let x_translation = pcf.left;
        let y_translation = pcf.bottom;

        // if origin_point_location_ == OriginPointLocation::TopLeftCorner {
        //     landmarks.row_mut(1).map(|mut val| *val = 1.0 - *val);
        // }

        let nscale = &Vector3::new(x_scale, y_scale, x_scale);
        for mut col in landmarks.column_iter_mut() {
            col.component_mul_assign(nscale);
        }

        landmarks.column_mut(0).add_scalar_mut(x_translation);
        landmarks.column_mut(1).add_scalar_mut(y_translation);

        landmarks
    }

    fn unproject_xy(
        &self,
        pcf: &PerspectiveCameraFrustum,
        mut landmarks: Matrix3xX<f32>,
    ) -> Matrix3xX<f32> {
        let near_inverse = 1.0 / pcf.near;

        // Perform the element-wise operations
        for i in 0..landmarks.ncols() {
            landmarks[(0, i)] /= near_inverse;
            landmarks[(1, i)] /= near_inverse;
        }

        landmarks
    }

    fn estimate_scale(
        &self,
        landmarks: &Matrix3xX<f32>,
        canonical_landmarks: &Matrix3xX<f32>,
        canonical_landmarks_weights: &DVector<f32>,
        procrustes_solver: &FloatPrecisionProcrustesSolver,
    ) -> Result<f32, &'static str> {
        // todo> change this for performance
        // convert CANONICAL_FACE_MODEL_POINTS into a Matrix3X

        let transform_mat = procrustes_solver
            .solve_weighted_orthogonal_problem(
                &canonical_landmarks,
                &landmarks,
                &canonical_landmarks_weights,
            )
            .unwrap();

        Ok(transform_mat.column(0).norm())
    }

    fn move_and_rescale_z(
        &self,
        pcf: &PerspectiveCameraFrustum,
        depth_offset: f32,
        scale: f32,
        mut landmarks: Matrix3xX<f32>,
    ) -> Matrix3xX<f32> {
        for i in 0..landmarks.ncols() {
            landmarks[(2, i)] = (landmarks[(2, i)] - depth_offset + pcf.near) / scale;
        }
        landmarks
    }

    fn change_handedness(&self, mut landmarks: Matrix3xX<f32>) -> Matrix3xX<f32> {
        landmarks.row_mut(2).scale_mut(-1.0);
        landmarks
    }
}

impl ScreenFaceLandmarks for MediapipeFaceLandmarks {
    fn get_landmark(&self, index: usize) -> Option<Point3> {
        // convert to pixel coordinates
        let mut p = self.points[index].clone();

        p.x = p.x * self.face_bbox.2 as f32 + self.face_bbox.0 as f32;
        p.y = p.y * self.face_bbox.3 as f32 + self.face_bbox.1 as f32;

        Some(p)
    }

    fn get_landmarks(&self) -> Vec<Point3> {
        // convert to pixel coordinates
        let mut points = self.points.clone();

        for p in points.iter_mut() {
            p.x = p.x * self.face_bbox.2 as f32 + self.face_bbox.0 as f32;
            p.y = p.y * self.face_bbox.3 as f32 + self.face_bbox.1 as f32;
        }

        points
    }

    fn get_face_bbox(&self) -> (u32, u32, u32, u32) {
        self.face_bbox
    }

    fn get_confidence(&self) -> f32 {
        self.face_confidence
    }

    fn len(&self) -> usize {
        self.points.len()
    }

    fn to_metric(&self) -> Box<(dyn MetricFaceLandmarks + 'static)> {
        let points = self.points[0..468].to_vec();
        // make sure there are as many screen landmarks as canonical landmarks
        assert_eq!(
            points.len(),
            CANONICAL_FACE_MODEL_POINTS.transpose().ncols()
        );

        let pcf = PerspectiveCameraFrustum::new(
            &PerspectiveCamera {
                vertical_fov_degrees: 90.0,
                near: 0.01,
                //far: 10000.0,
            },
            self.image_width as i32,
            self.image_height as i32,
        );

        let procrustes_solver = FloatPrecisionProcrustesSolver::new();
        let mut canonical_landmarks = Matrix3xX::zeros(468);
        // convert CANONICAL_FACE_MODEL_POINTS into a Matrix3X
        for i in 0..468 {
            canonical_landmarks[(0, i)] = CANONICAL_FACE_MODEL_POINTS[(i, 0)];
            canonical_landmarks[(1, i)] = CANONICAL_FACE_MODEL_POINTS[(i, 1)];
            canonical_landmarks[(2, i)] = CANONICAL_FACE_MODEL_POINTS[(i, 2)];
        }

        let canonical_landmarks_weights = Self::get_weights_vector();
        let mut screen_landmarks_matrix = self.landmark_list_as_eigen_matrix(points);

        // Project X- and Y- screen landmark coordinates at the Z near plane.

        screen_landmarks_matrix = self.project_xy(&pcf, screen_landmarks_matrix);

        let depth_offset = screen_landmarks_matrix.row(2).mean();

        // 1st iteration: don't unproject XY because it's unsafe to do so due to
        //                the relative nature of the Z coordinate. Instead, run the
        //                first estimation on the projected XY and use that scale to
        //                unproject for the 2nd iteration.

        let mut intermediate_landmarks = screen_landmarks_matrix.clone();

        intermediate_landmarks = self.change_handedness(intermediate_landmarks);

        let first_iteration_scale = self
            .estimate_scale(
                &intermediate_landmarks,
                &canonical_landmarks,
                &canonical_landmarks_weights,
                &procrustes_solver,
            )
            .unwrap();

        intermediate_landmarks = screen_landmarks_matrix.clone();

        // 2nd iteration: unproject XY using the scale from the 1st iteration.
        intermediate_landmarks = self.move_and_rescale_z(
            &pcf,
            depth_offset,
            first_iteration_scale,
            intermediate_landmarks,
        );

        intermediate_landmarks = self.unproject_xy(&pcf, intermediate_landmarks);
        intermediate_landmarks = self.change_handedness(intermediate_landmarks);

        let second_iteration_scale = self
            .estimate_scale(
                &intermediate_landmarks,
                &canonical_landmarks,
                &canonical_landmarks_weights,
                &procrustes_solver,
            )
            .unwrap();

        let total_scale = first_iteration_scale * second_iteration_scale;

        screen_landmarks_matrix =
            self.move_and_rescale_z(&pcf, depth_offset, total_scale, screen_landmarks_matrix);
        screen_landmarks_matrix = self.unproject_xy(&pcf, screen_landmarks_matrix);
        screen_landmarks_matrix = self.change_handedness(screen_landmarks_matrix);

        // At this point, screen landmarks are converted into metric landmarks.
        let mut metric_landmarks_matrix = screen_landmarks_matrix;

        let pose_transform_mat = procrustes_solver
            .solve_weighted_orthogonal_problem(
                &metric_landmarks_matrix,
                &canonical_landmarks,
                &canonical_landmarks_weights,
            )
            .unwrap();

        // Multiply each of the metric landmarks by the inverse pose
        // transformation matrix to align the runtime metric face landmarks with
        // the canonical metric face landmarks.
        let pose_transform_mat_inverse = pose_transform_mat.try_inverse().unwrap();
        let metric_landmarks_matrix_homogeneous = metric_landmarks_matrix.insert_row(3, 0.0);
        metric_landmarks_matrix =
            (pose_transform_mat_inverse * metric_landmarks_matrix_homogeneous).remove_row(3);

        let mut points: Vec<nalgebra::OPoint<f32, nalgebra::Const<3>>> =
            Vec::with_capacity(metric_landmarks_matrix.ncols());

        for i in 0..metric_landmarks_matrix.ncols() {
            let x = metric_landmarks_matrix[(0, i)];
            let y = metric_landmarks_matrix[(1, i)];
            let z = metric_landmarks_matrix[(2, i)];
            points.push(nalgebra::OPoint::from(Vector3::new(x, y, z)));
        }

        Box::new(MediapipeMetricFaceLandmarks { points: points })
    }
}

impl MetricFaceLandmarks for MediapipeMetricFaceLandmarks {
    fn get_metric_landmarks(&self) -> Vec<nalgebra::Point3<f32>> {
        self.points.clone()
    }

    fn len(&self) -> usize {
        self.points.len()
    }
}

pub struct MediapipeFaceLandmarksModel<'a> {
    session: Arc<InMemorySession<'a>>,
}

impl MediapipeFaceLandmarksModel<'_> {
    pub fn new() -> MediapipeFaceLandmarksModel<'static> {
        let enviroment = Environment::builder()
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .build()
            .unwrap()
            .into_arc();

        let model = include_bytes!("assets/face_landmarks_detector.onnx");

        let session = SessionBuilder::new(&enviroment)
            .unwrap()
            .with_intra_threads(5)
            .unwrap()
            .with_model_from_memory(model)
            .unwrap();

        MediapipeFaceLandmarksModel {
            session: Arc::new(session),
        }
    }
}

impl FaceLandmarksModel for MediapipeFaceLandmarksModel<'_> {
    fn run(
        &self,
        input: &DynamicImage,
        face_bbox: Option<(u32, u32, u32, u32)>,
    ) -> (Box<dyn ScreenFaceLandmarks>, (u32, u32, u32, u32)) {
        // if face bounding box is set, crop the image accordingly
        let face_bbox = face_bbox.unwrap_or((0, 0, 720, 720));

        let original_face_bbox = face_bbox.clone();

        // add padding to face bounding box
        // add 30% padding to the face bounding box
        let padding = 0.25 * face_bbox.2 as f32;

        let face_bbox = (
            (face_bbox.0 as f32 - padding) as i32,
            (face_bbox.1 as f32 - padding) as i32,
            (face_bbox.2 as f32 + 2.0 * padding) as i32,
            (face_bbox.3 as f32 + 2.0 * padding) as i32,
        );

        // adjust face bounding box to fit the image
        let mut face_bbox = adjust_bbox(
            face_bbox.0,
            face_bbox.1,
            face_bbox.2,
            face_bbox.3,
            input.width(),
            input.height(),
        );

        let input = input.crop_imm(face_bbox.0, face_bbox.1, face_bbox.2, face_bbox.3);

        // resize to 256x256
        let input = input.resize_exact(256, 256, image::imageops::FilterType::Nearest);

        // convert to RGB
        let input = input.to_rgb8();

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

        let face_flag = outputs[1].try_extract::<f32>().unwrap();
        let face_flag: Vec<f32> = face_flag.view().deref().clone().iter().cloned().collect();
        let face_flag = face_flag[0];
        let face_flag = sigmoid(face_flag);

        let res: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
        let res = res.view().deref().clone();

        // convert to vector
        let res: Vec<f32> = res.iter().cloned().collect();

        // convert to FaceLandmarks (make sure to divide by 256)
        let face_landmarks = MediapipeFaceLandmarks::from_vec(
            res.iter().map(|p| p / 256.0).collect(),
            face_bbox,
            face_flag,
            input.width(),
            input.height(),
        );

        //println!("face confidence: {}", face_flag);
        // update face bounding box if a face was detected (more than 90% confidence)
        if face_landmarks.face_confidence > 0.5 {
            face_bbox = face_landmarks.get_face_bbox_from_landmarks();
        } else {
            // use original face bounding box
            face_bbox = original_face_bbox;
        }

        (Box::new(face_landmarks), face_bbox)
    }
}

pub fn ray_sphere_intersect(
    ray_origin: Point3,
    ray_dir: Vector3<f32>,
    sphere_origin: Point3,
    sphere_radius: f32,
) -> Point3 {
    let dx = ray_dir.x;
    let dy = ray_dir.y;
    let dz = ray_dir.z;
    let x0 = ray_origin.x;
    let y0 = ray_origin.y;
    let z0 = ray_origin.z;
    let cx = sphere_origin.x;
    let cy = sphere_origin.y;
    let cz = sphere_origin.z;
    let r = sphere_radius;

    let a = dx * dx + dy * dy + dz * dz;
    let b = 2.0 * dx * (x0 - cx) + 2.0 * dy * (y0 - cy) + 2.0 * dz * (z0 - cz);
    let c = cx * cx + cy * cy + cz * cz + x0 * x0 + y0 * y0 + z0 * z0
        - 2.0 * (cx * x0 + cy * y0 + cz * z0)
        - r * r;

    let disc = b * b - 4.0 * a * c;

    if disc < 0.0 {
        return Point3::new(0.0, 0.0, -1.0);
    }

    let t = (-b - disc.sqrt()) / (2.0 * a);

    ray_origin + ray_dir * t
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn adjust_bbox(x: i32, y: i32, w: i32, h: i32, image_w: u32, image_h: u32) -> (u32, u32, u32, u32) {
    let original_aspect_ratio = w as f32 / h as f32;

    // ajust the bounding box to fit the image
    let x = x.max(0);
    let y = y.max(0);

    let w = w.min(image_w as i32 - x);
    let h = h.min(image_h as i32 - y);

    // adjust aspect ratio by shortening the longer side
    let short_side = w.min(h);

    let w = (short_side as f32 * original_aspect_ratio) as i32;
    let h = (short_side as f32 / original_aspect_ratio) as i32;

    (x as u32, y as u32, w as u32, h as u32)
}

pub struct PerspectiveCamera {
    vertical_fov_degrees: f32,
    near: f32,
    //far: f32,
}

pub struct PerspectiveCameraFrustum {
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    //far: f32,
}

impl PerspectiveCameraFrustum {
    fn new(perspective_camera: &PerspectiveCamera, frame_width: i32, frame_height: i32) -> Self {
        const K_DEGREES_TO_RADIANS: f32 = std::f32::consts::PI / 180.0;

        let height_at_near = 2.0
            * perspective_camera.near
            * (0.5 * K_DEGREES_TO_RADIANS * perspective_camera.vertical_fov_degrees).tan();

        let width_at_near = frame_width as f32 * height_at_near / frame_height as f32;

        Self {
            left: -0.5 * width_at_near,
            right: 0.5 * width_at_near,
            bottom: -0.5 * height_at_near,
            top: 0.5 * height_at_near,
            near: perspective_camera.near,
            //far: perspective_camera.far,
        }
    }
}
