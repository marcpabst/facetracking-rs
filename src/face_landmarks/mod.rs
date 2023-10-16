pub mod model_mediapipe;

use image::DynamicImage;
use nalgebra::Point3;

// define the FaceLandmarks trait
pub trait ScreenFaceLandmarks: Send + Sync {
    fn get_landmark(&self, index: usize) -> Option<Point3<f32>>;
    fn get_landmarks(&self) -> Vec<Point3<f32>>;
    fn get_face_bbox(&self) -> (u32, u32, u32, u32);
    fn get_confidence(&self) -> f32;
    fn to_metric(&self) -> Box<dyn MetricFaceLandmarks>;
    fn len(&self) -> usize;
}

// define the MetricFaceLandmarks trait
pub trait MetricFaceLandmarks {
    fn get_metric_landmarks(&self) -> Vec<Point3<f32>>;
    fn len(&self) -> usize;
}

trait FaceLandmarks: ScreenFaceLandmarks + MetricFaceLandmarks {}

// define FaceLandmarksModel trait
pub trait FaceLandmarksModel {
    fn run(
        &self,
        image: &DynamicImage,
        face_bbox: Option<(u32, u32, u32, u32)>,
    ) -> (Box<dyn ScreenFaceLandmarks>, (u32, u32, u32, u32));
}
