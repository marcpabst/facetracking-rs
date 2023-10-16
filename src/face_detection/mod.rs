pub mod model_blazeface;

use core::fmt::Debug;

use image::DynamicImage;

pub trait FaceDetectionModel {
    fn run(&self, image: &DynamicImage) -> Box<dyn FaceBoundingBox>;
}

/// Trait representing a bounding box around a detected face in an image.
pub trait FaceBoundingBox {
    /// Returns the size of the original image as a tuple of width and height (in pixels).
    fn image_size(&self) -> (u32, u32);

    /// Returns the coordinates of the origin (top-left corner) of the face rectangle as a tuple of x and y (in pixels).
    fn origin(&self) -> (u32, u32);

    /// Returns the width of the face rectangle (in pixels).
    fn width(&self) -> u32;

    /// Returns the height of the face rectangle (in pixels).
    fn height(&self) -> u32;

    /// Returns the confidence score of the detected face.
    fn score(&self) -> f32;

    /// Converts the face rectangle information to a tuple of (x, y, width, height) representing the bounding box.
    /// The tuple elements correspond to the x and y coordinates of the top-left corner, and the width and height of the rectangle, respectively.
    fn to_tuple(&self) -> (u32, u32, u32, u32);
}

impl Debug for dyn FaceBoundingBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FaceBoundingBox")
            .field("image_size", &self.image_size())
            .field("origin", &self.origin())
            .field("width", &self.width())
            .field("height", &self.height())
            .field("score", &self.score())
            .finish()
    }
}
