use nalgebra::Point3;
use nalgebra::Matrix3;

// struct that stores face landmarks (478 3D points)
pub struct FaceLandmarks {
    // the 3D points of the face mesh
    pub points: Vec<Point3<f32>>,
    // affine transformation matrix to transform the face mesh to planar 2D space
    pub affine_transform: Matrix3<f32>,
}

pub enum LandmarkPoint {
    LeftPupil,
    RightPupil,
}

pub enum LandmarkMesh {
    LeftEye,
    RightEye,
    LeftPupil,
    RightPupil,
}

// construct from 1D vector of 478 3D points (flattened 3D points)

impl FaceLandmarks {

    pub fn from_vec(points: Vec<f32>) -> FaceLandmarks {
        let mut landmarks = FaceLandmarks {
            points: Vec::new(),
            affine_transform: Matrix3::identity(), // set to identity matrix for now
        };
        for i in 0..points.len() / 3 {
            landmarks.points.push(Point3::new(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]));
        }
        landmarks
    }

    pub fn get_point(&self, point: LandmarkPoint) -> Point3<f32> {
        match point {
            LandmarkPoint::LeftPupil => self.points[468],
            LandmarkPoint::RightPupil => self.points[473],
        }
    }

    pub fn get_mesh(&self, mesh: LandmarkMesh) -> Vec<Point3<f32>> {
        match mesh {
            // [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
            LandmarkMesh::LeftEye => [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33].iter().map(|&i| self.points[i]).collect(),
            // [362, 382, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 362]
            LandmarkMesh::RightEye => [362, 382, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 362].iter().map(|&i| self.points[i]).collect(),
            // [478 - 9, 478 - 8, 478 - 7, 478 - 6]
            LandmarkMesh::LeftPupil => [478 - 9, 478 - 8, 478 - 7, 478 - 6].iter().map(|&i| self.points[i]).collect(),
            //[478 - 4, 478 - 3, 478 - 2, 478 - 1]
            LandmarkMesh::RightPupil => [478 - 4, 478 - 3, 478 - 2, 478 - 1].iter().map(|&i| self.points[i]).collect(),
        }

    }


} 





// // Calculate the relative position of the pupil in reference to the face.
// // To do this, we first calculate the rectangle of the face in 3D space, then
// // we project the pupil onto the face rectangle, and then we calculate the relative
// // position of the pupil in reference to the face rectangle.
// fn get_relative_pupil_position()
// {
