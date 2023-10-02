// translated from mediapipe/modules/face_geometry/libs/procrustes_solver.cc

use nalgebra::allocator::Allocator;
use nalgebra::constraint::{SameNumberOfRows, ShapeConstraint};
use nalgebra::linalg::SVD;
use nalgebra::storage::Storage;
use nalgebra::{Scalar, ArrayStorage, Dim, Vector, DefaultAllocator};
use nalgebra::{
    Const, DMatrix, DVector, Dyn, Matrix, Matrix3, Matrix3xX, Matrix4, RowDVector, RowVector3,
    VecStorage, Vector3, U1, U3,
};
use ndarray::Axis;
use std::f32::EPSILON;

// define Matrix3X as a type alias for
type Matrix3X<T> = Matrix<T, U3, Dyn, VecStorage<T, U3, Dyn>>;

pub trait ProcrustesSolver {
    fn solve_weighted_orthogonal_problem(
        &self,
        source_points: &Matrix3xX<f32>,
        target_points: &Matrix3xX<f32>,
        point_weights: &DVector<f32>,
    ) -> Result<Matrix4<f32>, &'static str>;
}

pub struct FloatPrecisionProcrustesSolver;

impl FloatPrecisionProcrustesSolver {
    pub fn new() -> Self {
        FloatPrecisionProcrustesSolver
    }

    fn validate_input_points(
        source_points: &Matrix3xX<f32>,
        target_points: &Matrix3xX<f32>,
    ) -> Result<(), &'static str> {
        if source_points.ncols() == 0 {
            return Err("The number of source points must be positive!");
        }

        if source_points.ncols() != target_points.ncols() {
            return Err("The number of source and target points must be equal!");
        }

        Ok(())
    }

    fn validate_point_weights(
        num_points: usize,
        point_weights: &DVector<f32>,
    ) -> Result<(), &'static str> {
        if point_weights.len() == 0 {
            return Err("The number of point weights must be positive!");
        }

        if point_weights.len() != num_points {
            return Err("The number of points and point weights must be equal!");
        }

        let total_weight: f32 = point_weights.iter().sum();

        if total_weight <= EPSILON {
            return Err("The total point weight is too small!");
        }

        Ok(())
    }

    fn extract_square_root(point_weights: &DVector<f32>) -> DVector<f32> 
    {
        let out = point_weights.map(|x| x.sqrt());
        // convert to DVector
        let out = DVector::from_row_slice(out.as_slice());

        out
    }

    /// Combines a rotation matrix and a translation vector into a 4x4 transformation matrix.
    ///
    /// The resulting transformation matrix is a 4x4 matrix that represents a rigid-body transformation
    /// consisting of a rotation followed by a translation. The rotation is specified by a 3x3 matrix
    /// `rotation`, and the translation is specified by a 3x1 vector `translation`. The resulting
    /// transformation matrix is a 4x4 matrix of the form:
    ///
    /// ```text
    /// [ R11 R12 R13 Tx ]
    /// [ R21 R22 R23 Ty ]
    /// [ R31 R32 R33 Tz ]
    /// [  0   0   0  1 ]
    /// ```
    ///
    /// where `Rij` are the elements of the rotation matrix, `Tj` are the elements of the translation
    /// vector, and `0` and `1` are zero and one, respectively.
    ///
    /// # Arguments
    ///
    /// * `rotation` - A 3x3 matrix representing the rotation component of the transformation.
    /// * `translation` - A 3x1 vector representing the translation component of the transformation.
    ///
    /// # Returns
    ///
    /// A 4x4 matrix representing the combined rotation and translation transformation.
    fn combine_transform_matrix(
        rotation: &Matrix3<f32>,
        translation: &Vector3<f32>,
    ) -> Matrix4<f32> {
        let mut result = Matrix4::identity();

        result.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
        result.fixed_view_mut::<3, 1>(0, 3).copy_from(&translation);

        result
    }

    fn rowwise_elementwise_multiply(matrix: &Matrix3X<f32>, vector: &DVector<f32>) -> Matrix3X<f32> {
        // bounds checking
        assert_eq!(matrix.ncols(), vector.len());
        
        let mut result = matrix.clone();
        for i in 0..3 {
            result.row_mut(i).component_mul_assign(&vector.transpose());
        }
    
        result
    }

    fn internal_solve_weighted_orthogonal_problem(
        sources: &Matrix3X<f32>,
        targets: &Matrix3X<f32>,
        sqrt_weights: &DVector<f32>,
    ) -> Result<Matrix4<f32>, &'static str> {
        // perform row-wise multiplication of sources and sqrt_weights

       
        let weighted_sources = Self::rowwise_elementwise_multiply(sources, sqrt_weights);
        let weighted_targets = Self::rowwise_elementwise_multiply(targets, sqrt_weights);

      
        let total_weight = sqrt_weights.component_mul(&sqrt_weights).sum();

       
        let twice_weighted_sources = Self::rowwise_elementwise_multiply(&weighted_sources, sqrt_weights);

        let source_center_of_mass = twice_weighted_sources.column_sum() / total_weight;

        let centered_weighted_sources =
            weighted_sources.clone() - source_center_of_mass * sqrt_weights.transpose();

        let rotation = Self::compute_optimal_rotation(
            &(weighted_targets.clone() * centered_weighted_sources.transpose()),
        )
        .ok_or("Rotation is None!")?;

        let scale = Self::compute_optimal_scale(
            &centered_weighted_sources,
            &weighted_sources,
            &weighted_targets,
            &rotation,
        )?;

        let rotation_and_scale = scale * rotation;

        let pointwise_diffs = weighted_targets - rotation_and_scale * weighted_sources;
        let weighted_pointwise_diffs = Self::rowwise_elementwise_multiply(&pointwise_diffs, sqrt_weights);

        let translation: Vector3<f32> = weighted_pointwise_diffs.column_sum() / total_weight;

        let transform_mat = Self::combine_transform_matrix(&rotation, &translation);

        Ok(transform_mat)
    }

    fn compute_optimal_rotation(design_matrix: &Matrix3<f32>) -> Option<Matrix3<f32>> {
        let k_absolute_error_eps = 1e-6; // Define your value for kAbsoluteErrorEps here

        if design_matrix.norm() <= k_absolute_error_eps {
            return None;
        }

        let svd = design_matrix.svd(true, true);

        let mut post_rotation = svd.u.unwrap();
        let pre_rotation = svd.v_t.unwrap();

        // Disallow reflection by ensuring that det(rotation) = +1 (and not -1)
        if post_rotation.determinant() * pre_rotation.determinant() < 0.0 {
            let mut column = post_rotation.column_mut(2);
            column *= -1.0;
        }

        // Transposed from the paper
        let rotation = post_rotation * pre_rotation;

        Some(rotation)
    }

    fn compute_optimal_scale(
        centered_weighted_sources: &Matrix3X<f32>,
        weighted_sources: &Matrix3X<f32>,
        weighted_targets: &Matrix3X<f32>,
        rotation: &Matrix3<f32>,
    ) -> Result<f32, &'static str> {
        let rotated_centered_weighted_sources = rotation * centered_weighted_sources;
        let numerator = rotated_centered_weighted_sources.dot(weighted_targets);
        let denominator = centered_weighted_sources.dot(weighted_sources);

        if denominator <= EPSILON || numerator / denominator <= EPSILON {
            return Err("Scale is too small!");
        }

        Ok(numerator / denominator)
    }
}


impl ProcrustesSolver for FloatPrecisionProcrustesSolver {
    fn solve_weighted_orthogonal_problem(
        &self,
        source_points: &Matrix3xX<f32>,
        target_points: &Matrix3xX<f32>,
        point_weights: &DVector<f32>,
    )
    
    -> Result<Matrix4<f32>, &'static str> {
        // Validate inputs.
        //Self::validate_input_points(source_points, target_points)?;

        //Self::validate_point_weights(source_points.ncols(), point_weights)?;

        // 

        // Extract square root from the point weights.
        let sqrt_weights = Self::extract_square_root(point_weights);

        // Try to solve the WEOP problem.
        let out = Self::internal_solve_weighted_orthogonal_problem(
            source_points,
            target_points,
            &sqrt_weights,
        )?;

        Ok(out)
    }
}