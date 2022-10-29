use nalgebra::{Vector2, Vector4, Vector3};

/// Gets the 1D length of this vector (x - y)
pub fn vec2_one_d_lenght(vector: Vector2<f32>) -> f32 {
    (vector.x - vector.y).abs()
}

pub fn mul_vector4(vec1: Vector4<f32>, vec2: Vector4<f32>) -> Vector4<f32> {
    Vector4::new(vec1.x * vec2.x, vec1.y * vec2.y, vec1.z * vec2.z, vec1.w * vec2.w)
}

/// Checks if vec1 is inside vec2
pub fn vec2_one_d_in_range(vec1: Vector2<f32>, vec2: Vector2<f32>) -> bool {
    vec1.x >= vec2.x && vec1.y <= vec2.y
}

/// Checks if vec1 and vec2 are overlapping
pub fn vec2_one_d_overlapping(vec1: Vector2<u32>, vec2: Vector2<u32>) -> bool {
        (vec2.x < vec1.x && vec2.y > vec1.y)
        || (vec2.x > vec1.x && vec2.y > vec1.y && vec2.x < vec1.y)
        || (vec2.x < vec1.x && vec2.y < vec1.y && vec2.y > vec1.x)
        || vec2_one_d_in_range(Vector2::new(vec1.x as f32, vec1.y as f32), Vector2::new(vec2.x as f32, vec2.y as f32))
        || vec2_one_d_in_range(Vector2::new(vec2.x as f32, vec2.y as f32), Vector2::new(vec1.x as f32, vec1.y as f32))
}

pub fn distance_between_points(point_1: Vector3<f64>, point_2: Vector3<f64>) -> f32 {
    ((point_1.x-point_2.x).powi(2) + (point_1.y-point_2.y).powi(2) + (point_1.z-point_2.z).powi(2)).sqrt() as f32
}

pub fn view_cm_size(pixel_rad: f32, distance: f32) -> f32 {
    (pixel_rad/2.0).tan() * distance * 2.0
}