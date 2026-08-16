use crate::Tensor;

/// Pixel-wise mean absolute difference in [0,1] between two tensors.
pub fn mean_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let n = a.data.len().min(b.data.len());
    if n == 0 {
        return 0.0;
    }
    let sum: f32 = a
        .data
        .iter()
        .zip(&b.data)
        .take(n)
        .map(|(x, y)| (x - y).abs())
        .sum();
    sum / n as f32
}

/// True when the mean absolute difference between two frames exceeds
/// `threshold`, i.e. a scene cut where cross-frame interpolation would ghost.
pub fn is_scene_cut(a: &Tensor, b: &Tensor, threshold: f32) -> bool {
    mean_abs_diff(a, b) > threshold
}

/// Linear blend at time `t` in [0,1] between two same-shaped tensors.
/// Reference interpolation until a RIFE engine is available.
pub fn blend(a: &Tensor, b: &Tensor, t: f32) -> Tensor {
    let data = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| x + (y - x) * t)
        .collect();
    Tensor::new(a.shape.clone(), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: Vec<f32>) -> Tensor {
        Tensor::new(vec![1, 1, 2, 2], data)
    }

    #[test]
    fn mean_abs_diff_zero_for_equal() {
        let a = tensor(vec![0.5; 4]);
        let b = tensor(vec![0.5; 4]);
        assert_eq!(mean_abs_diff(&a, &b), 0.0);
    }

    #[test]
    fn scene_cut_detects_large_change() {
        let a = tensor(vec![0.0; 4]);
        let b = tensor(vec![1.0; 4]);
        assert!(is_scene_cut(&a, &b, 0.25));
        assert!(!is_scene_cut(&a, &b, 1.5));
    }

    #[test]
    fn blend_midpoint_averages() {
        let a = tensor(vec![0.0; 4]);
        let b = tensor(vec![1.0; 4]);
        let mid = blend(&a, &b, 0.5);
        assert!(mid.data.iter().all(|v| (*v - 0.5).abs() < 1e-6));
    }

    #[test]
    fn blend_endpoints_are_identity() {
        let a = tensor(vec![0.2; 4]);
        let b = tensor(vec![0.8; 4]);
        assert_eq!(blend(&a, &b, 0.0).data, a.data);
        assert_eq!(blend(&a, &b, 1.0).data, b.data);
    }
}
