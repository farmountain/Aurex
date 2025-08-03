//! Simple tensor primitive stubs used for tests.

#[derive(Default, Debug)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Self { data: vec![0.0; size], shape: shape.to_vec() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_zeros() {
        let t = Tensor::zeros(&[2, 2]);
        assert_eq!(t.data, vec![0.0; 4]);
        assert_eq!(t.shape, vec![2, 2]);
    }
}
