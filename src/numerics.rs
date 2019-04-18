use statrs::distribution::{Continuous, Normal};

/**
 * Comput sample mean of a slice
 */
pub fn mean(x: &[f32]) -> f32 {
    let mut mean = 0.0;
    for value in x.iter() {
        mean += value;
    }
    mean / (x.len() as f32)
}

/**
 * Compute sample variance of a slice given it's mean
 */
pub fn std(x: &[f32], mu: f32) -> f32 {
    let mut std = 0.0;
    for value in x.iter() {
        std += f32::powf(value - mu, 2.0);
    }
    f32::sqrt(std / (x.len() as f32))
}

/**
 * Min in slice, ignoring infinite values
 */
pub fn min(x: &[f32]) -> f32 {
    let mut min = std::f32::INFINITY;
    for frame in x {
        if *frame < min && frame.is_finite() {
            min = *frame;
        }
    }
    min
}

/**
 * Max in slice ignoring infinite values
 */
pub fn max(x: &[f32]) -> f32 {
    let mut max = std::f32::NEG_INFINITY;
    for frame in x {
        if *frame > max && frame.is_finite() {
            max = *frame;
        }
    }
    max
}

/**
 * Hamming Window
 */
pub fn hamming(len: usize) -> Vec<f32> {
    let mut hamming = Vec::new();
    for i in 0..len {
        hamming.push(0.54 + 0.46 * f32::cos((2.0 * std::f32::consts::PI * i as f32) / len as f32));
    }
    hamming
}

/**
 * Z-scoring
 */
pub fn z_score(x: f32, mu: f32, sigma: f32) -> f32 {
    (x - mu) / sigma
}

/**
 * Triangle filter
 */
pub fn triag(len: usize) -> Vec<f32> {
    let mut triag = vec![0.0; len];
    let center = (len - 1) / 2;
    for i in 0..=center {
        triag[i] = i as f32 / len as f32;
        triag[len - 1 - i] = i as f32 / len as f32;
    }
    triag
}

/**
 * Dot product between two vectors
 */
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    let mut dot = 0.0;
    for i in 0..x.len() {
        dot += x[i] * y[i];
    }
    dot
}

/**
 * Convolve slice with smaller slice
 */
pub fn convolve(frames: &[f32], filter: &[f32], step: usize) -> Vec<f32> {
    let n = filter.len();
    let mut convolved = Vec::new();
    for i in (n..frames.len()).step_by(step) {
        convolved.push(dot(filter, &frames[i - n..i]));
    }
    convolved
}

/**
 * Euclidean Distance
 */
pub fn euclidean(x: &[f32], y: &[f32]) -> f32 {
    let mut distance = 0.0;
    for i in 0..x.len() {
        distance += f32::powf(x[i] - y[i], 2.0);
    }
    f32::sqrt(distance)
}

/**
 * Extract percentile for example the median is at percentile(x, 0.5)
 */
pub fn percentile(x: &mut [f32], perc: f32) -> f32 {
    let n = x.len() as f32 * perc;
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    x[n as usize]
}

/**
 * absolute distance for unsigned
 */
pub fn abs(n: usize, m: usize) -> usize {
    if n > m {
        n - m
    } else {
        m - n
    }
}

/**
 *  distance for unsigned bounded by 0
 */
pub fn diff(n: usize, m: usize) -> usize {
    if n > m {
        n - m
    } else {
        0
    }
}

/**
 * Log likelihood of gaussian
 */
pub fn ll(x: &[f32], mu: &[f32], std: &[f32]) -> f32 {
    let mut ll = 0.0;
    for i in 0..x.len() {
        let normal = Normal::new(f64::from(mu[i]), f64::from(std[i])).unwrap();
        ll += normal.ln_pdf(f64::from(x[i]));
    }
    ll as f32
}


/// A flat matrix
pub struct Mat {
    pub flat: Vec<f32>,
    pub cols: usize,
}

impl Mat {

    pub fn rows(&self) -> usize {
        self.flat.len() / self.cols
    }

    /// transposed matrix as new matrix
    pub fn transpose(&self) -> Mat {
        let rows = self.rows();
        let cols = self.cols;
        let mut transposed = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = self.flat[i * cols + j];
            }
        }
        Mat{flat: transposed, cols: rows}
    }

    /// sigmoid of matrix as new matrix
    pub fn sigmoid(&self) -> Mat {
        Mat {
            cols: self.cols,
            flat: self.flat.iter().map(|x| {
                1.0 / (1.0 + f32::exp(-x))
            }).collect()        
        }
    }

    /// add a column vector to all columns
    pub fn add_col(&self, x: &Mat) -> Mat {
        let mut flat = self.flat.clone();
        for i in 0 .. self.rows() {
            for j in 0 .. self.cols {
                flat[i * self.cols + j] += x.flat[j];
            }
        }
        Mat {flat, cols: self.cols}
    }

    /// matrix multiplication
    pub fn mul(&self, other: &Mat) -> Mat {
        assert!(self.cols == other.rows());
        let n = self.rows();
        let d = self.cols;
        let cols = other.cols;
        let mut flat = vec![0.0; n * cols];
        for i in 0 .. n {
            for j in 0 .. cols {
                for k in 0 .. d {
                    flat[i * cols + j] += self.flat[i * d + k] * other.flat[k * cols + j];
                }
            }
        }
        Mat {flat, cols}
    }
}
