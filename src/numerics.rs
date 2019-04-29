use statrs::distribution::{Continuous, Normal};

extern crate bincode;
extern crate rand;
extern crate serde_derive;

use rand::Rng;

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
    let mut numbers: Vec<f32> = x
        .iter()
        .filter_map(|x| if x.is_nan() { None } else { Some(*x) })
        .collect();
    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    numbers[n as usize]
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
#[derive(Serialize, Deserialize, Clone)]
pub struct Mat {
    pub flat: Vec<f32>,
    pub cols: usize,
}

/// All functions should behave in an immutable manner
impl Mat {
    pub fn seeded(rows: usize, cols: usize) -> Mat {
        let mut rng = rand::thread_rng();
        let mut flat = vec![];
        for _i in 0..rows * cols {
            let uniform = (rng.gen_range(0.0, 1.0) - 0.5) / cols as f32;
            flat.push(uniform);
        }
        Mat { flat, cols }
    }

    pub fn rows(&self) -> usize {
        self.flat.len() / self.cols
    }

    pub fn norm(&self) -> Mat {
        let min_val = min(&self.flat);
        let max_val = max(&self.flat);
        let scaler = max_val - min_val;
        if (min_val - max_val).abs() < 1e-8 {
            let flat = self.flat.iter().map(|x| (x - min_val) / scaler).collect();
            Mat {
                flat,
                cols: self.cols,
            }
        } else {
            Mat {
                flat: self.flat.clone(),
                cols: self.cols,
            }
        }
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
        Mat {
            flat: transposed,
            cols: rows,
        }
    }

    /// relu of matrix as new matrix
    pub fn sigmoid(&self) -> Mat {
        Mat {
            cols: self.cols,
            flat: self
                .flat
                .iter()
                .map(|x| 1.0 / (1.0 + f32::exp(-x)))
                .collect(),
        }
    }

    /// Compute derivative as new matrix
    pub fn delta_sigmoid(&self) -> Mat {
        Mat {
            cols: self.cols,
            flat: self.flat.iter().map(|x| x * (1.0 - x)).collect(),
        }
    }

    /// add a column vector to all columns
    pub fn add_col(&self, x: &Mat) -> Mat {
        let mut flat = self.flat.clone();
        for i in 0..self.rows() {
            for j in 0..self.cols {
                flat[i * self.cols + j] += x.flat[j];
            }
        }
        Mat {
            flat,
            cols: self.cols,
        }
    }

    /// multiplication, element by element
    pub fn mul_ebe(&self, other: &Mat) -> Mat {
        let mut flat = self.flat.clone();
        for i in 0..flat.len() {
            flat[i] *= other.flat[i];
        }
        Mat {
            cols: self.cols,
            flat,
        }
    }

    /// addition, element by element
    pub fn add_ebe(&self, other: &Mat) -> Mat {
        let mut flat = self.flat.clone();
        for i in 0..flat.len() {
            flat[i] += other.flat[i];
        }
        Mat {
            cols: self.cols,
            flat,
        }
    }

    /// addition, element by element
    pub fn sub_ebe(&self, other: &Mat) -> Mat {
        let mut flat = self.flat.clone();
        for i in 0..flat.len() {
            flat[i] -= other.flat[i];
        }
        Mat {
            cols: self.cols,
            flat,
        }
    }

    /// scale a matrix
    pub fn scale(&self, scaler: f32) -> Mat {
        Mat {
            cols: self.cols,
            flat: self.flat.iter().map(|a| a * scaler).collect(),
        }
    }

    /// matrix multiplication
    pub fn mul(&self, other: &Mat) -> Mat {
        assert!(self.cols == other.rows());
        let n = self.rows();
        let d = self.cols;
        let cols = other.cols;
        let mut flat = vec![0.0; n * cols];
        for i in 0..n {
            for j in 0..cols {
                for k in 0..d {
                    flat[i * cols + j] += self.flat[i * d + k] * other.flat[k * cols + j];
                }
            }
        }
        Mat { flat, cols }
    }
}
