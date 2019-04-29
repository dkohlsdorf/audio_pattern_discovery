extern crate bincode;
extern crate serde_derive;

use crate::error::*;
use crate::numerics::*;
use bincode::{deserialize, serialize};
use std::fs::File;
use std::io::prelude::*;

/// Single layer Autoencoder
#[derive(Serialize, Deserialize, Clone)]
pub struct AutoEncoder {
    pub w_encode: Mat,
    pub w_decode: Mat,
    pub b_encode: Mat,
    pub b_decode: Mat,
}

impl AutoEncoder {
    pub fn n_latent(&self) -> usize {
        self.b_encode.cols
    }

    pub fn from_file(file: &str) -> Result<AutoEncoder> {
        let mut fp = File::open(file)?;
        let mut buf: Vec<u8> = vec![];
        let _ = fp.read_to_end(&mut buf)?;
        let decoded: AutoEncoder = deserialize(&buf).unwrap();
        Ok(decoded)
    }

    /// save file
    pub fn save_file(&self, file: &str) -> Result<()> {
        let mut fp = File::create(file)?;
        let encoded: Vec<u8> = serialize(&self).unwrap();
        fp.write_all(&encoded)?;
        Ok(())
    }

    pub fn new(input_dim: usize, latent: usize) -> AutoEncoder {
        AutoEncoder {
            w_encode: Mat::seeded(input_dim, latent),
            w_decode: Mat::seeded(latent, input_dim),
            b_encode: Mat::seeded(1, latent),
            b_decode: Mat::seeded(1, input_dim),
        }
    }

    pub fn predict(&self, x: &Mat) -> Mat {
        let prediction = x
            .mul(&self.w_encode)
            .add_col(&self.b_encode)
            .sigmoid()
            .scale(255.0);
        let mu = mean(&prediction.flat);
        let sigma = f32::max(std(&prediction.flat, mu), 1.0);
        Mat {
            flat: prediction
                .flat
                .iter()
                .map(|x| z_score(*x, mu, sigma))
                .collect(),
            cols: prediction.cols,
        }
    }

    /// one sgd step given a learning rate
    pub fn take_step(&mut self, x: &Mat, alpha: f32) -> f32 {
        let y = x.norm();
        let latent = x.mul(&self.w_encode).add_col(&self.b_encode);
        let latent_activation = latent.sigmoid();
        let output = latent.mul(&self.w_decode).add_col(&self.b_decode);
        let activation = output.sigmoid();
        let delta_out = y
            .sub_ebe(&activation)
            .scale(-1.0)
            .mul_ebe(&activation.delta_sigmoid());
        let delta_decode = delta_out
            .mul(&self.w_decode.transpose())
            .mul_ebe(&latent_activation.delta_sigmoid());
        let grad_decode = latent.transpose().mul(&delta_out).scale(alpha);
        let grad_encode = x.transpose().mul(&delta_decode).scale(alpha);
        self.w_encode = self.w_encode.sub_ebe(&grad_encode);
        self.w_decode = self.w_decode.sub_ebe(&grad_decode);
        self.b_encode = self.b_encode.sub_ebe(&delta_decode.scale(alpha));
        self.b_decode = self.b_decode.sub_ebe(&delta_out.scale(alpha));
        0.5 * euclidean(&activation.flat, &y.flat)
    }
}
