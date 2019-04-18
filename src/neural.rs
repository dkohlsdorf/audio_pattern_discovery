use crate::numerics::*;

/// Single layer Autoencoder 
pub struct AutoEncoder {
    pub w_encode: Mat,
    pub w_decode: Mat, 
    pub b_encode: Mat,
    pub b_decode: Mat
}

impl AutoEncoder {

    pub fn new(input_dim: usize, latent: usize) -> AutoEncoder {
        AutoEncoder {
            w_encode: Mat{ flat: vec![0.0; input_dim * latent], cols: latent },
            w_decode: Mat{ flat: vec![0.0; latent * input_dim], cols: input_dim },
            b_encode: Mat{ flat: vec![0.0; latent], cols: latent },
            b_decode: Mat{ flat: vec![0.0; input_dim], cols: input_dim }
        }
    }

    /// one sgd step given a learning rate
    pub fn take_step(&mut self, x: &Mat, alpha: f32) {
        let latent = x.mul(&self.w_encode)
            .add_col(&self.b_encode)
            .sigmoid();
        let reconstruction = latent.mul(&self.w_decode)
            .add_col(&self.b_decode);
        println!("{:?}", latent.flat);
        println!("{:?}", reconstruction.flat);
    }

}
