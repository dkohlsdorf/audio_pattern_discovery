extern crate toml;
use std::fs::File;
use std::io::prelude::*;

use crate::alignments::AlignmentParams;

#[derive(Deserialize, Debug, Clone)]
pub struct Discovery {
    pub dft_win: usize,
    pub dft_step: usize,
    pub ceps_filter: usize,
    pub vat_moving: usize,
    pub vat_percentile: f32,
    pub vat_min_len: usize,
    pub alignment_workers: usize,
    pub clustering_percentile: f32,
    pub merging_percentile: f32,
    pub merging_internal_percentile: f32,
    pub merging_moving: usize,
    pub warping_band_percentage: f32,
    pub insertion_penalty: f32,
    pub deletion_penalty: f32,
    pub match_penalty: f32,
}

impl Discovery {

    pub fn from_toml(file: String) -> Discovery {
        let mut template_conf = String::new();
        let _ = File::open(file)
            .expect("Template file not found")
            .read_to_string(&mut template_conf);
        let conf: Discovery = toml::from_str(&template_conf).unwrap();
        conf
    }

    pub fn alignment_params(&self, n_size: usize) -> AlignmentParams {
        AlignmentParams {
            warping_band: (self.warping_band_percentage * n_size as f32) as usize,
            insertion_penalty: self.insertion_penalty,
            match_penalty: self.match_penalty,
            deletion_penalty: self.deletion_penalty
        }
    }

}