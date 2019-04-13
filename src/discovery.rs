extern crate toml;
use std::fs::File;
use std::io::prelude::*;

#[derive(Deserialize, Debug)]
pub struct Discovery {
    pub dft_win: usize,
    pub dft_step: usize,
    pub ceps_filter: usize,
    pub vat_moving: usize,
    pub vat_percentile: f32,
    pub vat_min_len: usize,
    pub alignment_workers: usize,
    pub clustering_percentile: f32
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

}