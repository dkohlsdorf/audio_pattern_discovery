extern crate toml;
extern crate glob;

use crate::audio::*;
use crate::clustering::*;
use crate::neural::*;
use crate::spectrogram::*;
use crate::error::*;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::prelude::*;

#[derive(Deserialize, Debug)]
pub struct Templates {
    pub img_w: String,
    pub img_h: String,
    pub out_docs: String,
    pub out_images: String,
    pub out_audio: String,
    pub document: String,
    pub dendogram: String,
    pub figure: String,
    pub result_html: String,
    pub out_encoder: String
}

impl Templates {

    /// save an auto encoder
    pub fn save_encoder(&self, encoder: AutoEncoder) -> Result<()> {
        encoder.save_file(&format!("{}/auto_encoder.bin", self.out_encoder))?;       
        Ok(())
    }

    /// load encoder
    pub fn read_encoder(&self) -> Result<AutoEncoder> {        
        AutoEncoder::from_file(&format!("{}/auto_encoder.bin", self.out_encoder))
    }        
    
    /// save all slices to disc
    pub fn dump_slices(
        &self,
        filename: String,
        clustering: &[Vec<usize>],
        slices: &[Slice],
        audio_filename: &[String],
        frame_rates: &[u32],
        sample_step: usize,
    ) -> Result<()> {
        let mut fp = File::create(filename)?;
        fp.write_fmt(format_args!("audio_file\tstart\tstop\tcluster\n"))?;
        for (i, cluster) in clustering.iter().enumerate() {
            for slice_id in cluster {
                let slice = &slices[*slice_id];
                let audio_id = audio_filename[slice.sequence.audio_id].clone();
                let rate  = frame_rates[slice.sequence.audio_id] as f32;
                let start = slice.start * sample_step;
                let stop  = slice.stop  * sample_step;
                fp.write_fmt(format_args!("{}\t{}\t{}\t{}\n", audio_id, start as f32 / rate, stop as f32 / rate, i))?;
            }
        }
        Ok(())
    }

    /// load from config
    pub fn from_toml(file: String) -> Templates {
        let mut template_conf = String::new();
        let _ = File::open(file)
            .expect("Template file not found")
            .read_to_string(&mut template_conf);
        let templates: Templates = toml::from_str(&template_conf).unwrap();
        templates
    }

    pub fn write_html(
        &self,
        out: String,
        cluster_files: &[String],
        sub_sequence: &[String]
    ) -> Result<()> {
        let mut clusters = String::new();
        let mut sequences = String::new();
        clusters.push_str("<ul>");
        for cluster in cluster_files.iter() {
            let p = format!("{}/{}", "audio", cluster);
            clusters.push_str(&format!(
                "<li><a href=\"{}\" download={}>{}</a></li>\n",
                &p, &p, &p
            ));
        }
        clusters.push_str("</ul>");
        sequences.push_str("<ul>");
        for cluster in sub_sequence.iter() {
            let p = format!("{}/{}", "audio", cluster);
            sequences.push_str(&format!(
                "<li><a href=\"{}\" download={}>{}</a></li>\n",
                &p, &p, &p
            ));
        }
        sequences.push_str("</ul>");    
        let mut file = File::open(&self.result_html)?;
        let mut template = String::new();
        file.read_to_string(&mut template)?;
        let filled = template
            .replace("[CLUSTERS_WAV]", &clusters)
            .replace("[SUB_WAV]", &sequences);
        let mut output = File::create(out)?;
        output.write_fmt(format_args!("{}", filled))?;
        
        Ok(())
    }

    /// Generate document from parts
    pub fn generate_doc(&self, filename: String, parts: Vec<String>) -> Result<()> {
        let mut file = File::open(&self.document)?;
        let mut template = String::new();
        file.read_to_string(&mut template)?;
        let mut body = String::new();
        for part in parts {
            body.push_str(&part);
        }
        let result = template.replace("<body>", &body);
        let mut file_out = File::create(format!("{}/{}", self.out_docs, filename))?;
        file_out.write_fmt(format_args!("{}", result))?;
        Ok(())
    }

    /// Dendogram generation from clustering results
    pub fn dendograms(
        &self,
        operations: &[ClusteringOperation],
        clusters: &HashSet<usize>,
        images: Vec<String>,
        prefix: &str
    ) -> Result<Vec<String>> {
        let mut results: HashMap<usize, String> = HashMap::new();
        for op in operations {
            let i = op.merge_i;
            let j = op.merge_j;
            let k = op.into;
            match op.operation {
                Merge::Sequence2Sequence => {
                    let graphics_i = self.image_ref(&images[i], true);
                    let graphics_j = self.image_ref(&images[j], true);
                    results.insert(k, format!("[.{} [{} {} ] ]", k, graphics_i, graphics_j));
                }
                Merge::Sequence2Cluster => {
                    let graphics_i = self.image_ref(&images[i], true);
                    let subtree_j = &results[&j];
                    results.insert(k, format!("[.{} [{} {} ] ]", k, graphics_i, subtree_j));
                }
                Merge::Cluster2Sequence => {
                    let subtree_i = &results[&i];
                    let graphics_j = self.image_ref(&images[j], true);
                    results.insert(k, format!("[.{} [{} {} ] ]", k, subtree_i, graphics_j));
                }
                Merge::Cluster2Cluster => {
                    let subtree_j = &results[&j];
                    let subtree_i = &results[&i];
                    results.insert(k, format!("[.{} [{} {} ] ]", k, subtree_i, subtree_j));
                }
            }
        }

        let mut latex_parts = vec![];
        for (i, cluster) in clusters.iter().enumerate() {
            let caption = format!("Dendogram {}", i);
            match results.get(cluster) {
                Some(result) => {
                    let graph = self.tikz(result)?;                    
                    let mut fp_graph = File::create(format!("{}/cluster_{}_{}.tikz", self.out_images, prefix, i))?;
                    fp_graph.write_fmt(format_args!("{}", graph))?;
                    let figure = self.figure(&self.image_ref(&format!("cluster_{}_{}.tikz", prefix, i), false), &caption)?;
                    latex_parts.push(figure);
                }
                _ => println!("Cluster not found: {} | Singular cluster", cluster),
            }
        }
        Ok(latex_parts)
    }

    /// Plot a gray scale image
    pub fn plot(&self, file: String, pixels: &[u8], rows: u32, cols: u32) -> Result<()> {
        let output = File::create(format!("{}/{}", self.out_images, file))?;
        let encoder = image::png::PNGEncoder::new(output);
        encoder.encode(&pixels, cols, rows, image::ColorType::Gray(8))?;
        Ok(())
    }

    /// Latex image reference
    fn image_ref(&self, name: &str, set_dim: bool) -> String {
        let path = format!("../{}/{}", self.out_images, name);
        if set_dim {
            format!(
                "{{\\includegraphics[width={}, height={}]{{{}}}}}\n",
                self.img_w, self.img_h, path
            )
        } else {
            format!(
                "\\includegraphics[width=\\textwidth]{{{}}}\n",
                path
            )
        }
    }

    /// Set a tikz image
    fn tikz(&self, tree: &str) -> Result<String> {
        let mut file = File::open(&self.dendogram)?;
        let mut template = String::new();
        file.read_to_string(&mut template)?;
        let tree_latex = template.replace("<tree>", tree);
        Ok(tree_latex)
    }

    /// Set a figure
    fn figure(&self, img_ref: &str, caption: &str) -> Result<String> {
        let mut file = File::open(&self.figure)?;
        let mut template = String::new();
        file.read_to_string(&mut template)?;
        let tree_latex = template.replace("<img_ref>", img_ref);
        Ok(tree_latex.replace("<caption>", caption))
    }

    // output audio
    pub fn write_slices_audio(
        &self,
        clustering: &[Vec<usize>],
        audio: &[AudioData],
        n_gaps: usize
    ) {
        for (i, cluster) in clustering.iter().enumerate() {
            if cluster.len() > 0 {
                let filename = format!("{}/cluster_{}.wav", self.out_audio, i);
                let spec = audio[cluster[0]].spec;
                let mut output = AudioData {
                    id: 0,
                    spec,
                    data: vec![],
                };
                for audio_id in cluster {
                    output.append(
                        n_gaps,
                        &mut audio[*audio_id].clone()
                    );
                }
                output.write(filename);
            }
        }
    }
}

