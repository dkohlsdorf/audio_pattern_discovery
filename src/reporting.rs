extern crate toml;

use crate::hidden_markov_model::*;
use crate::audio::*;
use crate::clustering::*;
use crate::spectrogram::*;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::prelude::*;

#[derive(Deserialize, Debug)]
pub struct Templates {
    img_w: String,
    img_h: String,
    out_docs: String,
    out_images: String,
    out_audio: String,
    document: String,
    dendogram: String,
    figure: String,
    table: String
}

impl Templates {    
    /// load from config
    pub fn from_toml(file: String) -> Templates {
        let mut template_conf = String::new();
        let _ = File::open(file)
            .expect("Template file not found")
            .read_to_string(&mut template_conf);
        let templates: Templates = toml::from_str(&template_conf).unwrap();
        templates
    }

    /// Generates a latex table    
    pub fn table(&self, col_names: Vec<String>, cols: Vec<Vec<String>>) -> Result<String> {
        let mut file = File::open(&self.table)?;
        let mut template = String::new();
        file.read_to_string(&mut template)?;
        let mut formating = "|".to_string();
        for _ in 0 .. cols.len() {
            formating.push_str("c|");
        }
        let mut header = col_names[0].clone();
        for i in 1 .. col_names.len() {
            header.push_str(&format!("& {}", col_names[i]));
        }
        header.push_str("\\\\\n");
        let mut content = String::new();
        for col in cols {
            content.push_str(&col[0].clone());
            for i in 1 .. col.len() {
                content.push_str(&format!("& {}", &col[i]));
            }
            content.push_str("\\\\\n");
        }
        Ok(template
            .replace("<format>",  &formating)
            .replace("<heading>", &header)
            .replace("<content>", &content))
    }

    /// Generate a latex figure
    fn figure(&self, name: &String, caption: &String) -> Result<String> {
        let mut file = File::open(&self.figure)?;
        let mut template = String::new();
        file.read_to_string(&mut template)?;
        let img_ref = self.image_ref(name, false);
        let filled = template
            .replace("<caption>", caption)
            .replace("<img_ref>", &img_ref);
        Ok(filled)
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

    /// Generate tikz of markov chain
    pub fn gen_markov(
        &self,
        filename_without_extension: String,
        markov_model: &HiddenMarkovModel,
    ) -> Result<String> {
        let mut n_transitions = 0;
        let mut s = "digraph {\n".to_string();
        for i in 0 .. markov_model.n_states {
            if markov_model.is_segmental[i] {
                s.push_str(&format!("{} [color=red];\n", i));
            }
        }
        for i in 0..markov_model.n_states {
            if markov_model.start[i] > 0.0 {
                s.push_str(&format!("\tstart -> {};\n", i));
            }
            if markov_model.stop[i] > 0.0 {
                s.push_str(&format!("\t{} -> stop;\n", i));
            }
            for j in 0..markov_model.n_states {
                if markov_model.trans[i * markov_model.n_states + j] > 0.0 {
                    s.push_str(&format!("\t{} -> {} [label=\"{:.2}\"];\n", i, j, markov_model.trans[i * markov_model.n_states + j]));
                    n_transitions += 1;
                }
            }
        }
        println!("Creating model with: {}", n_transitions);
        s.push_str("}");
        let filename = format!("{}.dot", filename_without_extension);
        let mut fp = File::create(&format!("{}/{}", self.out_docs, filename))?;
        fp.write_fmt(format_args!("{}", s))?;
        self.figure(&filename_without_extension, &format!("${}$", &filename_without_extension))
    }

    /// Dendogram generation from clustering results
    pub fn dendograms(
        &self,
        operations: &[ClusteringOperation],
        clusters: &HashSet<usize>,
        images: Vec<String>,
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
                    let graph = self.tikz(result, &caption)?;
                    latex_parts.push(graph);
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
        encoder
            .encode(&pixels, cols, rows, image::ColorType::Gray(8))
            .unwrap();
        Ok(())
    }

    /// Latex image reference
    fn image_ref(&self, name: &String, set_dim: bool) -> String {
        let path = format!("../{}/{}", self.out_images, name);
        if set_dim {
            format!(
                "{{\\includegraphics[width={}, height={}]{{{}}}}}\n",
                self.img_w, self.img_h, path
            )
        } else {
            format!("\\includegraphics[height=\\textheight]{{{}}}\n", path)
        }
    }

    /// Set a tikz image
    fn tikz(&self, tree: &String, caption: &String) -> Result<String> {
        let mut file = File::open(&self.dendogram)?;
        let mut template = String::new();
        file.read_to_string(&mut template)?;
        let tree_latex = template.replace("<tree>", tree);
        Ok(tree_latex.replace("<caption>", caption))
    }

    // output audio
    pub fn write_slices_audio(
        &self,
        clustering: &[Vec<usize>],
        slices: &[Slice],
        audio: &[AudioData],
        win_step: usize,
        n_gaps: usize,
    ) {
        for (i, cluster) in clustering.iter().enumerate() {
            let filename = format!("{}/cluster_{}.wav", self.out_audio, i);
            let spec = audio[slices[cluster[0]].sequence.audio_id].spec;
            let mut output = AudioData {
                id: 0,
                spec,
                data: vec![],
            };
            for slice_id in cluster {
                let slice = &slices[*slice_id];
                let raw = &audio[slice.sequence.audio_id];
                output.append(
                    n_gaps,
                    &mut raw.slice(slice.start * win_step, slice.stop * win_step),
                );
            }
            output.write(filename);
        }
    }
}

#[derive(Debug)]
pub enum Error {
    IO(io::Error),
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Error {
        Error::IO(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
