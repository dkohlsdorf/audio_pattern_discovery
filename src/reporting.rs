extern crate toml;

use crate::clustering::*;
use crate::aligned_model_merging::*;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::prelude::*;

/**
 * Latex templating
 */
#[derive(Deserialize, Debug)]
pub struct Templates {
    img_w: String,
    img_h: String,
    output: String,
    document: String,
    dendogram: String,
}

impl Templates {
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
        let mut file_out = File::create(format!("{}/{}", self.output, filename))?;
        file_out.write_fmt(format_args!("{}", result))?;
        Ok(())
    }

    /// Generate tikz of markov chain
    pub fn gen_markov(&self, filename: String, markov_model: HiddenMarkovModel) -> Result<()> {
        let mut s = "digraph {".to_string();
        for i in 0 .. markov_model.n_states {
            if markov_model.start[i] > 0.0 {
                s.push_str(&format!("\tstart -> {};", i));
            }
            if markov_model.stop[i] > 0.0 {
                s.push_str(&format!("\t{} -> stop;", i));
            }
            for j in 0 .. markov_model.n_states {
                if markov_model.trans[i * markov_model.n_states + j] > 0.0 {
                    s.push_str(&format!("\t{} -> {};", i, j));
                }
            }
        }
        s.push_str("}");
        let mut fp = File::create(&format!("{}/{}", self.output, filename))?;
        fp.write_fmt(format_args!("{}", s));                
        Ok(())
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
                    let graphics_i = self.image_ref(&images[i]);
                    let graphics_j = self.image_ref(&images[j]);
                    results.insert(k, format!("[.{} [{} {} ] ]", k, graphics_i, graphics_j));
                }
                Merge::Sequence2Cluster => {
                    let graphics_i = self.image_ref(&images[i]);
                    let subtree_j = &results[&j];
                    results.insert(k, format!("[.{} [{} {} ] ]", k, graphics_i, subtree_j));
                }
                Merge::Cluster2Sequence => {
                    let subtree_i = &results[&i];
                    let graphics_j = self.image_ref(&images[j]);
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
                },
                _ => println!("Cluster not found: {} | Singular cluster", cluster)
            }
        }
        Ok(latex_parts)
    }

    /// Plot a gray scale image 
    pub fn plot(&self, file: String, pixels: &[u8], rows: u32, cols: u32) -> Result<()> {
        let output = File::create(format!("{}/{}", self.output, file))?;
        let encoder = image::png::PNGEncoder::new(output);
        encoder
            .encode(&pixels, cols, rows, image::ColorType::Gray(8))
            .unwrap();
        Ok(())
    }

    /// Latex image reference
    fn image_ref(&self, name: &String) -> String {
        format!(
            "{{\\includegraphics[width={}, height={}]{{{}}}}}\n",
            self.img_w, self.img_h, name
        )
    }

    /// Set a tikz image
    fn tikz(&self, tree: &String, caption: &String) -> Result<String> {
        let mut file = File::open(&self.dendogram)?;
        let mut template = String::new();
        file.read_to_string(&mut template)?;
        let tree_latex = template.replace("<tree>", tree);
        Ok(tree_latex.replace("<caption>", caption))
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
