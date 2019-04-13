#[macro_use]
extern crate serde_derive;
extern crate image;
extern crate rayon;

use std::time::Instant;

// TODO: define model / align against model stop merging
// TODO: only merge in each cluster
// TODO: implement merge and decode
// TODO: config
// TODO: compile better report putting dot images into pdf
// TODO: structure output

pub mod aligned_model_merging;
pub mod alignments;
pub mod audio;
pub mod clustering;
pub mod numerics;
pub mod reporting;
pub mod spectrogram;

use std::fs::File;
use std::io::prelude::*;

fn main() {
    let mut template_conf = String::new();
    let _ = File::open("project/config/Templates.toml")
        .expect("Template file not found")
        .read_to_string(&mut template_conf);
    let templates: reporting::Templates = toml::from_str(&template_conf).unwrap();
    println!("{:?}", templates);

    let wav = audio::AudioData::from_file(String::from("test.wav"), 0);
    let spec = spectrogram::NDSequence::new(256, 128, 32, &wav);
    let interesting = spec.interesting_ranges(15, 0.85, 50);
    let mut signals = vec![];
    for (_, slice) in interesting.iter().enumerate() {
        signals.push(slice.extract());
    }

    let mut file_names = vec![];
    let mut file_names_ceps = vec![];
    for (i, signal) in signals.iter().enumerate() {
        let file_id = format!("spec_{}", i);
        let file_id_ceps = format!("ceps_{}", i);
        let file_spec = format!("spec_{}.png", i);
        let file_ceps = format!("ceps_{}.png", i);
        let _ = templates.plot(
            file_spec,
            &signal.img_spec(),
            signal.len_spec() as u32,
            signal.dft_win as u32,
        );
        let _ = templates.plot(
            file_ceps,
            &signal.img_ceps(),
            signal.len() as u32,
            signal.n_bins as u32,
        );
        file_names_ceps.push(file_id_ceps);
        file_names.push(file_id);
    }

    let n = signals.len();
    let mut workers = alignments::AlignmentWorkers::new(signals);
    let now = Instant::now();
    workers.align_all(8);
    println!("Align 8 threads took {}", now.elapsed().as_secs());

    let result = workers.result.lock().unwrap();
    let distances: Vec<f32> = result.iter().map(|node| node.score()).collect();
    let (operations, clusters) = clustering::AgglomerativeClustering::clustering(distances, n, 0.1);

    let grouped = clustering::AgglomerativeClustering::cluster_sets(
        &operations,
        &clusters,
        interesting.len(),
    );

    templates.write_slices_audio(&grouped, &interesting, &vec![wav], 128, 10000);
    let (model, _) = aligned_model_merging::HiddenMarkovModel::from_slices(&interesting);
    if let Ok(image) = templates.gen_markov("markov".to_string(), model) {
        if let Ok(ceps_tex) = templates.dendograms(&operations, &clusters, file_names_ceps) {
            if let Ok(spec_tex) = templates.dendograms(&operations, &clusters, file_names) {
                let mut latex_parts = ceps_tex;
                latex_parts.extend(spec_tex);
                latex_parts.push(image);
                let _ = templates.generate_doc("results.tex".to_string(), latex_parts);
            }
        }
    }
}
