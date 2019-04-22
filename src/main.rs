#[macro_use]
extern crate serde_derive;
extern crate glob;
extern crate image;
extern crate rayon;
extern crate rand;

use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::env;
use std::time::Instant;

pub mod alignments;
pub mod audio;
pub mod clustering;
pub mod discovery;
pub mod error;
pub mod numerics;
pub mod reporting;
pub mod spectrogram;
pub mod neural;

fn main() {
    println!("==== Pattern Discovery ====");
    println!("# by Daniel Kohlsdorf     #");
    println!("===========================");

    let templates = reporting::Templates::from_toml("project/config/Templates.toml".to_string());
    let discover = discovery::Discovery::from_toml("project/config/Discovery.toml".to_string());
    println!("Template Config:  {:?}", templates);
    println!("Discovery Config: {:?}", discover);

    let args: Vec<String> = env::args().collect();
    let folder = &args[1];

    println!("Args: {:?}", args);
    auto_encoder(folder, &templates, &discover);
    learn(folder, &templates, &discover);
}

fn all_files(folder: &str) -> Vec<String> {
    let mut audio_files: Vec<String> = vec![];
    for entry in glob::glob(&format!("{}/**/*.wav", folder)).unwrap() {
        match entry {
            Ok(path) => {
                if !path.to_string_lossy().contains("cluster") {
                    println!("File: {}", path.to_string_lossy());
                    audio_files.push(String::from(path.to_string_lossy().clone()));
                }
            }
            Err(e) => println!("{:?}", e),
        }
    }
    audio_files
}

fn auto_encoder(folder: &str, templates: &reporting::Templates, discover: &discovery::Discovery) {
    let audio_files: Vec<String> = all_files(folder);
    println!("==== Extract Interesting Regions ==== ");
    let raw: Vec<audio::AudioData> = audio_files
        .par_iter()
        .enumerate()
        .map(|(i, file)| audio::AudioData::from_file(&file, i))
        .collect();
    println!("Extracting Spectrograms");
    let spectrograms: Vec<spectrogram::NDSequence> = raw
        .par_iter()
        .map(|raw| {
            spectrogram::NDSequence::new(
                discover.dft_win,
                discover.dft_step,
                discover.ceps_filter,
                raw
            )
        })
        .collect();
    println!("Extracting Interesting Regions");
    let interesting: Vec<spectrogram::Slice> = spectrograms
        .par_iter()
        .flat_map(|spectrogram| {
            spectrogram.interesting_ranges(
                discover.vat_moving,
                discover.vat_percentile,
                discover.vat_min_len,
            )
        })
        .collect();
    println!("Extracting Signals");
    let signals: Vec<spectrogram::NDSequence> = interesting
        .par_iter()
        .map(spectrogram::Slice::extract)
        .collect();
    println!("==== Learn Auto Encoder ==== ");
    let mut nn = neural::AutoEncoder::new(signals[0].n_bins, discover.auto_encoder);
    for _epoch in 0 .. discover.epochs {
        let mut total = 0.0;
        let mut total_err = 0.0;
        for signal in &signals {
            let mut order: Vec<usize> = (0 .. signal.len()).collect();
            let slice: &mut [usize] = &mut order;
            thread_rng().shuffle(slice);

            for i in slice {
                let x = numerics::Mat{ flat: signal.vec(*i).to_vec(), cols: signal.n_bins };
                let error = nn.take_step(&x, discover.learning_rate);
                total_err += error;
                total += 1.0;
            }
        }
        println!("{}", total_err/ total);
    }
    templates.save_encoder(nn).unwrap();
    println!("==== Done! ==== ");
}

fn learn(folder: &str, templates: &reporting::Templates, discover: &discovery::Discovery) {
    let audio_files: Vec<String> = all_files(folder);
    let nn = templates.read_encoder().unwrap();
    println!("==== Extract Interesting Regions ==== ");
    let raw: Vec<audio::AudioData> = audio_files
        .par_iter()
        .enumerate()
        .map(|(i, file)| audio::AudioData::from_file(&file, i))
        .collect();
    println!("Extracting Spectrograms");
    let spectrograms: Vec<spectrogram::NDSequence> = raw
        .par_iter()
        .map(|raw| {
            spectrogram::NDSequence::new(
                discover.dft_win,
                discover.dft_step,
                discover.ceps_filter,
                raw
            )
        })
        .collect();
    println!("Extracting Interesting Regions");
    let interesting: Vec<spectrogram::Slice> = spectrograms
        .par_iter()
        .flat_map(|spectrogram| {
            spectrogram.interesting_ranges(
                discover.vat_moving,
                discover.vat_percentile,
                discover.vat_min_len,
            )
        })
        .collect();
    println!("Extracting Signals");
    let signals: Vec<spectrogram::NDSequence> = interesting
        .par_iter()
        .map(|x| {
            spectrogram::Slice::extract(x).encoded(&nn)
        })
        .collect();
    let rates: Vec<u32> = raw.iter().map(|wav| wav.spec.sample_rate).collect();
    
    println!("==== Plot All Regions ==== ");
    let mut file_names = vec![];
    let mut file_names_ceps = vec![];
    for (i, signal) in signals.iter().enumerate() {
        let rate = raw[interesting[i].sequence.audio_id].spec.sample_rate as f32;
        println!(
            "Region {} {} {}",
            interesting[i].sequence.audio_id,
            (interesting[i].start * discover.dft_step) as f32 / rate,
            (interesting[i].stop * discover.dft_step) as f32 / rate
        );
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

    println!("==== Starting Alignment And Clustering ==== ");
    let n = signals.len();
    let mut workers = alignments::AlignmentWorkers::new(signals);
    let now = Instant::now();
    workers.align_all(&discover);
    println!("Align 8 threads took {}", now.elapsed().as_secs());

    let result = workers.result.lock().unwrap();
    let distances: Vec<f32> = result.iter().map(alignments::Alignment::score).collect();
    let (operations, clusters) = clustering::AgglomerativeClustering::clustering(
        distances,
        n,
        discover.clustering_percentile,
    );

    println!("==== Writing Cluster Audio ==== ");
    let grouped = clustering::AgglomerativeClustering::cluster_sets(
        &operations,
        &clusters,
        interesting.len(),
    );
    templates.write_slices_audio(&grouped, &interesting, &raw, discover.dft_step, 10000);
    let _ = templates.dump_slices(
        "output/detections_clusters.txt".to_string(),
        &grouped,
        &interesting,
        &audio_files,
        &rates,
        discover.dft_step,
    );
    println!("==== Generate Report ==== ");
    let mut clustering_files = vec![];
    for cluster in 0..grouped.len() {
        let filename = format!("cluster_{}.wav", cluster);
        clustering_files.push(filename);
    }
    let _ = templates.write_html(
        "output/result.html".to_string(),
        &clustering_files,
        &[],
    );
        if let Ok(ceps_tex) = templates.dendograms(&operations, &clusters, file_names_ceps) {
            if let Ok(spec_tex) = templates.dendograms(&operations, &clusters, file_names) {
                let mut latex_parts =
                    vec!["\\chapter{Clusters With Cepstrum Visualisation}".to_string()];
                latex_parts.extend(ceps_tex);
                latex_parts.push("\\chapter{Clusters With Spectrum Visualisation}".to_string());
                latex_parts.extend(spec_tex);
                let _ = templates.generate_doc("results.tex".to_string(), latex_parts);
            }
        }
    
    println!("==== Done! ==== ");
}
