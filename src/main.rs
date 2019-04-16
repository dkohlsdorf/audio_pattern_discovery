#[macro_use]
extern crate serde_derive;
extern crate glob;
extern crate image;
extern crate rayon;

use rayon::prelude::*;
use std::env;
use std::time::Instant;

pub mod aligned_model_merging;
pub mod alignments;
pub mod audio;
pub mod clustering;
pub mod discovery;
pub mod hidden_markov_model;
pub mod numerics;
pub mod reporting;
pub mod spectrogram;

fn main() {
    println!("==== Pattern Discovery ====");
    println!("# by Daniel Kohlsdorf     #");
    println!("===========================");
    let templates = reporting::Templates::from_toml("project/config/Templates.toml".to_string());
    let discover = discovery::Discovery::from_toml("project/config/Discovery.toml".to_string());
    println!("Template Config:  {:?}", templates);
    println!("Discovery Config: {:?}", discover);

    let args: Vec<String> = env::args().collect();
    println!("Args: {:?}", args);
    let mut audio_files: Vec<String> = vec![];
    for entry in glob::glob(&format!("{}/**/*.wav", args[1])).unwrap() {
        match entry {
            Ok(path) => {
                println!("File: {}", path.to_string_lossy());
                audio_files.push(String::from(path.to_string_lossy().clone()));
            }
            Err(e) => println!("{:?}", e),
        }
    }

    println!("==== Extract Interesting Regions ==== ");
    let raw: Vec<audio::AudioData> = audio_files
        .par_iter()
        .map(|file| audio::AudioData::from_file(&file, 0))
        .collect();
    let spectrograms: Vec<spectrogram::NDSequence> = raw
        .par_iter()
        .map(|raw| {
            spectrogram::NDSequence::new(
                discover.dft_win,
                discover.dft_step,
                discover.ceps_filter,
                raw,
            )
        })
        .collect();
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
    let signals: Vec<spectrogram::NDSequence> = interesting
        .par_iter()
        .map(|slice| slice.extract())
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
    let distances: Vec<f32> = result.iter().map(|alignment| alignment.score()).collect();
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
    println!("==== Model Merging ==== ");
    let mut sample_distances = vec![];
    let n = interesting.len();
    for i in 0..n {
        for j in 0..n {
            let path = result[i * n + j].path();
            if path.len() == 0 {
                println!(
                    "No Alignment Found {} {} {} {}",
                    i,
                    j,
                    result[i * n + j].n,
                    result[i * n + j].m
                );
            }
            for node in path {
                sample_distances.push(node.score);
            }
        }
    }
    let merge_threshold = numerics::percentile(&mut sample_distances, discover.merging_percentile);
    let mut hmm_parts = vec![];
    let mut hmms = vec![];
    for (c, cluster) in grouped.iter().enumerate() {
        println!("HMM model merging from: {}", c);
        let mut instances = vec![];
        let mut paths = vec![];
        let n = workers.data.len();
        for (x, i) in cluster.iter().enumerate() {
            instances.push(interesting[*i].clone());
            for (y, j) in cluster.iter().enumerate() {
                if y < x {
                    let alignment = result[*i * n + *j].path();
                    println!("Path between {} and {} is : {}", x, y, alignment.len());
                    paths.push((x, y, alignment));
                }
            }
        }
        let mut merger = aligned_model_merging::ModelMerging::from_slices(&instances);
        merger.merge_all(
            &paths,
            &instances,
            discover.merging_internal_percentile,
            merge_threshold,
            discover.merging_moving,
        );
        hmms.push(merger.shrink());
        if let Ok(img) = templates.gen_markov(format!("markovchain_{}", c), &merger.shrink()) {
            hmm_parts.push(img);
        }
    }

    println!("==== Decoding ==== ");
    let col_names = vec![
        "Cluster".to_string(),
        "HMM".to_string(),
        "Sequence".to_string(),
        "LL".to_string(),
    ];
    let mut cols = vec![];
    let mut accuracy = 0.0;
    let mut n = 0.0;
    for (c, cluster) in grouped.iter().enumerate() {
        for s in cluster {
            let mut max_ll = std::f32::NEG_INFINITY;
            let mut max_hmm = 0;
            for (i, hmm) in hmms.iter().enumerate() {
                let ll = hmm.viterbi(&interesting[*s].extract());
                if ll > max_ll {
                    max_hmm = i;
                    max_ll = ll;
                }
            }
            if max_hmm == c {
                accuracy += 1.0;
            }
            n += 1.0;
            cols.push(vec![
                c.to_string(),
                max_hmm.to_string(),
                s.to_string(),
                format!("{:.1}", max_ll),
            ]);
        }
    }
    let accuracy = accuracy / n;

    println!("==== Generate Report ==== ");
    let mut clustering_files = vec![];
    for cluster in 0..grouped.len() {
        let filename = format!("cluster_{}.wav", cluster);
        clustering_files.push(filename);
    }
    let _ = templates.write_html("output/result.html".to_string(), &clustering_files, &vec![]);
    if let Ok(table) = templates.table(col_names, cols) {
        if let Ok(ceps_tex) = templates.dendograms(&operations, &clusters, file_names_ceps) {
            if let Ok(spec_tex) = templates.dendograms(&operations, &clusters, file_names) {
                let mut latex_parts =
                    vec!["\\chapter{Clusters With Cepstrum Visualisation}".to_string()];
                latex_parts.extend(ceps_tex);
                latex_parts.push("\\chapter{Clusters With Spectrum Visualisation}".to_string());
                latex_parts.extend(spec_tex);
                latex_parts.push("\\chapter{Hidden Markov Models For Each Cluster}".to_string());
                latex_parts.extend(hmm_parts);
                latex_parts.push("\\chapter{Log Likelihoods}".to_string());
                latex_parts.push(table);
                latex_parts.push(format!("Accuracy: ${}$\n", accuracy));
                let _ = templates.generate_doc("results.tex".to_string(), latex_parts);
            }
        }
    }
    println!("==== Done! ==== ");
}
