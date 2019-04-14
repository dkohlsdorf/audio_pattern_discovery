#[macro_use]
extern crate serde_derive;
extern crate image;
extern crate rayon;

use std::time::Instant;

// TODO: Pretty up reports
// TODO: Pretty up output
// TODO: Move main -> discovery
// TODO: UI with html and javascript >> poll piped log file from server
// TODO: Project management ... save project folders ... move them to a specific location

pub mod aligned_model_merging;
pub mod alignments;
pub mod audio;
pub mod clustering;
pub mod discovery;
pub mod numerics;
pub mod reporting;
pub mod spectrogram;

fn main() {
    let templates = reporting::Templates::from_toml("project/config/Templates.toml".to_string());
    let discover = discovery::Discovery::from_toml("project/config/Discovery.toml".to_string());
    let wav = audio::AudioData::from_file(String::from("test.wav"), 0);
    let spec = spectrogram::NDSequence::new(
        discover.dft_win,
        discover.dft_step,
        discover.ceps_filter,
        &wav,
    );
    let interesting = spec.interesting_ranges(
        discover.vat_moving,
        discover.vat_percentile,
        discover.vat_min_len,
    );
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
    workers.align_all(discover.alignment_workers);
    println!("Align 8 threads took {}", now.elapsed().as_secs());

    let result = workers.result.lock().unwrap();
    let distances: Vec<f32> = result.iter().map(|node| node.score()).collect();
    let (operations, clusters) = clustering::AgglomerativeClustering::clustering(
        distances,
        n,
        discover.clustering_percentile,
    );

    let grouped = clustering::AgglomerativeClustering::cluster_sets(
        &operations,
        &clusters,
        interesting.len(),
    );
    templates.write_slices_audio(&grouped, &interesting, &vec![wav], 128, 10000);

    let mut hmm_parts = vec![];
    for cluster in grouped {
        let mut instances = vec![];
        let mut paths = vec![];
        let n = workers.data.len();
        for (x, i) in cluster.iter().enumerate() {
            instances.push(interesting[*i].clone());        
            for (y, j) in cluster.iter().enumerate() {
                let alignment = workers.result.clone().lock().unwrap()[*i * n + *j].path();
                paths.push((x, y, alignment));
            }
        }
        let mut merger = aligned_model_merging::ModelMerging::from_slices(&instances);
        merger.merge_all(
            &paths,
            &instances,
            discover.merging_percentile,
            discover.merging_moving,
        );
        if let Ok(img) = templates.gen_markov("markov".to_string(), merger.shrink()) {
            hmm_parts.push(img);
        }
    }

    if let Ok(ceps_tex) = templates.dendograms(&operations, &clusters, file_names_ceps) {
        if let Ok(spec_tex) = templates.dendograms(&operations, &clusters, file_names) {
            let mut latex_parts = ceps_tex;
            latex_parts.extend(spec_tex);
            latex_parts.extend(hmm_parts);
            let _ = templates.generate_doc("results.tex".to_string(), latex_parts);
        }
    }
}