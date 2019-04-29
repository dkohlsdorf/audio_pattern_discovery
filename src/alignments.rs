use crate::discovery::Discovery;
use crate::numerics::*;
use crate::spectrogram::NDSequence;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

/**
 * Aligns all sequences in parallel and saves the results in a flat matrix  
 */
pub struct AlignmentWorkers {
    pub data: Arc<Vec<NDSequence>>,
    pub result: Arc<Mutex<Vec<f32>>>,
}

impl AlignmentWorkers {
    pub fn new(data: Vec<NDSequence>) -> AlignmentWorkers {
        let n = data.len();
        let data = Arc::from(data);
        let mut alignments = vec![];
        for _ in 0..n * n {
            alignments.push(0.0);
        }
        let result = Arc::from(Mutex::from(alignments));
        AlignmentWorkers { data, result }
    }

    /**
     * The actual alignment job using n workers
     */
    pub fn align_all(&mut self, params: &Discovery) {
        let n = self.data.len();
        let batch_size = (n / params.alignment_workers) + 1;
        let mut children = vec![];
        for batch in 0..params.alignment_workers {
            let start = batch * batch_size;
            let stop = usize::min((batch + 1) * batch_size, n);
            let data = self.data.clone();
            let result = self.result.clone();
            let params = params.clone();
            let th = thread::spawn(move || {
                for i in start..stop {
                    println!(
                        "Thread: {} instance: {}: {} x {}",
                        batch,
                        i,
                        data[i].len(),
                        data[i].n_bins
                    );
                    for j in 0..n {
                        if i != j {
                            let len = usize::max(data[i].len(), data[j].len());
                            let params = params.alignment_params(len);
                            let mut alignment = Alignment::new();
                            alignment.construct_alignment(&data[i], &data[j], &params);
                            let mut result = result.lock().unwrap();
                            result[i * n + j] = alignment.score();
                        }
                    }
                }
            });
            children.push(th);
        }
        for child in children {
            let _ = child.join();
        }
    }
}

/**
 * Parameters of alignment
 *
 * The warping band is described by sakoe and chiba.
 * The restart threshold allows for local alignments.
 * The insertion, deletion and match penalty allow to weigh errors differently.
 */
#[derive(Clone, Debug)]
pub struct AlignmentParams {
    pub warping_band: usize,
    pub insertion_penalty: f32,
    pub deletion_penalty: f32,
    pub match_penalty: f32,
}

impl AlignmentParams {
    pub fn default(len: usize) -> AlignmentParams {
        AlignmentParams {
            warping_band: len,
            insertion_penalty: 1.0,
            deletion_penalty: 1.0,
            match_penalty: 1.0,
        }
    }
}

/**
 * Compute alignment between two sequences
 */
#[derive(Debug)]
pub struct Alignment {
    pub n: usize,
    pub m: usize,
    pub sparse: HashMap<(usize, usize), f32>,
}

impl Alignment {
    pub fn new() -> Alignment {
        let mut sparse = HashMap::new();
        sparse.insert((0, 0), 0.0);
        Alignment { n: 0, m: 0, sparse }
    }

    /**
     * Alignment score, normalised to account for length variations
     */
    pub fn score(&self) -> f32 {
        if self.m == 0 && self.n == 0 {
            std::f32::INFINITY
        } else {
            match self.sparse.get(&(self.n - 1, self.m - 1)) {
                Some(score) => score / (self.n + self.m) as f32,
                None => std::f32::INFINITY,
            }
        }
    }
    /**
     * Build the best alignment node at the current stage of the alignment
     */
    fn alignment_score(
        &self,
        i: usize,
        j: usize,
        x: &NDSequence,
        y: &NDSequence,
        params: &AlignmentParams,
    ) -> f32 {
        let distance = euclidean(x.vec(i - 1), y.vec(j - 1));
        // Check for a match on the diagonal
        let match_score = match self.sparse.get(&(i - 1, j - 1)) {
            Some(score) => *score,
            None => std::f32::INFINITY,
        };
        // Check for an insertion error
        let insert_score = match self.sparse.get(&(i - 1, j)) {
            Some(score) => *score,
            None => std::f32::INFINITY,
        };
        // Check for a deletion error
        let delete_score = match self.sparse.get(&(i, j - 1)) {
            Some(score) => *score,
            None => std::f32::INFINITY,
        };
        if delete_score < match_score && delete_score < insert_score {
            delete_score + params.deletion_penalty * distance
        } else if insert_score < match_score && insert_score < delete_score {
            insert_score + params.insertion_penalty * distance
        } else {
            match_score + params.match_penalty * distance
        }
    }

    /**
     * Compute the dynamic time warping distance along with all alignment information.
     */
    pub fn construct_alignment(
        &mut self,
        x: &NDSequence,
        y: &NDSequence,
        params: &AlignmentParams,
    ) {
        self.n = x.len();
        self.m = y.len();
        let w = usize::max(params.warping_band, abs(self.n, self.m)) + 2;
        for i in 1..=self.n {
            for j in usize::max(diff(i, w), 1)..usize::min(i + w, self.m + 1) {
                let node = self.alignment_score(i, j, &x, &y, params);
                self.sparse.insert((i, j), node);
            }
        }
    }
}
