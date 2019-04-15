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
    pub result: Arc<Mutex<Vec<Alignment>>>,
}

impl AlignmentWorkers {
    pub fn new(data: Vec<NDSequence>) -> AlignmentWorkers {
        let n = data.len();
        let data = Arc::from(data);
        let mut alignments = vec![];
        for _ in 0..n * n {
            alignments.push(Alignment::new());
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
                    println!("Thread: {} instance: {}", batch, i);
                    for j in 0..n {
                        if i != j {
                            let len = usize::max(data[i].len(), data[j].len());
                            let params = params.alignment_params(len);
                            let mut alignment = Alignment::new();
                            alignment.construct_alignment(&data[i], &data[j], &params);
                            let mut result = result.lock().unwrap();
                            result[i * n + j] = alignment;
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
 * Labels for the error types during alignment
 */
#[derive(Debug, Clone)]
pub enum AlignmentLabel {
    Insertion,
    Deletion,
    Match,
}

/**
 * The alignment's progress, holds which node we came from,
 * what the error is at this stage, the distance at exactly this location
 * and the distance along the warping path
 */
#[derive(Debug, Clone)]
pub struct AlignmentNode {
    pub prev_i: usize,
    pub prev_j: usize,
    pub label: AlignmentLabel,
    pub score: f32,
    pub score_on_path: f32,
}

impl AlignmentNode {
    pub fn cur(&self) -> (usize, usize) {
        match self.label {
            AlignmentLabel::Match => (self.prev_i + 1, self.prev_j + 1),
            AlignmentLabel::Insertion => (self.prev_i + 1, self.prev_j),
            AlignmentLabel::Deletion => (self.prev_i, self.prev_j + 1),
        }
    }

    pub fn start_node() -> AlignmentNode {
        AlignmentNode {
            prev_i: 0,
            prev_j: 0,
            label: AlignmentLabel::Match,
            score: 0.0,
            score_on_path: 0.0,
        }
    }

    pub fn new(
        prev_i: usize,
        prev_j: usize,
        label: AlignmentLabel,
        score: f32,
        score_on_path: f32,
    ) -> AlignmentNode {
        AlignmentNode {
            prev_i,
            prev_j,
            label,
            score,
            score_on_path,
        }
    }

    pub fn is_start(&self) -> bool {
        self.prev_i == 0 && self.prev_j == 0 && self.score_on_path == 0.0
    }
}

/**
 * Compute alignment between two sequences
 */
#[derive(Debug)]
pub struct Alignment {
    pub n: usize,
    pub m: usize,
    pub sparse: HashMap<(usize, usize), AlignmentNode>,
}

impl Alignment {
    pub fn new() -> Alignment {
        let mut sparse = HashMap::new();
        sparse.insert((0, 0), AlignmentNode::start_node());
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
                Some(node) => node.score_on_path / (self.n + self.m) as f32,
                None => std::f32::INFINITY,
            }
        }
    }

    /**
     * Back tracking through alignment
     */
    pub fn path(&self) -> Vec<AlignmentNode> {
        if self.n > 0 && self.m > 0 {
            if let Some(node) = self.sparse.get(&(self.n, self.m)) {
                let mut node = node.clone();
                let mut path = vec![];
                while !node.is_start() {
                    path.push(node.clone());
                    node = self.sparse[&(node.prev_i, node.prev_j)].clone();
                }
                path.reverse();
                path
            } else {
                let is_empty = self.m == 0 && self.n == 0;
                println!("Warning: can not find {} {} {}", self.n, self.m, is_empty);
                vec![]
            }
        } else {
            vec![]
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
    ) -> AlignmentNode {
        let distance = euclidean(x.vec(i - 1), y.vec(j - 1));
        // Check for a match on the diagonal
        let (match_i, match_j, match_score) = match self.sparse.get(&(i - 1, j - 1)) {
            Some(node) => (i - 1, j - 1, node.score_on_path),
            None => (i - 1, j - 1, std::f32::INFINITY),
        };
        // Check for an insertion error
        let (insert_i, insert_j, insert_score) = match self.sparse.get(&(i - 1, j)) {
            Some(node) => (i - 1, j, node.score_on_path),
            None => (i - 1, j, std::f32::INFINITY),
        };
        // Check for a deletion error
        let (delete_i, delete_j, delete_score) = match self.sparse.get(&(i, j - 1)) {
            Some(node) => (i, j - 1, node.score_on_path),
            None => (i, j - 1, std::f32::INFINITY),
        };
        if delete_score < match_score && delete_score < insert_score {
            AlignmentNode {
                prev_i: delete_i,
                prev_j: delete_j,
                label: AlignmentLabel::Deletion,
                score: distance,
                score_on_path: delete_score + params.deletion_penalty * distance,
            }
        } else if insert_score < match_score && insert_score < delete_score {
            AlignmentNode {
                prev_i: insert_i,
                prev_j: insert_j,
                label: AlignmentLabel::Insertion,
                score: distance,
                score_on_path: insert_score + params.insertion_penalty * distance,
            }
        } else {
            AlignmentNode {
                prev_i: match_i,
                prev_j: match_j,
                label: AlignmentLabel::Match,
                score: distance,
                score_on_path: match_score + params.match_penalty * distance,
            }
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
        let mut last_seen = (0, 0);
        for i in 1..self.n + 1 {
            for j in usize::max(diff(i, w), 1)..usize::min(i + w, self.m + 1) {
                let node = self.alignment_score(i, j, &x, &y, params);
                self.sparse.insert((i, j), node);
                last_seen = (i, j);
            }
        }
        if self.n != last_seen.0 && self.m != last_seen.1 {
            println!(
                "Aligning: {} {} {} {} {}",
                self.n, self.m, last_seen.0, last_seen.1, w
            );
        }
    }
}
