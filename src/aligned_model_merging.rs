use crate::alignments::*;
use crate::numerics::*;
use crate::spectrogram::*;

use std::collections::{HashMap, HashSet};
use crate::hidden_markov_model::*;

/**
 * Model merging: Build HMM by merging states
 */
pub struct ModelMerging {
    pub hmm: HiddenMarkovModel,
    pub state_map: HashMap<(usize, usize), usize>, // (original_seq, timestep) -> state
    pub merge_parent: Vec<usize>,                  // union find parent structure to manage merges
}

impl ModelMerging {
    /**
     * Shrink hidden markov model to smaller compressed version     
     */
    pub fn shrink(&self) -> HiddenMarkovModel {
        let mut used_states = HashMap::new();
        let mut n_states = 0;
        // map each state to a state in the clustered version
        for i in 0..self.merge_parent.len() {
            let state = self.find_parent(i);
            if !used_states.contains_key(&state) {
                used_states.insert(state, n_states);
                n_states += 1;
            }
        }
        println!(
            "Original model: {} Compressed model: {}",
            self.hmm.n_states, n_states
        );
        // initialize all transitions
        let mut start = vec![0.0; n_states];
        let mut stop = vec![0.0; n_states];
        let mut trans = vec![0.0; n_states * n_states];
        let mut states = vec![0.0; n_states * self.hmm.dim];
        let mut is_segmental = vec![false; n_states]; 
        // copy the states
        for i in 0..self.hmm.n_states {
            let i = self.find_parent(i);
            let state_i    = used_states[&i];
            start[state_i] = self.hmm.start[i];
            stop[state_i]  = self.hmm.stop[i];
            is_segmental[state_i] = self.hmm.is_segmental[i];
            for j in 0..self.hmm.dim {
                states[state_i * self.hmm.dim + j] = self.hmm.states[i * self.hmm.dim + j];
            }
            for j in 0..self.hmm.n_states {
                let j = self.find_parent(j);
                let state_j = used_states[&j];
                trans[state_i * n_states + state_j] = self.hmm.trans[i * self.hmm.n_states + j];
            }
        }
        let dim = self.hmm.dim;
        let mut hmm = HiddenMarkovModel {
            n_states,
            dim,
            trans,
            start,
            stop,
            states,
            is_segmental
        };
        hmm.normalize_transitions();
        hmm
    }

    /**
     * Convert paths to merge operations. 
     * A merge happens only for matches within a certain distance.
     * All distances within a sequence are smoothed using a moving average of size k.
     */
    pub fn merge_all(
        &mut self,
        paths: &[(usize, usize, Vec<AlignmentNode>)],
        slices: &[Slice],
        perc: f32,
        th: f32,
        k: usize,
    ) { 
        let operations = ModelMerging::merges_from_alignments(paths, slices, perc, th , k);
        let mut closed = HashSet::new();
        for op in operations {
            let i = self.state_map[&(op.slice_i, op.i)];
            let j = self.state_map[&(op.slice_j, op.j)];
            let state_i = self.find_parent(i);
            let state_j = self.find_parent(j);
            println!(
                "Merging: {} and {} || with parent state {} and {}",
                i, j, state_i, state_j
            );
            if !closed.contains(&(state_i, state_j)) {
                closed.insert((state_i, state_j));
                closed.insert((state_j, state_i));
                self.merge(state_i, state_j, op.is_from_alignment);
            }
        }
    }

    /** 
     * Merge two states. 
     */
    fn merge(&mut self, state_i: usize, state_j: usize, from_alignment: bool) {
        if state_i <= state_j {
            // merge j into i
            if state_i != state_j {
                // move all outgoing connections from j to i
                for k in 0..self.hmm.n_states {
                    let to = k;
                    self.hmm.trans[state_i * self.hmm.n_states + to] +=
                        self.hmm.trans[state_j * self.hmm.n_states + to];
                }
                // move all incomming connections from j to i
                for k in 0..self.hmm.n_states {
                    let from = k;
                    self.hmm.trans[from * self.hmm.n_states + state_i] +=
                        self.hmm.trans[from * self.hmm.n_states + state_j];
                }
                // fix self transition, start and stop
                self.hmm.trans[state_i * self.hmm.n_states + state_i] += 1.0;
                self.hmm.start[state_i] += self.hmm.start[state_j];
                self.hmm.stop[state_i] += self.hmm.stop[state_j];
                for d in 0..self.hmm.dim {
                    self.hmm.states[state_i * self.hmm.dim + d] +=
                        self.hmm.states[state_j * self.hmm.dim + d];
                    self.hmm.states[state_i * self.hmm.dim + d] /= 2.0;
                }
                // change parent in union find               
                self.hmm.is_segmental[state_i] = self.hmm.is_segmental[state_i] || from_alignment;
                self.merge_parent[state_j] = state_i;
            }
        } else {
            self.merge(state_j, state_i, from_alignment);
        }
    }

    /**
     * Find which state this one is merged into using union find
     */
    fn find_parent(&self, i: usize) -> usize {
        let mut p = i;
        while p != self.merge_parent[p] {
            p = self.merge_parent[p];
        }
        p
    }

    /**
     * Create a simple chain of states one for each frame.
     */
    pub fn from_slices(slices: &[Slice]) -> ModelMerging {
        let dim = slices[0].sequence.n_bins;
        let mut states: Vec<f32> = vec![];
        let mut n_states = 0;
        let mut state_map: HashMap<(usize, usize), usize> = HashMap::new();
        for (i, slice) in slices.iter().enumerate() {
            let spec = slice.extract();
            states.extend(&spec.frames);
            for t in 0..spec.len() {
                state_map.insert((i, t), n_states);
                n_states += 1;
            }
        }
        let mut start = vec![0.0; n_states];
        let mut stop = vec![0.0; n_states];
        let mut trans = vec![0.0; n_states * n_states];
        for (i, slice) in slices.iter().enumerate() {
            let spec = slice.extract();
            for t in 0..spec.len() {
                if t == 0 {
                    start[state_map[&(i, t)]] = 1.0;
                }
                if t == spec.len() - 1 {
                    stop[state_map[&(i, t)]] = 1.0;
                }
                if t < spec.len() - 1 {
                    let from = state_map[&(i, t)];
                    let to = state_map[&(i, t + 1)];
                    trans[from * n_states + to] = 1.0;
                }
            }
        }
        let mut merge_parent = vec![];
        let mut is_segmental = vec![];
        for i in 0..n_states {
            merge_parent.push(i);
            is_segmental.push(false);
        }
        let hmm = HiddenMarkovModel {
            n_states,
            dim,
            start,
            stop,
            trans,
            states,
            is_segmental
        };
        ModelMerging {
            hmm,
            state_map,
            merge_parent,
        }
    }

    /**
     * Get all merges from a set of alignments
     */
    fn merges_from_alignments(
        paths: &[(usize, usize, Vec<AlignmentNode>)],
        slices: &[Slice],
        perc: f32,
        th: f32,
        k: usize,
    ) -> Vec<MergeOperation> {
        paths
            .iter()
            .flat_map(|(i, j, p)| ModelMerging::merges_from_alignment(*i, *j, slices, p, perc, th, k))
            .collect()
    }

    /**
     * Get all merges from one alignment
     */
    fn merges_from_alignment(
        x: usize,
        y: usize,
        slices: &[Slice],
        path: &[AlignmentNode],
        perc: f32,
        th: f32,
        k: usize,
    ) -> Vec<MergeOperation> {
        let distances: Vec<f32> = path.iter().map(|node| node.score).collect();
        println!("Merging along alignment with threshold: {}", th);
        // smooth all distances
        let mut moving_avg = vec![];
        for i in 0..distances.len() {
            let mut avg = 0.0;
            if i >= k {
                avg = mean(&distances[i - k..i]);
            }
            moving_avg.push(avg);
        }

        let slice_i = &slices[x].extract();
        let slice_j = &slices[y].extract();

        let mut dist_i = vec![0.0];
        for i in 1 .. slice_i.len() {
            dist_i.push(euclidean(slice_i.vec(i), slice_i.vec(i - 1)));
        }
        let mut dist_j = vec![0.0];
        for i in 1 .. slice_j.len() {
            dist_j.push(euclidean(slice_j.vec(i), slice_j.vec(i - 1)));
        }
        let internal_th_i = percentile(&mut dist_i.clone(), perc);
        let internal_th_j = percentile(&mut dist_j.clone(), perc);

        let mut operations = vec![];
        for (t, node) in path.iter().enumerate() {
            let (i, j) = node.cur();
            if i < slice_i.len() && j < slice_j.len() {
                // use smoothed distances along alignment path
                let distance = if t < k {
                    distances[t]
                } else {
                    moving_avg[t - k]
                };
                // check for merge possibility in the sequence
                let dist2prev_i = if i > 0 {
                    dist_i[i]
                } else {
                    std::f32::INFINITY
                };
                let dist2prev_j = if j > 0 {
                    dist_j[j]
                } else {
                    std::f32::INFINITY
                };            
                if dist2prev_i < internal_th_i {
                    operations.push(MergeOperation {
                        slice_i: x,
                        slice_j: x,
                        i: i,
                        j: i - 1,
                        dist: dist2prev_i,
                        is_from_alignment: false,
                    });
                }
                if dist2prev_j < internal_th_j {
                    operations.push(MergeOperation {
                        slice_i: y,
                        slice_j: y,
                        i: j,
                        j: j - 1,
                        dist: dist2prev_j,
                        is_from_alignment: false,
                    });
                }
                // we might merge on each match
                match node.label {
                    AlignmentLabel::Match if distance < th => operations.push(MergeOperation {
                        slice_i: x,
                        slice_j: y,
                        i: i,
                        j: j,
                        dist: distance,
                        is_from_alignment: true,
                    }),
                    _ => (),
                };
            }
        }
        operations
    }
}

#[derive(Debug)]
pub struct MergeOperation {
    pub slice_i: usize,
    pub slice_j: usize,
    pub i: usize,
    pub j: usize,
    pub dist: f32,
    pub is_from_alignment: bool
}
