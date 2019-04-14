use crate::alignments::*;
use crate::numerics::*;
use crate::spectrogram::*;

use std::collections::{HashMap, HashSet};

/**
 * Hidden Markov Model
 */
pub struct HiddenMarkovModel {
    pub n_states: usize,
    pub dim: usize,
    pub trans: Vec<f32>,
    pub start: Vec<f32>,
    pub stop: Vec<f32>,
    pub states: Vec<f32>,
}

/**
 * Model merging: Build HMM by merging states
 */
pub struct ModelMerging {
    pub hmm:          HiddenMarkovModel,
    pub state_map:    HashMap<(usize, usize), usize>, // (original_seq, timestep) -> state
    pub merge_parent: Vec<usize> // union find parent structure to manage merges
}

impl ModelMerging {

    pub fn shrink(&self) -> HiddenMarkovModel {
        let mut used_states = HashMap::new();
        let mut n_states = 0;
        for i in 0 .. self.merge_parent.len() {
            let state = self.find_parent(i);
            if !used_states.contains_key(&state) {
                used_states.insert(n_states, state);
                n_states += 1;
            }
        }
        let mut start  = vec![0.0; n_states];
        let mut stop   = vec![0.0; n_states];
        let mut trans  = vec![0.0; n_states * n_states];
        let mut states = vec![0.0; n_states * self.hmm.dim];
        for i in 0 .. n_states {
            let state_i = used_states[&i];
            start[i] = self.hmm.start[state_i];
            stop[i]  = self.hmm.stop[state_i];
            for j in 0 .. self.hmm.dim {
                states[i * self.hmm.dim + j] = self.hmm.states[state_i * self.hmm.dim + j];
            }
            for j in 0 .. n_states {
                let state_j = used_states[&j];
                trans[i * n_states + j] = self.hmm.trans[state_i * self.hmm.n_states + state_j];
            }
        }
        let dim = self.hmm.dim;
        HiddenMarkovModel {n_states, dim, trans, start, stop, states}
    }


    pub fn merge_all (
        &mut self, 
        paths: &[(usize, usize, Vec<AlignmentNode>)],
        slices: &[Slice],
        perc: f32,
        k: usize,
    ) {
        let operations = ModelMerging::merges_from_alignments(paths, slices, perc, k);
        let mut closed = HashSet::new();
        for op in operations {
            let state_i = self.state_map[&(op.slice_i, op.i)];
            let state_j = self.state_map[&(op.slice_j, op.j)];
            if closed.contains(&(state_i, state_j)) {
                closed.insert((state_i, state_j));
                self.merge(state_i, state_j);
            }
        }
    }

    pub fn merge(&mut self, i: usize, j: usize) {
        if i < j {
            let state_i = self.find_parent(i);
            let state_j = self.find_parent(j);
            if state_i != state_j {
                for k in 0 .. self.hmm.n_states {
                    let to = if k != state_j { k } else { state_i };
                    self.hmm.trans[state_i * self.hmm.n_states + to] += self.hmm.trans[state_j * self.hmm.n_states + to]
                }
                self.hmm.start[state_i] += self.hmm.start[state_j];
                self.hmm.stop[state_i]  += self.hmm.stop[state_j];
                for d in 0 .. self.hmm.dim {
                    self.hmm.states[state_i * self.hmm.dim + d] += self.hmm.states[state_j * self.hmm.dim + d];
                    self.hmm.states[state_i * self.hmm.dim + d] /= 2.0;
                }
                self.merge_parent[state_j] = state_i;                
            }
        } else {
            self.merge(j, i);
        }        
    }

    pub fn find_parent(&self, i: usize) -> usize {
        let mut p = i;
        while p != self.merge_parent[p] {
            p = self.merge_parent[p];
        }
        p
    }

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
        for i in 0 .. n_states {
            merge_parent.push(i);
        }
        let hmm = HiddenMarkovModel {
                n_states,
                dim,
                start,
                stop,
                trans,
                states,
        };
        ModelMerging { hmm, state_map, merge_parent }
    }

    pub fn merges_from_alignments(
        paths: &[(usize, usize, Vec<AlignmentNode>)],
        slices: &[Slice],
        perc: f32,
        k: usize,
    ) -> Vec<MergeOperation> {
        paths.iter()
            .flat_map(|(i, j, p)| {ModelMerging::merges_from_alignment(*i, *j, slices, p, perc, k)})        
            .collect()            
    }

    pub fn merges_from_alignment(
        x: usize,
        y: usize,
        slices: &[Slice],
        path: &[AlignmentNode],
        perc: f32,
        k: usize,
    ) -> Vec<MergeOperation> {
        let distances: Vec<f32> = path.iter().map(|node| node.score).collect();
        let th = percentile(&mut distances.clone(), perc);
        let mut moving_avg = vec![];
        for i in 0..distances.len() {
            let mut avg = 0.0;
            if i >= k {
                avg = mean(&distances[i - k..i]);
            }
            moving_avg.push(avg);
        }
        let slice_i = &slices[x];
        let slice_j = &slices[y];
        let mut operations = vec![];
        for (t, node) in path.iter().enumerate() {
            let (i, j) = node.cur();
            let distance = if t < k {
                distances[t]
            } else {
                moving_avg[t - k]
            };
            let dist2prev_i = if i > 0 {
                euclidean(slice_i.sequence.vec(i), slice_i.sequence.vec(i - 1))
            } else {
                std::f32::INFINITY
            };
            let dist2prev_j = if j > 0 {
                euclidean(slice_j.sequence.vec(j), slice_j.sequence.vec(j - 1))
            } else {
                std::f32::INFINITY
            };
            if dist2prev_i < th {
                operations.push(MergeOperation {
                    slice_i: x,
                    slice_j: x,
                    i: i,
                    j: i - 1,
                    dist: dist2prev_i,
                });
            }
            if dist2prev_j < th {
                operations.push(MergeOperation {
                    slice_i: y,
                    slice_j: y,
                    i: j,
                    j: j - 1,
                    dist: dist2prev_j,
                });
            }
            match node.label {
                AlignmentLabel::Match if distance < th => operations.push(MergeOperation {
                    slice_i: x,
                    slice_j: y,
                    i: i,
                    j: j,
                    dist: distance,
                }),
                _ => (),
            };
        }
        operations
    }
}

pub struct MergeOperation {
    pub slice_i: usize,
    pub slice_j: usize,
    pub i: usize,
    pub j: usize,
    pub dist: f32,
}
