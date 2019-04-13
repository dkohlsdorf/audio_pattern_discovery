use crate::alignments::*;
use crate::numerics::*;
use crate::spectrogram::*;

use std::collections::HashMap;

pub struct HiddenMarkovModel {
    pub n_states: usize,
    pub dim: usize,
    pub trans: Vec<f32>,
    pub start: Vec<f32>,
    pub stop: Vec<f32>,
    pub states: Vec<f32>,
}

impl HiddenMarkovModel {
    pub fn from_slices(slices: &[Slice]) -> (HiddenMarkovModel, HashMap<(usize, usize), usize>) {
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
        (HiddenMarkovModel {
                n_states,
                dim,
                start,
                stop,
                trans,
                states,
        }, state_map)
    }

    pub fn merges_from_alignments(
        &self,
        paths: &[(usize, usize, Vec<AlignmentNode>)],
        slices: &[Slice],
        th: f32,
        k: usize,
    ) -> Vec<MergeOperation> {
        paths.iter()
            .flat_map(|(i, j, p)| {HiddenMarkovModel::merges_from_alignment(*i, *j, slices, p, th, k)})        
            .collect()            
    }

    pub fn merges_from_alignment(
        x: usize,
        y: usize,
        slices: &[Slice],
        path: &[AlignmentNode],
        th: f32,
        k: usize,
    ) -> Vec<MergeOperation> {
        let distances: Vec<f32> = path.iter().map(|node| node.score).collect();
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
