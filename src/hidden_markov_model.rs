use crate::spectrogram::*;
use crate::numerics::*;
/**
 * Hidden Markov Model
 */
#[derive(Clone)]
pub struct HiddenMarkovModel {
    pub n_states: usize,
    pub dim: usize,
    pub trans: Vec<f32>,
    pub start: Vec<f32>,
    pub stop: Vec<f32>,
    pub states: Vec<f32>,
}

impl HiddenMarkovModel {

    /**
     * Normalize the transitions
     */
    pub fn normalize_transitions(&mut self) {
        let mut start_scaler = 0.0;
        let mut stop_scaler = 0.0;
        for i in 0 .. self.n_states {
            start_scaler += self.start[i];
            stop_scaler  += self.stop[i];
            let mut scaler = 0.0;
            for j in 0 .. self.n_states {
                scaler += self.trans[i * self.n_states + j];
            }
            for j in 0 .. self.n_states {
                self.trans[i * self.n_states + j] /= scaler;
            }
        }
        for i in 0 .. self.n_states {
            self.start[i] /= start_scaler;
            self.stop[i]  /= stop_scaler;
        }
    }

    /**
     * Viterbi score
     */
    pub fn viterbi(&self, spec: &NDSequence) -> f32 {
        let n_states = self.n_states;
        let len = spec.len();
        let mut vite = vec![0.0; len * n_states];
        for i in 0..n_states {
            vite[i] = f32::ln(self.start[i])
                + ll(spec.vec(0), &self.states[i * self.dim .. (i + 1) * self.dim]);
        }
        for t in 1..len {
            for i in 0..n_states {
                let mut max = std::f32::NEG_INFINITY;
                for j in 0..n_states {
                    let ll = vite[(t - 1) * n_states + j] + f32::ln(self.trans[j * self.n_states + i]);
                    if ll > max {
                        max = ll;
                    }
                }
                let obs = ll(spec.vec(t), &self.states[i * self.dim .. (i + 1) * self.dim]);
                vite[t * n_states + i] = max + obs;
            }
        }
        let mut max = std::f32::NEG_INFINITY;
        for i in 0..n_states {
            vite[(len - 1) * n_states + i] += f32::ln(self.stop[i]);
            if vite[(len - 1) * n_states + i] > max {
                max = vite[(len - 1) * n_states + i];
            }
        }
        max / (n_states * len) as f32
    }

}