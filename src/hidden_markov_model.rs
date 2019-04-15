use crate::spectrogram::*;
use crate::numerics::*;

#[derive(Debug, Clone)]
pub struct OnlineStats {
    pub dim: usize,
    pub inserted: Vec<f32>,    
    pub means: Vec<f32>,
    pub sum_of_square: Vec<f32>,
    pub variance: Vec<f32>
}

impl OnlineStats {

    pub fn new(dim: usize, states: usize) -> OnlineStats {
        OnlineStats {
            dim, 
            inserted: vec![0.0; states],
            means: vec![0.0; states * dim],
            sum_of_square: vec![0.0; states * dim],
            variance: vec![1.0; states * dim]
        }
    }

    pub fn from_init(dim: usize, states: usize, init: Vec<f32>) -> OnlineStats {
        OnlineStats {
            dim, 
            inserted: vec![1.0; states],
            means: init,
            sum_of_square: vec![0.0; states * dim],
            variance: vec![1.0; states * dim]
        }
    }

    pub fn update(&mut self, x: &[f32], state: usize) {
        for i in 0 .. self.dim {
            let last_mu     = self.means[state * self.dim + i];
            let last_square = self.sum_of_square[state * self.dim + i];
            let next_mu     = last_mu + (x[i] - last_mu) / self.inserted[state];
            let next_square = last_square + (x[i] - last_mu) * (x[i] - next_mu);
            let variance    = next_square / self.inserted[state];
            self.means[state * self.dim + i] = next_mu;
            self.sum_of_square[state * self.dim + i] = next_square;
            self.variance[state * self.dim + i] = variance;
        }
        self.inserted[state] += 1.0;
    }

    pub fn merge(&mut self, i: usize, j: usize) {
        for d in 0 .. self.dim {
            let delta  = self.means[j * self.dim + d] -  self.means[i * self.dim + d];
            let mean   = self.means[i * self.dim + d] + delta * (self.inserted[j] / (self.inserted[j] + self.inserted[i]));
            let square = 
                self.sum_of_square[i * self.dim + d] + 
                self.sum_of_square[j * self.dim + d] + 
                f32::powf(delta, 2.0) * 
                ((self.inserted[i] * self.inserted[j]) / (self.inserted[i] + self.inserted[j]));
            let variance = square / (self.inserted[i] + self.inserted[j]);
            self.means[i * self.dim + d] = mean;
            self.sum_of_square[i * self.dim + d] = square;
            self.variance[i * self.dim + d] = variance;
        }
        self.inserted[i] +=  self.inserted[j];
    }

}


/**
 * Hidden Markov Model
 */
#[derive(Clone)]
pub struct HiddenMarkovModel {
    pub n_states: usize,
    pub trans: Vec<f32>,
    pub start: Vec<f32>,
    pub stop: Vec<f32>,
    pub states: OnlineStats,
    pub is_segmental: Vec<bool>
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
                + ll(spec.vec(0), 
                    &self.states.means[i * self.states.dim .. (i + 1) * self.states.dim],
                    &self.states.variance[i * self.states.dim .. (i + 1) * self.states.dim]
                );
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
                let obs = ll(
                    spec.vec(t), 
                    &self.states.means[i * self.states.dim .. (i + 1) * self.states.dim],
                    &self.states.variance[i * self.states.dim .. (i + 1) * self.states.dim]
                );
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
        max / len as f32
    }

}