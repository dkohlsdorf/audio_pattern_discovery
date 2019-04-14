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
