use rustdct::DCTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;
use rustfft::FFTplanner;

use crate::audio::*;
use crate::neural::*;
use crate::numerics::*;

/**
 * A flat Spectrogram / Cepstrum
 */
pub struct NDSequence {
    /// number of cepstral components
    pub n_bins: usize,
    /// flat cepstrum data `[x00 ... x0D ... xT0 ... xTD]`
    pub frames: Vec<f32>,
    /// size of dft
    pub dft_win: usize,
    /// spectrogram data `[x00 ... x0D ... xT0 ... xTD]`
    pub spectrogram: Vec<f32>,
    /// id of audio file
    pub audio_id: usize,
}

impl NDSequence {
    /**
     * Build a spectrogram from raw audio given a dft window and a step size.
     * Each resulting frame is normalised to standard score.
     */
    pub fn new(
        fft_size: usize,
        fft_step: usize,
        filter_size: usize,
        raw_audio: &AudioData,
    ) -> NDSequence {
        let hamming = hamming(fft_size);
        let triag = triag(fft_size / filter_size);
        let samples: Vec<Complex<f32>> = raw_audio
            .data
            .iter()
            .map(|x| Complex::new(f32::from(*x), 0.0))
            .collect();
        let mut planner_dft = FFTplanner::new(false);
        let mut planner_dct = DCTplanner::new();
        let mut ceps: Vec<f32> = Vec::new();
        let mut spectrogram: Vec<f32> = Vec::new();
        let fft = planner_dft.plan_fft(fft_size);
        let n = samples.len();
        let mut n_bins = 0;
        for i in (fft_size..n).step_by(fft_step) {
            let start = i - fft_size;
            let stop = i;
            let mut output: Vec<Complex<f32>> = vec![Complex::zero(); fft_size];
            let mut input: Vec<Complex<f32>> = samples[start..stop]
                .iter()
                .enumerate()
                .map(|(i, x)| x * hamming[i])
                .collect();
            fft.process(&mut input[..], &mut output);
            let result: Vec<f32> = output
                .iter()
                .map(|complex| f32::sqrt(complex.norm_sqr()))
                .take(fft_size / 2)
                .collect();
            let mut convolved: Vec<f32> =
                convolve(&result[0..result.len()], &triag[..], triag.len() / 2)
                    .iter()
                    .map(|x| f32::ln(*x + 1e-6))
                    .collect();
            let mut cepstrum: Vec<f32> = vec![0f32; convolved.len()];
            let dct = planner_dct.plan_dct1(convolved.len());
            dct.process_dct1(&mut convolved, &mut cepstrum);
            let mu_ceps = mean(&cepstrum[4..cepstrum.len()]);
            let final_ceps: Vec<f32> = cepstrum.iter().skip(4).map(|c| c - mu_ceps).collect();
            n_bins = cepstrum.len() - 4;
            for c in final_ceps.iter() {
                ceps.push(*c);
            }

            let mu_spec = mean(&result[10..result.len()]);
            let std_spec = f32::max(std(&result[10..result.len()], mu_spec), 1.0);
            for result in result.iter().skip(10) {
                spectrogram.push((result - mu_spec) / std_spec);
            }
        }
        NDSequence {
            audio_id: raw_audio.id,
            n_bins,
            frames: ceps,
            dft_win: fft_size / 2 - 10,
            spectrogram,
        }
    }

    /**
     *  Return reference to vector in sequence at time t
     */
    pub fn vec(&self, t: usize) -> &[f32] {
        &self.frames[t * self.n_bins..(t + 1) * self.n_bins]
    }

    pub fn encoded(&self, nn: &AutoEncoder) -> NDSequence {
        let mut flat = vec![];
        for i in 0..self.len() {
            flat.extend(
                nn.predict(&Mat {
                    flat: self.vec(i).to_vec(),
                    cols: self.n_bins,
                })
                .flat,
            );
        }
        NDSequence {
            audio_id: self.audio_id,
            n_bins: nn.n_latent(),
            frames: flat,
            dft_win: self.dft_win,
            spectrogram: self.spectrogram.clone(),
        }
    }

    /**
     * Cepstrum as bytes of gray scale image.
     * The values are min-max normalized.
     */
    pub fn img_ceps(&self) -> Vec<u8> {
        let max = max(&self.frames[..]);
        let min = min(&self.frames[..]);
        self.frames
            .iter()
            .map(|x| ((x - min) / (max - min) * 255.0) as u8)
            .collect()
    }

    /**
     * Spectrogram as bytes of gray scale image.
     * The values are min-max normalized.
     */
    pub fn img_spec(&self) -> Vec<u8> {
        let max = max(&self.spectrogram[..]);
        let min = min(&self.spectrogram[..]);
        self.spectrogram
            .iter()
            .map(|x| ((x - min) / (max - min) * 255.0) as u8)
            .collect()
    }

    /**
     * Len of cepstrum is the length of the flat spectrogram divided by the number of bins
     */
    pub fn len(&self) -> usize {
        self.frames.len() / (self.n_bins as usize)
    }

    /**
     * Len of spectrogram is the length of the flat spectrogram divided by the number of bins
     */
    pub fn len_spec(&self) -> usize {
        self.spectrogram.len() / (self.dft_win as usize)
    }

    /**     
     *  Return value at time t and frequency f
     *  which is at position: `t * bins + f` in the flat array.
     */
    pub fn at(&self, t: usize, f: usize) -> f32 {
        self.frames[t * self.n_bins + f]
    }

    /**
     * Variance in each frame, smoothed by moving average
     **/
    pub fn variance(&self, k: usize) -> Vec<f32> {
        let mut deltas = vec![];
        for i in 0..self.len() {
            let m = mean(self.vec(i));
            let s = std(self.vec(i), m);
            deltas.push(s);
        }
        let mut moving_avg = vec![];
        for i in 0..self.len() {
            let avg = if i >= k { mean(&deltas[i - k..i]) } else { 0.0 };
            moving_avg.push(avg);
        }
        moving_avg
    }

    /**
     * Extract regions by  
     **/
    pub fn interesting_ranges(
        &self,
        moving_average: usize,
        perc: f32,
        min_len: usize,
    ) -> Vec<Slice> {
        let th = percentile(&mut self.variance(moving_average), perc);
        let variances = self.variance(moving_average);
        let mut ranges = vec![];
        let mut start = 0;
        let mut recording = true;
        for (i, variance) in variances.iter().enumerate() {
            if *variance >= th && !recording {
                start = i;
                recording = true;
            }
            if *variance < th && recording {
                recording = false;
                if i - start > min_len {
                    ranges.push(Slice::new(start, i, self));
                }
            }
        }
        ranges
    }
}

/**
 * A range in an ND Sequence
 */
#[derive(Clone, Copy)]
pub struct Slice<'a> {
    pub start: usize,
    pub stop: usize,
    pub sequence: &'a NDSequence,
}

impl<'a> Slice<'a> {
    pub fn new(start: usize, stop: usize, sequence: &'a NDSequence) -> Slice<'a> {
        Slice {
            start,
            stop,
            sequence,
        }
    }

    pub fn len(&self) -> usize {
        self.stop - self.start
    }

    /**
     * Materialise the range as a new spectrogram
     */
    pub fn extract(&self) -> NDSequence {
        let start = self.start * self.sequence.n_bins;
        let stop = self.stop * self.sequence.n_bins;
        let frames = Vec::from(&self.sequence.frames[start..stop]);
        let n_bins = self.sequence.n_bins;
        let spec_start = self.start * self.sequence.dft_win;
        let spec_stop = self.stop * self.sequence.dft_win;
        let spectrogram = Vec::from(&self.sequence.spectrogram[spec_start..spec_stop]);
        let dft_win = self.sequence.dft_win;
        let audio_id = self.sequence.audio_id;
        NDSequence {
            audio_id,
            n_bins,
            frames,
            dft_win,
            spectrogram,
        }
    }
}
