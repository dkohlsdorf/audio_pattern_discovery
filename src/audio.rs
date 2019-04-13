use hound::*;
use std::iter::FromIterator;

/**
 * Simply holds audio data from hound
 */
pub struct AudioData {
    pub id: usize,
    pub spec: WavSpec,
    pub data: Vec<i16>
}

impl AudioData {

    /**
     * Read audio data. For multiple channels, we only take the first.
     */
    pub fn from_file(file: String, id: usize) -> AudioData {
        let mut reader = WavReader::open(file).unwrap();
        let n_channels = reader.spec().channels as usize;
        let samples = reader
            .samples::<i16>()
            .enumerate()
            .filter_map(|(i, x)| {
                if i % n_channels == 0 {
                    Some(x.unwrap())
                } else {
                    None
                }
            })
            .collect();
        let mut spec = reader.spec().clone();
        spec.channels = 1;
        AudioData {
            id,
            spec: spec,
            data: samples,
        }
    }

    /**
     * Append audio samples to this file, seperated by zeros
     */
    pub fn append(&mut self, insert_zeros: usize, audio: &mut AudioData) {
        for _i in 0 .. insert_zeros {
            self.data.push(0);
        }
        self.data.append(&mut audio.data);
    }

    /**
     * Extract a slice of audio
     */    
    pub fn slice(&self, t_start: usize, t_stop: usize) -> AudioData {
        AudioData{id: self.id, spec: self.spec.clone(), data: Vec::from_iter(self.data[t_start .. t_stop].iter().cloned())}
    }

    /**
     * Write this audio file
     */
    pub fn write(&self, file: String) {
        let mut writer = hound::WavWriter::create(file, self.spec).unwrap();
        for sample in self.data.iter() {
            writer.write_sample(*sample).unwrap();
        }
    }    

}
