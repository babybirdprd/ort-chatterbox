//! Audio I/O utilities.

use crate::{Error, Result};
use ndarray::Array1;
use std::path::Path;

/// Sample rate used by Chatterbox models.
pub const SAMPLE_RATE: u32 = 24000;

/// Read a WAV file and return samples as f32 in range [-1, 1].
///
/// Automatically converts from various bit depths (16-bit, 24-bit, 32-bit).
pub fn read_wav(path: impl AsRef<Path>) -> Result<(Array1<f32>, u32)> {
    let path = path.as_ref();
    let reader = hound::WavReader::open(path)
        .map_err(|e| Error::Audio(format!("Failed to open '{}': {}", path.display(), e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Int, 16) => reader
            .into_samples::<i16>()
            .map(|s| s.map(|v| v as f32 / 32768.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read samples: {}", e)))?,
        (hound::SampleFormat::Int, 24) => reader
            .into_samples::<i32>()
            .map(|s| s.map(|v| v as f32 / 8388608.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read samples: {}", e)))?,
        (hound::SampleFormat::Int, 32) => reader
            .into_samples::<i32>()
            .map(|s| s.map(|v| v as f32 / 2147483648.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read samples: {}", e)))?,
        (hound::SampleFormat::Float, 32) => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read samples: {}", e)))?,
        _ => {
            return Err(Error::Audio(format!(
                "Unsupported format: {:?} {}bit",
                spec.sample_format, spec.bits_per_sample
            )))
        }
    };

    // Convert stereo to mono by averaging channels
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((Array1::from_vec(samples), sample_rate))
}

/// Write samples to a WAV file.
///
/// Samples should be f32 in range [-1, 1].
pub fn write_wav(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
    let path = path.as_ref();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| Error::Audio(format!("Failed to create '{}': {}", path.display(), e)))?;

    for &sample in samples {
        writer
            .write_sample(sample)
            .map_err(|e| Error::Audio(format!("Failed to write sample: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| Error::Audio(format!("Failed to finalize WAV: {}", e)))?;

    Ok(())
}

/// Resample audio to target sample rate using linear interpolation.
///
/// For production, consider using the `rubato` crate for higher quality resampling.
pub fn resample_linear(samples: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if from_sr == to_sr {
        return samples.to_vec();
    }

    let ratio = to_sr as f64 / from_sr as f64;
    let output_len = (samples.len() as f64 * ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;
        let frac = src_pos - src_idx as f64;

        let sample = if src_idx + 1 < samples.len() {
            samples[src_idx] * (1.0 - frac as f32) + samples[src_idx + 1] * frac as f32
        } else if src_idx < samples.len() {
            samples[src_idx]
        } else {
            0.0
        };

        output.push(sample);
    }

    output
}
