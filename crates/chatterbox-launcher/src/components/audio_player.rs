//! Audio player component with real waveform and proper playback state.

use crate::state::AppState;
use dioxus::prelude::*;

#[component]
pub fn AudioPlayer(state: Signal<AppState>) -> Element {
    let is_playing = state.read().is_playing;
    let audio = state.read().output_audio.clone();

    let duration = audio
        .as_ref()
        .map(|a| a.len() as f32 / 24000.0)
        .unwrap_or(0.0);

    // Calculate waveform bars from actual audio data
    let waveform_heights: Vec<f32> = audio
        .as_ref()
        .map(|samples| {
            // Downsample to 40 bars
            let chunk_size = (samples.len() / 40).max(1);
            (0..40)
                .map(|i| {
                    let start = i * chunk_size;
                    let end = (start + chunk_size).min(samples.len());
                    if start >= samples.len() {
                        0.0
                    } else {
                        // Get RMS amplitude for this chunk
                        let rms: f32 = samples[start..end].iter().map(|s| s * s).sum::<f32>()
                            / (end - start) as f32;
                        rms.sqrt().min(1.0)
                    }
                })
                .collect()
        })
        .unwrap_or_else(|| vec![0.0; 40]);

    rsx! {
        div { class: "glass-card p-4",
            div { class: "flex items-center gap-2 mb-3",
                span { class: "text-accent-400", "♪" }
                span { class: "text-sm font-medium text-gray-300", "Audio Output" }
                span { class: "text-xs text-gray-500 ml-auto",
                    "{duration:.1}s generated"
                }
            }

            div { class: "flex items-center gap-4",
                // Play/Pause button
                button {
                    class: "w-12 h-12 rounded-full bg-accent-500 flex items-center justify-center text-white text-xl hover:bg-accent-600 transition-colors",
                    onclick: move |_| {
                        let audio = state.read().output_audio.clone();
                        if let Some(samples) = audio {
                            if !is_playing {
                                state.write().is_playing = true;

                                // Play audio and reset state when done
                                spawn(async move {
                                    play_audio(&samples).await;
                                    state.write().is_playing = false;
                                });
                            } else {
                                // Stop playback (just update state, audio will finish naturally)
                                state.write().is_playing = false;
                            }
                        }
                    },
                    if is_playing { "⏸" } else { "▶" }
                }

                // Waveform visualization - real data, animated only when playing
                div { class: "flex-1 h-12 flex items-end justify-center gap-0.5",
                    for (i, height) in waveform_heights.iter().enumerate() {
                        {
                            let bar_height = (height * 100.0).max(10.0);
                            let animation_class = if is_playing { "waveform-bar-playing" } else { "" };
                            rsx! {
                                div {
                                    key: "{i}",
                                    class: "w-1 bg-purple-500 rounded-full transition-all duration-100 {animation_class}",
                                    style: "height: {bar_height}%; animation-delay: {i * 30}ms;"
                                }
                            }
                        }
                    }
                }

                // Save button
                button {
                    class: "btn-ghost flex items-center gap-2",
                    onclick: move |_| {
                        let audio = state.read().output_audio.clone();
                        if let Some(samples) = audio {
                            spawn(async move {
                                if let Some(path) = rfd::AsyncFileDialog::new()
                                    .add_filter("WAV Audio", &["wav"])
                                    .set_file_name("output.wav")
                                    .save_file()
                                    .await
                                {
                                    let path = path.path();
                                    if let Err(e) = save_wav(&path, &samples) {
                                        eprintln!("Failed to save: {}", e);
                                    }
                                }
                            });
                        }
                    },
                    span { "💾" }
                    span { "Save" }
                }
            }
        }
    }
}

async fn play_audio(samples: &[f32]) {
    use rodio::{OutputStream, Source};
    use std::time::Duration;

    if let Ok((_stream, handle)) = OutputStream::try_default() {
        let source = rodio::buffer::SamplesBuffer::new(1, 24000, samples.to_vec());
        if handle.play_raw(source.convert_samples()).is_ok() {
            // Wait for playback to complete
            let duration = samples.len() as f32 / 24000.0;
            tokio::time::sleep(Duration::from_secs_f32(duration)).await;
        }
    }
}

fn save_wav(path: &std::path::Path, samples: &[f32]) -> anyhow::Result<()> {
    use hound::{WavSpec, WavWriter};

    let spec = WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for &sample in samples {
        let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(s)?;
    }
    writer.finalize()?;

    Ok(())
}
