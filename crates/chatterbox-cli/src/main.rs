//! Chatterbox CLI - Command-line interface for TTS generation.

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use chatterbox_core::{
    audio::{write_wav, SAMPLE_RATE},
    ChatterboxTTS, Config, Device, GenerateOptions, ModelDtype,
};

#[derive(Parser)]
#[command(name = "chatterbox")]
#[command(about = "Chatterbox TTS - Zero-shot voice cloning text-to-speech")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate speech from text
    Generate {
        /// Text to synthesize
        #[arg(short, long)]
        text: String,

        /// Path to voice reference audio (WAV, 5+ seconds)
        #[arg(short, long)]
        voice: PathBuf,

        /// Output WAV file path
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,

        /// Model precision: fp32, fp16, q8
        #[arg(long, default_value = "fp16")]
        dtype: String,

        /// Use CPU instead of CUDA
        #[arg(long)]
        cpu: bool,

        /// Temperature for sampling (higher = more random)
        #[arg(long, default_value = "0.8")]
        temperature: f32,

        /// Repetition penalty
        #[arg(long, default_value = "1.2")]
        repetition_penalty: f32,

        /// Maximum tokens to generate
        #[arg(long, default_value = "1024")]
        max_tokens: usize,
    },

    /// Manage voice cache
    Voices {
        #[command(subcommand)]
        action: VoiceAction,
    },

    /// Process multiple texts from a JSON file
    Batch {
        /// Input JSON file with batch items
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory for generated WAV files
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Model precision: fp32, fp16, q8
        #[arg(long, default_value = "fp16")]
        dtype: String,

        /// Use CPU instead of CUDA
        #[arg(long)]
        cpu: bool,
    },
}

#[derive(Subcommand)]
enum VoiceAction {
    /// List cached voices
    List,
    /// Add a voice to cache
    Add {
        /// Voice ID
        id: String,
        /// Path to WAV file
        path: PathBuf,
    },
    /// Remove a voice from cache
    Remove {
        /// Voice ID
        id: String,
    },
}

fn parse_dtype(s: &str) -> ModelDtype {
    match s.to_lowercase().as_str() {
        "fp32" => ModelDtype::Fp32,
        "fp16" => ModelDtype::Fp16,
        "q8" => ModelDtype::Q8,
        _ => {
            eprintln!("Unknown dtype '{}', using fp16", s);
            ModelDtype::Fp16
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            text,
            voice,
            output,
            dtype,
            cpu,
            temperature,
            repetition_penalty,
            max_tokens,
        } => {
            println!("🎤 Chatterbox TTS - Generate");
            println!("  Text: \"{}\"", text);
            println!("  Voice: {}", voice.display());
            println!("  Output: {}", output.display());

            let device = if cpu { Device::Cpu } else { Device::default() };

            let config = Config::builder()
                .device(device)
                .dtype(parse_dtype(&dtype))
                .build();

            let mut tts = ChatterboxTTS::new(config)?;

            println!("📎 Loading voice...");
            tts.add_voice("voice", &voice)?;

            println!("🔊 Generating speech...");
            let opts = GenerateOptions {
                temperature,
                repetition_penalty,
                max_tokens,
                ..Default::default()
            };

            let samples = tts.generate(&text, "voice", opts)?;

            println!("💾 Saving to {}...", output.display());
            write_wav(&output, &samples, SAMPLE_RATE)?;

            println!(
                "✅ Done! Generated {:.2}s of audio",
                samples.len() as f32 / SAMPLE_RATE as f32
            );
        }

        Commands::Voices { action } => {
            match action {
                VoiceAction::List => {
                    println!("Voice caching is per-session. Use 'generate' with --voice to specify audio.");
                }
                VoiceAction::Add { id, path } => {
                    println!("Voice '{}' would be added from: {}", id, path.display());
                    println!("Note: Voice caching is per-session. Use the server for persistent caching.");
                }
                VoiceAction::Remove { id } => {
                    println!("Voice '{}' would be removed.", id);
                    println!("Note: Voice caching is per-session. Use the server for persistent caching.");
                }
            }
        }

        Commands::Batch {
            input,
            output_dir,
            dtype,
            cpu,
        } => {
            println!("📦 Batch processing from: {}", input.display());
            println!("📂 Output directory: {}", output_dir.display());

            // Create output directory
            std::fs::create_dir_all(&output_dir)?;

            let device = if cpu { Device::Cpu } else { Device::default() };

            let config = Config::builder()
                .device(device)
                .dtype(parse_dtype(&dtype))
                .build();

            // Read batch file
            let content = std::fs::read_to_string(&input)?;
            let items: Vec<BatchItem> = serde_json::from_str(&content)?;

            println!("Processing {} items...", items.len());

            let mut tts = ChatterboxTTS::new(config)?;

            for (i, item) in items.iter().enumerate() {
                println!(
                    "[{}/{}] Processing: {}",
                    i + 1,
                    items.len(),
                    item.text.chars().take(50).collect::<String>()
                );

                // Add voice if not cached
                if !tts.list_voices().contains(&item.voice_id.as_str()) {
                    tts.add_voice(&item.voice_id, &item.voice_path)?;
                }

                let samples =
                    tts.generate(&item.text, &item.voice_id, GenerateOptions::default())?;

                let output_path = output_dir.join(&item.output_file);
                write_wav(&output_path, &samples, SAMPLE_RATE)?;
            }

            println!("✅ Batch complete!");
        }
    }

    Ok(())
}

#[derive(serde::Deserialize)]
struct BatchItem {
    text: String,
    voice_id: String,
    voice_path: PathBuf,
    output_file: String,
}
