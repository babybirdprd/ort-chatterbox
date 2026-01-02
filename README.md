# Chatterbox-ORT

**Production-ready Rust TTS library with zero-shot voice cloning using ONNX Runtime.**

Built on top of [ResembleAI's Chatterbox Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) model.

## Features

- 🎤 **Zero-shot voice cloning** from 5+ second audio samples
- 🏷️ **Paralinguistic tags**: `[laugh]`, `[chuckle]`, `[sigh]`, `[cough]`, etc.
- 🚀 **GPU acceleration** via CUDA (feature-gated)
- 📦 **Multiple precision levels**: FP32, FP16, Q8
- 🔊 **Streaming generation** with token callbacks
- 💾 **Voice caching** for efficient repeated inference
- 🌐 **HTTP Server** with REST API

## Installation

### As a Library

```toml
[dependencies]
chatterbox-core = { git = "https://github.com/yourusername/chatterbox-ort" }
```

### Build from Source

```bash
git clone https://github.com/yourusername/chatterbox-ort.git
cd chatterbox-ort

# Build with CUDA support (default)
cargo build --release

# Build CPU-only
cargo build --release --no-default-features
```

## Quick Start

### Library Usage

```rust
use chatterbox_core::{ChatterboxTTS, Config, Device, GenerateOptions};
use chatterbox_core::audio::{write_wav, SAMPLE_RATE};

fn main() -> anyhow::Result<()> {
    // Initialize with GPU
    let config = Config::builder()
        .device(Device::Cuda(0))
        .build();
    
    let mut tts = ChatterboxTTS::new(config)?;
    
    // Add a voice (requires 5+ second WAV file)
    tts.add_voice("narrator", "path/to/voice.wav")?;
    
    // Generate speech with paralinguistic tags
    let audio = tts.generate(
        "Hello! [chuckle] That's quite funny.",
        "narrator",
        GenerateOptions::default()
    )?;
    
    // Save to file
    write_wav("output.wav", &audio, SAMPLE_RATE)?;
    
    Ok(())
}
```

### CLI Usage

```bash
# Generate speech
chatterbox generate \
    --text "Hello world! [laugh]" \
    --voice voice.wav \
    --output output.wav

# Batch processing
chatterbox batch \
    --input batch.json \
    --output-dir outputs/

# With options
chatterbox generate \
    --text "Custom settings" \
    --voice voice.wav \
    --dtype fp16 \
    --temperature 0.9 \
    --cpu  # Force CPU mode
```

### Server Usage

```bash
# Start server
chatterbox-server

# Or with environment variables
CHATTERBOX_API_KEY=your-secret \
CHATTERBOX_DTYPE=fp16 \
PORT=8080 \
chatterbox-server
```

#### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/generate` | Generate speech |
| GET | `/voices` | List cached voices |
| POST | `/voices` | Add voice (multipart) |
| DELETE | `/voices/:id` | Remove voice |

**Generate Request:**
```json
{
  "text": "Hello world! [chuckle]",
  "voice_id": "narrator",
  "temperature": 0.8,
  "max_tokens": 1024
}
```

**Add Voice (multipart/form-data):**
- `id`: Voice identifier
- `audio`: WAV file (5+ seconds)

## Paralinguistic Tags

Add expressiveness with built-in tags:

| Tag | Effect |
|-----|--------|
| `[laugh]` | Laughing |
| `[chuckle]` | Light chuckling |
| `[sigh]` | Sighing |
| `[cough]` | Coughing |
| `[clear throat]` | Throat clearing |
| `[gasp]` | Gasping |
| `[groan]` | Groaning |
| `[sniff]` | Sniffing |
| `[shush]` | Shushing |

## Configuration

### Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `temperature` | 0.8 | Sampling randomness (higher = more varied) |
| `top_k` | 1000 | Top-K sampling |
| `top_p` | 0.95 | Nucleus sampling |
| `repetition_penalty` | 1.2 | Penalty for repeating tokens |
| `max_tokens` | 1024 | Maximum tokens to generate |

### Model Precision

| Dtype | VRAM | Quality | Speed |
|-------|------|---------|-------|
| `fp32` | ~3.3GB | Best | Slower |
| `fp16` | ~1.7GB | Good | Faster |
| `q8` | ~1.1GB | Good | Fastest |

> **Note**: Q4 quantization is deferred for future release.

## Project Structure

```
chatterbox-ort/
├── Cargo.toml              # Workspace manifest
├── crates/
│   ├── chatterbox-core/    # Core library
│   ├── chatterbox-cli/     # CLI binary
│   └── chatterbox-server/  # HTTP server
└── demo/                   # Original demo code
```

## Requirements

- Rust 1.75+
- For GPU: CUDA 11.x or 12.x, cuDNN 8.x

## License

MIT

## Acknowledgements

- [ResembleAI](https://resemble.ai/) for the Chatterbox Turbo model
- [ONNX Runtime](https://onnxruntime.ai/) for inference
