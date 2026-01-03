//! Main TTS inference engine.

use crate::audio::{read_wav, SAMPLE_RATE};
use crate::config::{Config, GenerateOptions, ModelDtype};
use crate::models::{create_sessions, download_models, ModelSessions};
use crate::voices::{validate_voice_audio, VoiceCache, VoiceEmbedding};
use crate::{Error, Result};

use half::f16;
use ndarray::{concatenate, s, Array, Array2, Array3, Array4, ArrayD, Axis};
use ort::{inputs, session::SessionInputValue, value::Value};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

// Model constants
const START_SPEECH_TOKEN: i64 = 6561;
const STOP_SPEECH_TOKEN: i64 = 6562;
const SILENCE_TOKEN: i64 = 4299;
const NUM_KV_HEADS: usize = 16;
const HEAD_DIM: usize = 64;

/// Chatterbox TTS engine.
///
/// Provides text-to-speech generation with voice cloning capabilities.
pub struct ChatterboxTTS {
    sessions: ModelSessions,
    tokenizer: Tokenizer,
    voice_cache: VoiceCache,
    config: Config,
    /// Tracks which LM inputs expect f16 vs f32
    input_dtypes: HashMap<String, bool>,
}

impl ChatterboxTTS {
    /// Create a new ChatterboxTTS instance.
    ///
    /// Downloads models if not cached, initializes ONNX sessions.
    pub fn new(mut config: Config) -> Result<Self> {
        // Only force FP32 for DirectML (which has known FP16 compatibility issues)
        // CUDA should use FP16 to fit in limited VRAM
        {
            use crate::config::Device;
            if matches!(config.device, Device::DirectML(_))
                && config.dtype == crate::config::ModelDtype::Fp16
            {
                println!("Note: DirectML selected, switching to FP32 models for compatibility");
                config.dtype = crate::config::ModelDtype::Fp32;
            }
        }

        println!("Downloading models (dtype={:?})...", config.dtype);
        let paths = download_models(config.dtype)?;

        println!("Loading tokenizer...");
        let tokenizer =
            Tokenizer::from_file(&paths.tokenizer).map_err(|e| Error::Tokenizer(e.to_string()))?;

        println!("Creating ONNX sessions (lazy loading)...");
        let mut sessions = create_sessions(&paths, &config)?;

        // Inspect LM inputs to determine which need f16 (this triggers lazy load of LM)
        let mut input_dtypes = HashMap::new();
        for input in sessions.language_model()?.inputs.iter() {
            let type_str = format!("{:?}", input.input_type);
            let is_f16 = type_str.contains("Float16");
            input_dtypes.insert(input.name.clone(), is_f16);
        }

        Ok(Self {
            sessions,
            tokenizer,
            voice_cache: VoiceCache::new(),
            config,
            input_dtypes,
        })
    }

    /// Add a voice to the cache from an audio file.
    ///
    /// The audio must be at least 5 seconds long.
    pub fn add_voice(&mut self, id: impl Into<String>, audio_path: impl AsRef<Path>) -> Result<()> {
        let id = id.into();
        let (samples, sr) = read_wav(audio_path)?;

        // Validate duration
        validate_voice_audio(samples.as_slice().unwrap(), sr)?;

        // Resample to 24kHz if needed
        let samples = if sr != SAMPLE_RATE {
            let resampled =
                crate::audio::resample_linear(samples.as_slice().unwrap(), sr, SAMPLE_RATE);
            ndarray::Array1::from_vec(resampled)
        } else {
            samples
        };

        // Run speech encoder to get embeddings
        let embedding = self.encode_voice(&samples)?;
        self.voice_cache.add(id, embedding);

        Ok(())
    }

    /// Remove a voice from the cache.
    pub fn remove_voice(&mut self, id: &str) -> bool {
        self.voice_cache.remove(id)
    }

    /// List all cached voice IDs.
    pub fn list_voices(&self) -> Vec<&str> {
        self.voice_cache.list()
    }

    /// Generate speech from text using a cached voice.
    pub fn generate(
        &mut self,
        text: &str,
        voice_id: &str,
        opts: GenerateOptions,
    ) -> Result<Vec<f32>> {
        let voice = self
            .voice_cache
            .get(voice_id)
            .ok_or_else(|| Error::VoiceNotFound(voice_id.to_string()))?
            .clone();

        self.generate_with_embedding(text, &voice, opts, &mut |_| {})
    }

    /// Generate speech with streaming callback.
    ///
    /// The callback receives either token IDs or audio chunks depending on stream mode.
    pub fn generate_streaming<F>(
        &mut self,
        text: &str,
        voice_id: &str,
        opts: GenerateOptions,
        mut callback: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(StreamEvent),
    {
        let voice = self
            .voice_cache
            .get(voice_id)
            .ok_or_else(|| Error::VoiceNotFound(voice_id.to_string()))?
            .clone();

        self.generate_with_embedding(text, &voice, opts, &mut callback)
    }

    /// Encode a voice sample into embeddings.
    fn encode_voice(&mut self, samples: &ndarray::Array1<f32>) -> Result<VoiceEmbedding> {
        let audio_with_batch = samples.clone().insert_axis(Axis(0));
        let audio_val = Value::from_array(audio_with_batch.into_dyn())?;

        let speech_encoder = self.sessions.speech_encoder()?;
        let outputs = speech_encoder.run(inputs!["audio_values" => audio_val])?;

        let cond_emb = extract_f32_tensor(&outputs["audio_features"], self.config.dtype)?;
        let speaker_embeddings =
            extract_f32_tensor(&outputs["speaker_embeddings"], self.config.dtype)?;
        let speaker_features = extract_f32_tensor(&outputs["speaker_features"], self.config.dtype)?;

        let (s, d) = outputs["audio_tokens"].try_extract_tensor::<i64>()?;
        let prompt_token = ArrayD::from_shape_vec(shape_to_vec(&s), d.to_vec())?;

        Ok(VoiceEmbedding {
            cond_emb,
            speaker_embeddings,
            speaker_features,
            prompt_token,
        })
    }

    /// Main generation logic.
    fn generate_with_embedding<F>(
        &mut self,
        text: &str,
        voice: &VoiceEmbedding,
        opts: GenerateOptions,
        callback: &mut F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(StreamEvent),
    {
        // Tokenize text
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let input_ids_vec: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let input_ids = Array2::from_shape_vec((1, input_ids_vec.len()), input_ids_vec)?;

        // Get text embeddings - scoped to release borrow early
        let text_embeds = {
            let input_ids_ort = Value::from_array(input_ids.clone().into_dyn())?;
            let embed_tokens = self.sessions.embed_tokens()?;
            let embed_outputs = embed_tokens.run(inputs!["input_ids" => input_ids_ort])?;
            extract_f32_tensor(&embed_outputs["inputs_embeds"], self.config.dtype)?
        };

        // Concatenate conditioning and text embeddings
        let cond_emb_3d = voice
            .cond_emb
            .view()
            .into_dimensionality::<ndarray::Ix3>()?;
        let text_embeds_3d = text_embeds.view().into_dimensionality::<ndarray::Ix3>()?;
        let mut inputs_embeds: Array3<f32> =
            concatenate(Axis(1), &[cond_emb_3d.view(), text_embeds_3d.view()])?;

        // Initialize KV cache
        let batch_size = 1;
        let mut past_key_values: HashMap<String, Array4<f32>> = HashMap::new();
        for input in self.sessions.language_model()?.inputs.iter() {
            if input.name.contains("past_key_values") {
                let cache = Array4::<f32>::zeros((batch_size, NUM_KV_HEADS, 0, HEAD_DIM));
                past_key_values.insert(input.name.clone(), cache);
            }
        }

        let mut attention_mask = Array2::<i64>::ones((batch_size, inputs_embeds.shape()[1]));
        let mut position_ids = Array::from_iter(0..inputs_embeds.shape()[1] as i64)
            .into_shape_with_order((1, inputs_embeds.shape()[1]))?;

        let mut generate_tokens = Array2::<i64>::from_elem((1, 1), START_SPEECH_TOKEN);

        let dtype = self.config.dtype;
        let input_dtypes = self.input_dtypes.clone();

        // Generation loop
        for _step in 0..opts.max_tokens {
            // Build dynamic inputs
            let mut dynamic_inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::new();

            let embeds_is_f16 = *input_dtypes.get("inputs_embeds").unwrap_or(&false);
            dynamic_inputs.push((
                "inputs_embeds".into(),
                make_tensor(&inputs_embeds.clone().into_dyn(), embeds_is_f16)?.into(),
            ));
            dynamic_inputs.push((
                "attention_mask".into(),
                Value::from_array(attention_mask.clone().into_dyn())?.into(),
            ));
            dynamic_inputs.push((
                "position_ids".into(),
                Value::from_array(position_ids.clone().into_dyn())?.into(),
            ));

            for (k, v) in &past_key_values {
                let is_f16 = *input_dtypes.get(k).unwrap_or(&false);
                dynamic_inputs.push((
                    k.clone().into(),
                    make_tensor(&v.clone().into_dyn(), is_f16)?.into(),
                ));
            }

            let lm = self.sessions.language_model()?;
            let outputs = lm.run(dynamic_inputs)?;

            // Get logits and sample
            let (s, d) = outputs["logits"].try_extract_tensor::<f32>()?;
            let logits = ArrayD::from_shape_vec(shape_to_vec(&s), d.to_vec())?;
            let last_logits = logits.slice(s![.., -1, ..]).to_owned();

            // Apply repetition penalty
            let mut penalized_logits = last_logits.clone().into_dimensionality::<ndarray::Ix2>()?;
            for &token_id in generate_tokens.iter() {
                if token_id >= 0 && (token_id as usize) < penalized_logits.shape()[1] {
                    let score = penalized_logits[[0, token_id as usize]];
                    penalized_logits[[0, token_id as usize]] = if score < 0.0 {
                        score * opts.repetition_penalty
                    } else {
                        score / opts.repetition_penalty
                    };
                }
            }

            // Greedy sampling (TODO: add temperature/top-k/top-p)
            let next_token_id = penalized_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap();

            let next_token = Array2::from_elem((1, 1), next_token_id);
            generate_tokens = concatenate(Axis(1), &[generate_tokens.view(), next_token.view()])?;

            // Emit token event
            callback(StreamEvent::Token(next_token_id));

            if next_token_id == STOP_SPEECH_TOKEN {
                break;
            }

            // Extract KV cache before dropping outputs
            let mut new_cache_values: Vec<(String, Array4<f32>)> = Vec::new();
            for (input_name, _) in &past_key_values {
                let output_name = input_name.replace("past_key_values", "present");
                if let Some(val) = outputs.get(&output_name) {
                    let is_f16 = *input_dtypes.get(input_name).unwrap_or(&false);
                    let (s_vec, data) = if is_f16 {
                        let (s, d) = val.try_extract_tensor::<f16>()?;
                        (
                            shape_to_vec(&s),
                            d.iter().map(|x| x.to_f32()).collect::<Vec<f32>>(),
                        )
                    } else {
                        let (s, d) = val.try_extract_tensor::<f32>()?;
                        (shape_to_vec(&s), d.to_vec())
                    };
                    if s_vec.len() == 4 {
                        let tensor =
                            Array4::from_shape_vec((s_vec[0], s_vec[1], s_vec[2], s_vec[3]), data)?;
                        new_cache_values.push((input_name.clone(), tensor));
                    }
                }
            }

            // Drop outputs to release borrow
            drop(outputs);

            // Apply cached values
            for (name, tensor) in new_cache_values {
                past_key_values.insert(name, tensor);
            }

            let next_token_ort = Value::from_array(next_token.clone().into_dyn())?;
            let embed_tokens = self.sessions.embed_tokens()?;
            let embed_out = embed_tokens.run(inputs!["input_ids" => next_token_ort])?;
            inputs_embeds = extract_f32_tensor(&embed_out["inputs_embeds"], dtype)?
                .into_dimensionality::<ndarray::Ix3>()?;

            let ones = Array2::<i64>::ones((batch_size, 1));
            attention_mask = concatenate(Axis(1), &[attention_mask.view(), ones.view()])?;
            position_ids = position_ids.slice(s![.., -1..]).mapv(|x| x + 1);
        }

        // Decode audio
        let len = generate_tokens.shape()[1];
        let speech_tokens = if len > 2 {
            generate_tokens.slice(s![.., 1..len - 1]).to_owned()
        } else {
            generate_tokens.slice(s![.., 1..]).to_owned()
        };

        let silence = Array2::<i64>::from_elem((1, 3), SILENCE_TOKEN);
        let prompt_token_2d = voice
            .prompt_token
            .clone()
            .into_dimensionality::<ndarray::Ix2>()?;

        let speech_input = concatenate(
            Axis(1),
            &[prompt_token_2d.view(), speech_tokens.view(), silence.view()],
        )?;

        let decoder = self.sessions.conditional_decoder()?;
        let wav_output = decoder.run(inputs![
            "speech_tokens" => Value::from_array(speech_input.into_dyn())?,
            "speaker_embeddings" => Value::from_array(voice.speaker_embeddings.clone())?,
            "speaker_features" => Value::from_array(voice.speaker_features.clone())?
        ])?;

        let (s, d) = wav_output[0].try_extract_tensor::<f32>()?;
        let wav = ArrayD::from_shape_vec(shape_to_vec(&s), d.to_vec())?;

        Ok(wav.into_raw_vec_and_offset().0)
    }

    /// Generate speech by processing text in sentence chunks.
    ///
    /// This method splits long text into sentences and generates audio for each
    /// chunk separately. This bounds memory usage regardless of text length,
    /// making it suitable for systems with limited VRAM.
    ///
    /// # Arguments
    /// * `text` - The text to synthesize
    /// * `voice_id` - ID of a previously added voice
    /// * `opts` - Generation options (max_tokens applies per-chunk)
    /// * `callback` - Called with progress events for each chunk
    ///
    /// # Returns
    /// Concatenated audio samples for all chunks
    pub fn generate_chunked<F>(
        &mut self,
        text: &str,
        voice_id: &str,
        opts: GenerateOptions,
        mut callback: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(ChunkEvent),
    {
        let voice = self
            .voice_cache
            .get(voice_id)
            .ok_or_else(|| Error::VoiceNotFound(voice_id.to_string()))?
            .clone();

        let sentences = crate::text::split_by_sentence(text);

        // Handle empty or single-sentence case
        if sentences.is_empty() {
            return Ok(vec![]);
        }

        if sentences.len() == 1 {
            callback(ChunkEvent::ChunkStarted {
                index: 0,
                total: 1,
                text: sentences[0].clone(),
            });
            let audio =
                self.generate_with_embedding(&sentences[0], &voice, opts.clone(), &mut |_| {})?;
            callback(ChunkEvent::ChunkComplete {
                index: 0,
                total: 1,
                audio: audio.clone(),
            });
            return Ok(audio);
        }

        let total = sentences.len();
        let mut all_audio = Vec::new();

        for (index, sentence) in sentences.iter().enumerate() {
            callback(ChunkEvent::ChunkStarted {
                index,
                total,
                text: sentence.clone(),
            });

            // Generate this chunk
            let chunk_audio =
                self.generate_with_embedding(sentence, &voice, opts.clone(), &mut |_| {})?;

            callback(ChunkEvent::ChunkComplete {
                index,
                total,
                audio: chunk_audio.clone(),
            });

            all_audio.extend(chunk_audio);
        }

        Ok(all_audio)
    }
}

/// Events emitted during chunked generation.
#[derive(Debug, Clone)]
pub enum ChunkEvent {
    /// A sentence chunk started processing
    ChunkStarted {
        index: usize,
        total: usize,
        text: String,
    },
    /// A sentence chunk completed
    ChunkComplete {
        index: usize,
        total: usize,
        audio: Vec<f32>,
    },
}

/// Events emitted during streaming generation.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A speech token was generated
    Token(i64),
    /// An audio chunk is ready (samples)
    AudioChunk(Vec<f32>),
}

/// Extract f32 tensor, handling f16 conversion if needed.
fn extract_f32_tensor(val: &Value, dtype: ModelDtype) -> Result<ArrayD<f32>> {
    if matches!(dtype, ModelDtype::Fp16) {
        if let Ok((s, d)) = val.try_extract_tensor::<f16>() {
            let f32_data: Vec<f32> = d.iter().map(|x| x.to_f32()).collect();
            return Ok(ArrayD::from_shape_vec(shape_to_vec(&s), f32_data)?);
        }
    }
    let (s, d) = val.try_extract_tensor::<f32>()?;
    Ok(ArrayD::from_shape_vec(shape_to_vec(&s), d.to_vec())?)
}

/// Create ORT tensor with correct dtype.
fn make_tensor(arr: &ArrayD<f32>, is_f16: bool) -> Result<Value> {
    if is_f16 {
        let arr_f16 = arr.mapv(|x| f16::from_f32(x));
        Ok(Value::from_array(arr_f16)?.into_dyn())
    } else {
        Ok(Value::from_array(arr.to_owned())?.into_dyn())
    }
}

/// Convert ORT shape to Vec<usize>.
fn shape_to_vec(shape: &ort::tensor::Shape) -> Vec<usize> {
    let dims: &[i64] = shape.as_ref();
    dims.iter().map(|&x| x as usize).collect()
}
