use anyhow::Result;
use half::f16;
use hf_hub::{api::sync::Api, Repo, RepoType};
use hound;
use ndarray::{concatenate, s, Array, Array1, Array2, Array4, ArrayD, Axis};
use ort::{
    execution_providers::CUDAExecutionProvider,
    inputs,
    session::{Session, SessionInputValue},
    value::Value,
};
use std::collections::HashMap;
use std::path::PathBuf;
use tokenizers::Tokenizer; // Added import

// --- CONSTANTS ---
const MODEL_REPO: &str = "ResembleAI/chatterbox-turbo-ONNX";
const SAMPLE_RATE: u32 = 24000;
const START_SPEECH_TOKEN: i64 = 6561;
const STOP_SPEECH_TOKEN: i64 = 6562;
const SILENCE_TOKEN: i64 = 4299;
const NUM_KV_HEADS: usize = 16;
const HEAD_DIM: usize = 64;
const MODEL_DTYPE: &str = "fp16";

pub fn run_inference() -> Result<()> {
    // Initialize ORT with CUDA
    ort::init()
        .with_name("chatterbox")
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    println!("--- Chatterbox Turbo Rust Inference (ORT + CUDA) ---");
    println!("Target DType: {}", MODEL_DTYPE);

    // 1. Download/Locate Models
    let model_paths = download_models(MODEL_DTYPE)?;

    // 2. Load Tokenizer
    println!("Initializing API for tokenizer...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new(MODEL_REPO.to_string(), RepoType::Model));
    println!("Fetching tokenizer.json...");
    let tokenizer_path = repo.get("tokenizer.json")?;
    println!("Tokenizer fetched at {:?}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow::anyhow!(e))?;

    // 3. Create Sessions
    let mut speech_encoder = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(&model_paths.speech_encoder)?;
    let mut embed_tokens = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(&model_paths.embed_tokens)?;
    let mut language_model = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(&model_paths.language_model)?;
    let mut cond_decoder = Session::builder()?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .commit_from_file(&model_paths.conditional_decoder)?;

    // --- INPUT DATA ---
    let text = "Oh, that's hilarious! [chuckle] Um anyway, how are you doing today?";
    let ref_audio_path = if std::path::Path::new("voice_input.wav").exists() {
        "voice_input.wav"
    } else {
        "reference.wav"
    };

    if !std::path::Path::new(ref_audio_path).exists() {
        anyhow::bail!("Input file '{}' not found.", ref_audio_path);
    }

    println!("Processing: '{}' using '{}'...", text, ref_audio_path);

    // 4. Prepare Audio Input
    let (audio_values, sr) = read_wav_as_tensor(ref_audio_path)?;
    if sr != SAMPLE_RATE {
        println!(
            "Warning: Input samplerate {} != {}. Result may be bad.",
            sr, SAMPLE_RATE
        );
    }

    // 5. Prepare Text Input
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let input_ids_vec: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let input_ids = Array2::from_shape_vec((1, input_ids_vec.len()), input_ids_vec)?;

    // 6. Run Speech Encoder
    // Explicitly create Values to avoid inputs! macro issues with Views
    // Convert audio to fp16 if using fp16 model
    let audio_with_batch = audio_values.insert_axis(Axis(0));
    let audio_val_ort = Value::from_array(audio_with_batch.into_dyn())?;

    let outputs = speech_encoder.run(inputs!["audio_values" => audio_val_ort])?;

    // Helper to extract fp16 tensor and convert to f32 ArrayD
    let extract_f32_tensor = |val: &Value, name: &str| -> Result<ArrayD<f32>> {
        // In fp16 mode, model outputs are f16, try that first
        if MODEL_DTYPE == "fp16" {
            if let Ok((s, d)) = val.try_extract_tensor::<f16>() {
                let f32_data: Vec<f32> = d.iter().map(|x| x.to_f32()).collect();
                return Ok(ArrayD::from_shape_vec(shape_to_vec(&s), f32_data)?);
            }
        }
        // Fallback to f32
        let (s, d) = val
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract {} as f32: {}", name, e))?;
        Ok(ArrayD::from_shape_vec(shape_to_vec(&s), d.to_vec())?)
    };

    let cond_emb = extract_f32_tensor(&outputs["audio_features"], "audio_features")?;

    let (s, d) = outputs["audio_tokens"].try_extract_tensor::<i64>()?;
    let prompt_token = ArrayD::from_shape_vec(shape_to_vec(&s), d.to_vec())?;

    let speaker_embeddings =
        extract_f32_tensor(&outputs["speaker_embeddings"], "speaker_embeddings")?;
    let speaker_features = extract_f32_tensor(&outputs["speaker_features"], "speaker_features")?;

    // Helper to extract embeddings (works for both f16 and f32 models)
    let extract_embeds_tensor = |val: &Value| -> Result<ArrayD<f32>> {
        if MODEL_DTYPE == "fp16" {
            if let Ok((s, d)) = val.try_extract_tensor::<f16>() {
                let f32_data: Vec<f32> = d.iter().map(|x| x.to_f32()).collect();
                return Ok(ArrayD::from_shape_vec(shape_to_vec(&s), f32_data)?);
            }
        }
        let (s, d) = val.try_extract_tensor::<f32>()?;
        Ok(ArrayD::from_shape_vec(shape_to_vec(&s), d.to_vec())?)
    };

    // 7. Initial Embeddings
    let text_embeds = {
        let input_ids_ort = Value::from_array(input_ids.to_owned().into_dyn())?;
        let outputs = embed_tokens.run(inputs!["input_ids" => input_ids_ort])?;
        extract_embeds_tensor(&outputs["inputs_embeds"])?
    };

    // Concatenate cond_emb and inputs_embeds along axis 1
    let cond_emb_3d = cond_emb.view().into_dimensionality::<ndarray::Ix3>()?;
    let text_embeds_3d = text_embeds.view().into_dimensionality::<ndarray::Ix3>()?;
    let mut inputs_embeds = concatenate(Axis(1), &[cond_emb_3d.view(), text_embeds_3d.view()])?;

    // 8. Initialize Cache
    let batch_size = 1;
    let mut past_key_values: HashMap<String, Array4<f32>> = HashMap::new();
    for input in language_model.inputs.iter() {
        if input.name.contains("past_key_values") {
            let cache = Array4::<f32>::zeros((batch_size, NUM_KV_HEADS, 0, HEAD_DIM));
            past_key_values.insert(input.name.clone(), cache);
        }
    }

    let mut attention_mask = Array2::<i64>::ones((batch_size, inputs_embeds.shape()[1]));
    let mut position_ids = Array::from_iter(0..inputs_embeds.shape()[1] as i64)
        .into_shape_with_order((1, inputs_embeds.shape()[1]))?
        .mapv(|x| x as i64);

    let mut generate_tokens = Array2::<i64>::from_elem((1, 1), START_SPEECH_TOKEN);

    // Inspect and store expected input types for dynamic dtype handling
    let mut input_is_f16: HashMap<String, bool> = HashMap::new();
    for input in language_model.inputs.iter() {
        let type_str = format!("{:?}", input.input_type);
        let is_f16 = type_str.contains("Float16");
        input_is_f16.insert(input.name.clone(), is_f16);
    }

    println!("Starting generation loop...");
    let max_new_tokens = 1024;

    // Helper for tensor creation with dynamic dtype
    let make_tensor = |arr: &ArrayD<f32>, is_f16: bool| -> Result<Value> {
        if is_f16 {
            let arr_f16 = arr.mapv(|x| f16::from_f32(x));
            Ok(Value::from_array(arr_f16)?.into_dyn())
        } else {
            Ok(Value::from_array(arr.to_owned())?.into_dyn())
        }
    };

    for _i in 0..max_new_tokens {
        // Use Vec<(Cow, SessionInputValue)> to handle mixed types
        let mut dynamic_inputs: Vec<(std::borrow::Cow<'_, str>, SessionInputValue<'_>)> =
            Vec::new();

        // inputs_embeds - check model metadata for expected dtype
        let embeds_is_f16 = *input_is_f16.get("inputs_embeds").unwrap_or(&false);
        dynamic_inputs.push((
            "inputs_embeds".into(),
            make_tensor(&inputs_embeds.clone().into_dyn(), embeds_is_f16)?.into(),
        ));
        dynamic_inputs.push((
            "attention_mask".into(),
            Value::from_array(attention_mask.to_owned().into_dyn())?.into(),
        ));
        dynamic_inputs.push((
            "position_ids".into(),
            Value::from_array(position_ids.to_owned().into_dyn())?.into(),
        ));

        // past_key_values - check model metadata for expected dtype
        for (k, v) in &past_key_values {
            let is_f16 = *input_is_f16.get(k).unwrap_or(&false);
            dynamic_inputs.push((
                k.clone().into(),
                make_tensor(&v.clone().into_dyn(), is_f16)?.into(),
            ));
        }

        let outputs = language_model.run(dynamic_inputs)?;

        // Extract logits - always f32 (model outputs f32 even with fp16 internals)
        let (s, d) = outputs["logits"].try_extract_tensor::<f32>()?;
        let logits = ArrayD::from_shape_vec(shape_to_vec(&s), d.to_vec())?;

        let last_logits = logits.slice(s![.., -1, ..]).to_owned();

        // Apply repetition penalty to discourage repeating already-generated tokens
        const REPETITION_PENALTY: f32 = 1.2;
        let mut penalized_logits = last_logits.clone().into_dimensionality::<ndarray::Ix2>()?;
        for &token_id in generate_tokens.iter() {
            if token_id >= 0 && (token_id as usize) < penalized_logits.shape()[1] {
                let score = penalized_logits[[0, token_id as usize]];
                // Apply penalty: if score < 0, multiply by penalty; if score > 0, divide by penalty
                penalized_logits[[0, token_id as usize]] = if score < 0.0 {
                    score * REPETITION_PENALTY
                } else {
                    score / REPETITION_PENALTY
                };
            }
        }

        let next_token_id = penalized_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as i64)
            .unwrap();

        let next_token = Array2::from_elem((1, 1), next_token_id);
        generate_tokens = concatenate(Axis(1), &[generate_tokens.view(), next_token.view()])?;

        if next_token_id == STOP_SPEECH_TOKEN {
            break;
        }

        // --- Prepare for Next Step ---
        let next_token_ort = Value::from_array(next_token.to_owned().into_dyn())?;
        let outputs_embed = embed_tokens.run(inputs!["input_ids" => next_token_ort])?;
        inputs_embeds = extract_embeds_tensor(&outputs_embed["inputs_embeds"])?
            .into_dimensionality::<ndarray::Ix3>()?;

        let ones = Array2::<i64>::ones((batch_size, 1));
        attention_mask = concatenate(Axis(1), &[attention_mask.view(), ones.view()])?;

        let binding = position_ids
            .slice(s![.., -1])
            .mapv(|x| x + 1)
            .insert_axis(Axis(1));
        position_ids = binding.to_owned();

        for (input_name, cache_tensor) in past_key_values.iter_mut() {
            let output_name = input_name.replace("past_key_values", "present");
            if let Some(val) = outputs.get(&output_name) {
                // Extract present state - f16 for fp16 models (matches past_key_values input dtype)
                let is_f16 = *input_is_f16.get(input_name).unwrap_or(&false);
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
                // We know cache is 4D
                if s_vec.len() == 4 {
                    let tensor =
                        Array4::from_shape_vec((s_vec[0], s_vec[1], s_vec[2], s_vec[3]), data)?;
                    *cache_tensor = tensor;
                }
            }
        }
    }

    println!("Generation Complete. Decoding audio...");
    let len = generate_tokens.shape()[1];
    let speech_tokens = if len > 2 {
        generate_tokens.slice(s![.., 1..len - 1]).to_owned()
    } else {
        generate_tokens.slice(s![.., 1..]).to_owned()
    };

    let silence = Array2::<i64>::from_elem((1, 3), SILENCE_TOKEN);
    let prompt_token_2d = prompt_token.into_dimensionality::<ndarray::Ix2>()?;
    let speech_tokens_2d = speech_tokens.view();

    let speech_input = concatenate(
        Axis(1),
        &[prompt_token_2d.view(), speech_tokens_2d, silence.view()],
    )?;

    // cond_decoder inputs
    let wav_output = cond_decoder.run(inputs![
        "speech_tokens" => Value::from_array(speech_input.to_owned().into_dyn())?,
        "speaker_embeddings" => Value::from_array(speaker_embeddings.to_owned().into_dyn())?,
        "speaker_features" => Value::from_array(speaker_features.to_owned().into_dyn())?
    ])?;

    let (s, d) = wav_output[0].try_extract_tensor::<f32>()?;
    let wav = ArrayD::from_shape_vec(shape_to_vec(&s), d.to_vec())?;

    // Use into_raw_vec_and_offset() as per suggestion
    let wav_vec = wav.into_raw_vec_and_offset().0;

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create("output.wav", spec)?;
    for sample in wav_vec {
        writer.write_sample(sample)?;
    }

    println!("Audio saved to output.wav");
    Ok(())
}

// Helper to handle Shape (Vec<i64> or similar) to Vec<usize>
fn shape_to_vec(shape: &ort::tensor::Shape) -> Vec<usize> {
    // ort::tensor::Shape is often just a Vec<i64> type alias or struct wrapping it
    // If it allows iteration, we map to usize.
    // If it doesn't have iter(), we might need to check strict docs.
    // Error said `no method named 'as_slice'`.
    // Trying `.iter()` directly (if it derefs to slice? no).
    // ort 2.0 rc9 has `Shape` = `Vec<i64>`?
    // Error message: `&ort::tensor::Shape`.
    // If it is a Vec<i64>, `shape.iter()` works.
    // The previous error `s.as_slice()` suggests `s` is `&Shape`.
    // Let's assume generic iteration works, or `to_vec()`.
    // Actually, `Shape` might be a distinct type in 2.0.
    // Let's try `shape.iter()`
    // If that fails, `shape` might essentially be `&[i64]`.
    // Let's assume it implements `AsRef<[i64]>`
    let dims: &[i64] = shape.as_ref();
    dims.iter().map(|&x| x as usize).collect()
}

fn read_wav_as_tensor(path: &str) -> Result<(Array1<f32>, u32)> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = reader
        .into_samples::<i16>()
        .map(|x| x.map(|s| s as f32 / 32768.0))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((Array1::from_vec(samples), spec.sample_rate))
}

struct ModelPaths {
    speech_encoder: PathBuf,
    embed_tokens: PathBuf,
    language_model: PathBuf,
    conditional_decoder: PathBuf,
}

fn download_models(dtype: &str) -> Result<ModelPaths> {
    let api = Api::new()?;
    let repo = api.repo(Repo::new(MODEL_REPO.to_string(), RepoType::Model));

    println!("Checking/Downloading models (dtype={}) from HF...", dtype);
    let get_model = |name: &str| -> Result<PathBuf> {
        let suffix = match dtype {
            "fp32" => "".to_string(),
            "q8" => "_quantized".to_string(),
            val => format!("_{}", val),
        };
        let filename = format!("{}{}.onnx", name, suffix);
        println!("Fetching {}...", filename);
        let model_path = repo.get(&format!("onnx/{}", filename))?;

        let data_filename = format!("onnx/{}_data", filename);
        if let Ok(_) = repo.get(&data_filename) {
            println!("Fetched weights data: {}", data_filename);
        }
        Ok(model_path)
    };

    Ok(ModelPaths {
        conditional_decoder: get_model("conditional_decoder")?,
        speech_encoder: get_model("speech_encoder")?,
        embed_tokens: get_model("embed_tokens")?,
        language_model: get_model("language_model")?,
    })
}
