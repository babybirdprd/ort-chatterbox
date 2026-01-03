//! Build script to copy ONNX Runtime and cuDNN libraries.
//!
//! Automatically downloads cuDNN if CUDA feature is enabled and cuDNN isn't found.

use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, Read, Write};
use std::path::PathBuf;

// cuDNN download configuration
const CUDNN_VERSION: &str = "9.17.1.4";
const CUDNN_CUDA_VERSION: &str = "cuda12";

#[cfg(target_os = "windows")]
const CUDNN_URL: &str = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.17.1.4_cuda12-archive.zip";

#[cfg(target_os = "linux")]
const CUDNN_URL: &str = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.17.1.4_cuda12-archive.tar.xz";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = PathBuf::from(&out_dir);

    // Find the target directory (3 levels up from OUT_DIR)
    let target_dir = out_path
        .ancestors()
        .nth(3)
        .expect("Could not find target directory");

    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let target_bin_dir = target_dir.join(&profile);

    // Get dx output directory for Dioxus builds
    let dx_output = get_dx_output_dir(&profile);

    // Platform-specific library names
    #[cfg(target_os = "windows")]
    let (ort_patterns, cudnn_patterns) = (
        &[
            "onnxruntime.dll",
            "onnxruntime_providers_shared.dll",
            "onnxruntime_providers_cuda.dll",
        ],
        &[
            "cudnn64_9.dll",
            "cudnn_adv64_9.dll",
            "cudnn_cnn64_9.dll",
            "cudnn_engines_precompiled64_9.dll",
            "cudnn_engines_runtime_compiled64_9.dll",
            "cudnn_graph64_9.dll",
            "cudnn_heuristic64_9.dll",
            "cudnn_ops64_9.dll",
        ],
    );

    #[cfg(target_os = "linux")]
    let (ort_patterns, cudnn_patterns) = (
        &[
            "libonnxruntime.so*",
            "libonnxruntime_providers_shared.so",
            "libonnxruntime_providers_cuda.so",
        ],
        &[
            "libcudnn.so*",
            "libcudnn_adv.so*",
            "libcudnn_cnn.so*",
            "libcudnn_ops.so*",
            "libcudnn_graph.so*",
        ],
    );

    #[cfg(target_os = "macos")]
    let (ort_patterns, cudnn_patterns): (&[&str], &[&str]) = (
        &["libonnxruntime.dylib", "libonnxruntime.*.dylib"],
        &[], // No CUDA on macOS
    );

    // Search locations for libraries
    let mut search_paths = vec![target_bin_dir.clone(), target_dir.to_path_buf()];

    if let Some(parent) = target_dir.parent() {
        search_paths.push(parent.join(&profile));
    }

    // Check ORT_DYLIB_PATH
    if let Ok(ort_path) = env::var("ORT_DYLIB_PATH") {
        if let Some(dir) = PathBuf::from(&ort_path).parent() {
            search_paths.push(dir.to_path_buf());
        }
    }

    // Add ort.pyke.io cache directory (where ort crate downloads ONNX Runtime)
    if let Some(local_app_data) = dirs::data_local_dir() {
        let ort_cache = local_app_data.join("ort.pyke.io").join("dfbin");
        #[cfg(target_os = "windows")]
        let platform = "x86_64-pc-windows-msvc";
        #[cfg(target_os = "linux")]
        let platform = "x86_64-unknown-linux-gnu";
        #[cfg(target_os = "macos")]
        let platform = "x86_64-apple-darwin";

        let platform_cache = ort_cache.join(platform);
        if platform_cache.exists() {
            // Find ALL version directories and add their lib paths
            if let Ok(entries) = fs::read_dir(&platform_cache) {
                for entry in entries.flatten() {
                    let lib_dir = entry.path().join("onnxruntime").join("lib");
                    if lib_dir.exists() {
                        // Verify the lib directory has DLLs
                        let has_dlls = fs::read_dir(&lib_dir)
                            .map(|rd| rd.flatten().any(|e| {
                                e.path().extension().map(|ext| ext == "dll" || ext == "so" || ext == "dylib").unwrap_or(false)
                            }))
                            .unwrap_or(false);
                        if has_dlls {
                            println!("cargo:warning=Found ORT cache with libs: {}", lib_dir.display());
                            search_paths.push(lib_dir);
                        }
                    }
                }
            }
        }
    }

    // Copy ONNX Runtime libs
    for search_path in &search_paths {
        copy_libs_from_dir(search_path, &target_bin_dir, ort_patterns);
        if let Some(ref dx) = dx_output {
            copy_libs_from_dir(search_path, dx, ort_patterns);
        }
    }

    // Check if cuDNN is needed and available
    #[cfg(target_os = "windows")]
    let cudnn_check_lib = "cudnn64_9.dll";
    #[cfg(target_os = "linux")]
    let cudnn_check_lib = "libcudnn.so.9";
    #[cfg(target_os = "macos")]
    let cudnn_check_lib = "";

    if !cudnn_check_lib.is_empty() {
        let cudnn_found = search_paths
            .iter()
            .any(|p| p.join(cudnn_check_lib).exists())
            || target_bin_dir.join(cudnn_check_lib).exists()
            || dx_output
                .as_ref()
                .map(|dx| dx.join(cudnn_check_lib).exists())
                .unwrap_or(false);

        if !cudnn_found {
            println!("cargo:warning=cuDNN not found, attempting to download...");

            // Download to a cache directory
            let cache_dir = get_cudnn_cache_dir();

            if let Some(cudnn_dir) = ensure_cudnn_downloaded(&cache_dir) {
                // Copy cuDNN libs to output directories
                copy_libs_from_dir(&cudnn_dir, &target_bin_dir, cudnn_patterns);
                if let Some(ref dx) = dx_output {
                    copy_libs_from_dir(&cudnn_dir, dx, cudnn_patterns);
                }
            }
        } else {
            // cuDNN found, copy it
            for search_path in &search_paths {
                copy_libs_from_dir(search_path, &target_bin_dir, cudnn_patterns);
                if let Some(ref dx) = dx_output {
                    copy_libs_from_dir(search_path, dx, cudnn_patterns);
                }
            }
        }
    }
}

fn get_dx_output_dir(profile: &str) -> Option<PathBuf> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").ok()?;
    let manifest_path = PathBuf::from(&manifest_dir);
    let parent = manifest_path.parent()?;
    let workspace_root = parent.parent()?;

    let dx_output = workspace_root
        .join("target")
        .join("dx")
        .join("chatterbox-launcher")
        .join(profile)
        .join(if cfg!(windows) { "windows" } else { "linux" })
        .join("app");

    // Create the directory if it doesn't exist (important after cargo clean)
    if !dx_output.exists() {
        if let Err(e) = fs::create_dir_all(&dx_output) {
            println!("cargo:warning=Failed to create dx output dir: {}", e);
            return None;
        }
        println!(
            "cargo:warning=Created dx output dir: {}",
            dx_output.display()
        );
    }

    Some(dx_output)
}

fn get_cudnn_cache_dir() -> PathBuf {
    // Use a shared cache directory
    let cache_base = dirs::cache_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."));

    cache_base.join("chatterbox").join("cudnn")
}

fn ensure_cudnn_downloaded(cache_dir: &PathBuf) -> Option<PathBuf> {
    let version_dir = cache_dir.join(CUDNN_VERSION);

    #[cfg(target_os = "windows")]
    let bin_dir = version_dir.join("bin");
    #[cfg(not(target_os = "windows"))]
    let bin_dir = version_dir.join("lib");

    // Check if already downloaded
    if bin_dir.exists() {
        let entries = fs::read_dir(&bin_dir).ok()?;
        if entries.count() > 0 {
            println!(
                "cargo:warning=Using cached cuDNN from {}",
                bin_dir.display()
            );
            return Some(bin_dir);
        }
    }

    // Create cache directory
    if let Err(e) = fs::create_dir_all(&version_dir) {
        println!("cargo:warning=Failed to create cuDNN cache dir: {}", e);
        return None;
    }

    // Download cuDNN
    println!(
        "cargo:warning=Downloading cuDNN {} (this may take a while)...",
        CUDNN_VERSION
    );

    #[cfg(target_os = "windows")]
    {
        let zip_path = version_dir.join("cudnn.zip");

        if !zip_path.exists() {
            if let Err(e) = download_file(CUDNN_URL, &zip_path) {
                println!("cargo:warning=Failed to download cuDNN: {}", e);
                return None;
            }
        }

        // Extract
        println!("cargo:warning=Extracting cuDNN...");
        if let Err(e) = extract_zip(&zip_path, &version_dir) {
            println!("cargo:warning=Failed to extract cuDNN: {}", e);
            return None;
        }

        // Find the bin directory inside the extracted archive
        // The archive contains a folder like "cudnn-windows-x86_64-9.17.1.4_cuda12-archive"
        if let Ok(entries) = fs::read_dir(&version_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir()
                    && path
                        .file_name()
                        .map(|n| n.to_string_lossy().starts_with("cudnn-"))
                        .unwrap_or(false)
                {
                    let extracted_bin = path.join("bin");
                    if extracted_bin.exists() {
                        // Move/copy to our bin_dir location
                        if !bin_dir.exists() {
                            let _ = fs::create_dir_all(&bin_dir);
                        }
                        if let Ok(files) = fs::read_dir(&extracted_bin) {
                            for file in files.flatten() {
                                let src = file.path();
                                let dst = bin_dir.join(file.file_name());
                                let _ = fs::copy(&src, &dst);
                            }
                        }
                        break;
                    }
                }
            }
        }

        if bin_dir.exists() {
            println!("cargo:warning=cuDNN extracted to {}", bin_dir.display());
            Some(bin_dir)
        } else {
            println!("cargo:warning=cuDNN extraction failed - bin dir not found");
            None
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        let tar_path = version_dir.join("cudnn.tar.xz");

        if !tar_path.exists() {
            if let Err(e) = download_file(CUDNN_URL, &tar_path) {
                println!("cargo:warning=Failed to download cuDNN: {}", e);
                return None;
            }
        }

        // Extract tar.xz
        println!("cargo:warning=Extracting cuDNN...");
        if let Err(e) = extract_tar_xz(&tar_path, &version_dir) {
            println!("cargo:warning=Failed to extract cuDNN: {}", e);
            return None;
        }

        // Find the lib directory inside the extracted archive
        if let Ok(entries) = fs::read_dir(&version_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir()
                    && path
                        .file_name()
                        .map(|n| n.to_string_lossy().starts_with("cudnn-"))
                        .unwrap_or(false)
                {
                    let extracted_lib = path.join("lib");
                    if extracted_lib.exists() {
                        if !bin_dir.exists() {
                            let _ = fs::create_dir_all(&bin_dir);
                        }
                        if let Ok(files) = fs::read_dir(&extracted_lib) {
                            for file in files.flatten() {
                                let src = file.path();
                                let dst = bin_dir.join(file.file_name());
                                let _ = fs::copy(&src, &dst);
                            }
                        }
                        break;
                    }
                }
            }
        }

        if bin_dir.exists() {
            println!("cargo:warning=cuDNN extracted to {}", bin_dir.display());
            Some(bin_dir)
        } else {
            println!("cargo:warning=cuDNN extraction failed - lib dir not found");
            None
        }
    }
}

#[cfg(target_os = "linux")]
fn extract_tar_xz(
    tar_path: &PathBuf,
    dest_dir: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use tar::Archive;
    use xz2::read::XzDecoder;

    let file = File::open(tar_path)?;
    let decoder = XzDecoder::new(file);
    let mut archive = Archive::new(decoder);
    archive.unpack(dest_dir)?;

    Ok(())
}

fn download_file(url: &str, dest: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    const MAX_RETRIES: u32 = 3;

    for attempt in 1..=MAX_RETRIES {
        println!(
            "cargo:warning=Download attempt {}/{} for {}",
            attempt, MAX_RETRIES, url
        );

        // Delete any existing incomplete file
        if dest.exists() {
            let _ = fs::remove_file(dest);
        }

        match download_file_once(url, dest) {
            Ok(()) => {
                // Verify the file exists and has content
                if let Ok(metadata) = fs::metadata(dest) {
                    if metadata.len() > 1000 {
                        println!("cargo:warning=Download verified: {} bytes", metadata.len());
                        return Ok(());
                    } else {
                        println!(
                            "cargo:warning=Downloaded file too small ({} bytes), retrying...",
                            metadata.len()
                        );
                        let _ = fs::remove_file(dest);
                    }
                } else {
                    println!("cargo:warning=Downloaded file not found, retrying...");
                }
            }
            Err(e) => {
                println!("cargo:warning=Download attempt {} failed: {}", attempt, e);
                let _ = fs::remove_file(dest);
                if attempt < MAX_RETRIES {
                    println!("cargo:warning=Retrying in 2 seconds...");
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
    }

    Err(format!("Failed to download {} after {} attempts", url, MAX_RETRIES).into())
}

fn download_file_once(url: &str, dest: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let response = ureq::get(url)
        .timeout(std::time::Duration::from_secs(600)) // 10 min timeout for large files
        .call()?;

    let total_size = response
        .header("content-length")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);

    if total_size == 0 {
        return Err("Server didn't provide content-length".into());
    }

    println!(
        "cargo:warning=Downloading {} MB...",
        total_size / 1024 / 1024
    );

    let mut file = File::create(dest)?;
    let mut reader = response.into_reader();
    let mut buffer = [0u8; 65536]; // 64KB buffer for faster downloads
    let mut downloaded = 0u64;
    let mut last_progress = 0u64;

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        file.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;

        // Progress every 25MB
        if downloaded - last_progress >= 25 * 1024 * 1024 {
            last_progress = downloaded;
            println!(
                "cargo:warning=Progress: {} / {} MB ({:.0}%)",
                downloaded / 1024 / 1024,
                total_size / 1024 / 1024,
                (downloaded as f64 / total_size as f64) * 100.0
            );
        }
    }

    file.sync_all()?; // Ensure all data is written to disk

    // Verify download completed
    if downloaded != total_size {
        return Err(format!(
            "Incomplete download: got {} bytes, expected {} bytes",
            downloaded, total_size
        )
        .into());
    }

    println!(
        "cargo:warning=Download complete: {} MB",
        downloaded / 1024 / 1024
    );
    Ok(())
}

#[cfg(target_os = "windows")]
fn extract_zip(zip_path: &PathBuf, dest_dir: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(zip_path)?;
    let reader = BufReader::new(file);
    let mut archive = zip::ZipArchive::new(reader)?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = dest_dir.join(file.name());

        if file.is_dir() {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }

    Ok(())
}

fn copy_libs_from_dir(src_dir: &PathBuf, dest_dir: &PathBuf, patterns: &[&str]) {
    if !src_dir.exists() || !dest_dir.exists() {
        return;
    }

    if let Ok(entries) = fs::read_dir(src_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                for pattern in patterns {
                    let matches = if pattern.contains('*') {
                        let prefix = pattern.split('*').next().unwrap_or("");
                        filename.starts_with(prefix)
                    } else {
                        filename == *pattern
                    };

                    if matches && path.is_file() {
                        let dest_path = dest_dir.join(filename);
                        if !dest_path.exists() {
                            if let Err(e) = fs::copy(&path, &dest_path) {
                                println!("cargo:warning=Failed to copy {}: {}", filename, e);
                            } else {
                                println!(
                                    "cargo:warning=Copied {} to {}",
                                    filename,
                                    dest_dir.display()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
