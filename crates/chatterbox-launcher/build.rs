//! Build script to copy ONNX Runtime libraries to the output directory.
//!
//! This ensures CUDA and other execution providers work correctly when running
//! the app from the build directory.

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    // Get the target directory
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = PathBuf::from(&out_dir);

    // Find the target directory (3 levels up from OUT_DIR)
    let target_dir = out_path
        .ancestors()
        .nth(3)
        .expect("Could not find target directory");

    // Determine profile (debug/release)
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());
    let target_bin_dir = target_dir.join(&profile);

    println!("cargo:rerun-if-changed=build.rs");

    // Platform-specific library names
    #[cfg(target_os = "windows")]
    let lib_patterns = &[
        "onnxruntime.dll",
        "onnxruntime_providers_shared.dll",
        "onnxruntime_providers_cuda.dll",
    ];

    #[cfg(target_os = "linux")]
    let lib_patterns = &[
        "libonnxruntime.so*",
        "libonnxruntime_providers_shared.so",
        "libonnxruntime_providers_cuda.so",
    ];

    #[cfg(target_os = "macos")]
    let lib_patterns = &["libonnxruntime.dylib", "libonnxruntime.*.dylib"];

    // Search locations for ONNX Runtime libraries
    let search_paths = vec![
        // Main target directory (where cargo usually puts them)
        target_bin_dir.clone(),
        // Parent target directory
        target_dir.to_path_buf(),
        // Workspace target directory
        target_dir
            .parent()
            .map(|p| p.join(&profile))
            .unwrap_or_default(),
    ];

    // Also check ORT_DYLIB_PATH environment variable
    if let Ok(ort_path) = env::var("ORT_DYLIB_PATH") {
        let ort_dir = PathBuf::from(&ort_path).parent().map(|p| p.to_path_buf());
        if let Some(dir) = ort_dir {
            copy_libs_from_dir(&dir, &target_bin_dir, lib_patterns);
        }
    }

    // Try to find and copy libs from search paths
    for search_path in &search_paths {
        if search_path.exists() {
            copy_libs_from_dir(search_path, &target_bin_dir, lib_patterns);
        }
    }

    // For dx builds, also copy to the dx output directory
    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let manifest_path = PathBuf::from(&manifest_dir);
        let workspace_root = manifest_path.parent().and_then(|p| p.parent());

        if let Some(root) = workspace_root {
            let dx_output = root
                .join("target")
                .join("dx")
                .join("chatterbox-launcher")
                .join(&profile)
                .join("windows")
                .join("app");

            if dx_output.exists() {
                for search_path in &search_paths {
                    if search_path.exists() {
                        copy_libs_from_dir(search_path, &dx_output, lib_patterns);
                    }
                }
            }
        }
    }
}

fn copy_libs_from_dir(src_dir: &PathBuf, dest_dir: &PathBuf, patterns: &[&str]) {
    if !src_dir.exists() {
        return;
    }

    if let Ok(entries) = fs::read_dir(src_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(filename) = path.file_name().and_then(|n| n.to_str()) {
                for pattern in patterns {
                    // Simple glob matching
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
