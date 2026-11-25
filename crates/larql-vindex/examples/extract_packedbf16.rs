//! Remote-run extraction entry point for PackedBf16 checkpoints (Gemma 4 26B-A4B).
//!
//! Thin CLI wrapper over `build_vindex_streaming_opts` — same pipeline as
//! `larql extract-index`, plus:
//! - `--consume-shards` for tight-disk hosts: each safetensors shard is
//!   DELETED once extraction has consumed every tensor it needs from it
//!   (only the exact shard files are removed; config.json, tokenizer.json and
//!   every other file survive).
//! - Two-session split (M2 step-8): `--stage-manifest <path>` + `--stop-after
//!   <stage>` run a stage prefix and persist its products + the manifest;
//!   re-running with the same `--stage-manifest` skips completed stages.
//!   `--down-threads N` splits the down_meta stage across N worker threads
//!   (byte-identical output, ~N× wall clock on the down stage).
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example extract_packedbf16 -- \
//!       <model_dir> -o <output.vindex> [--f16] [--down-top-k 10] \
//!       [--level browse|inference|all] [--consume-shards] [--model <display name>] \
//!       [--stage-manifest <path>] [--stop-after gate_vectors|router_weights|embeddings|down_meta] \
//!       [--down-threads N]
//!
//! No model downloads happen here — point it at an already-downloaded snapshot.

use std::path::PathBuf;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut model_dir: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut model_name: Option<String> = None;
    let mut down_top_k = 10usize;
    let mut level = larql_vindex::ExtractLevel::Browse;
    let mut dtype = larql_vindex::StorageDtype::F32;
    let mut consume_shards = false;
    let mut stage_control = larql_vindex::StageControl::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-o" | "--output" => output = Some(PathBuf::from(args.next().expect("-o needs a value"))),
            "--model" => model_name = Some(args.next().expect("--model needs a value")),
            "--down-top-k" => {
                down_top_k = args
                    .next()
                    .expect("--down-top-k needs a value")
                    .parse()
                    .expect("--down-top-k needs a number");
            }
            "--level" => {
                level = match args.next().expect("--level needs a value").as_str() {
                    "browse" => larql_vindex::ExtractLevel::Browse,
                    "inference" => larql_vindex::ExtractLevel::Inference,
                    "all" => larql_vindex::ExtractLevel::All,
                    other => {
                        eprintln!("unknown level: {other}");
                        std::process::exit(2);
                    }
                };
            }
            "--f16" => dtype = larql_vindex::StorageDtype::F16,
            "--consume-shards" => consume_shards = true,
            "--stage-manifest" => {
                stage_control.manifest_path =
                    Some(PathBuf::from(args.next().expect("--stage-manifest needs a value")));
            }
            "--stop-after" => {
                stage_control.stop_after =
                    Some(args.next().expect("--stop-after needs a value"));
            }
            "--down-threads" => {
                stage_control.down_threads = args
                    .next()
                    .expect("--down-threads needs a value")
                    .parse()
                    .expect("--down-threads needs a number");
            }
            other => {
                if other.starts_with('-') {
                    eprintln!("unknown flag: {other}");
                    std::process::exit(2);
                }
                model_dir = Some(PathBuf::from(other));
            }
        }
    }

    let model_dir = model_dir.expect("usage: extract_packedbf16 <model_dir> -o <out.vindex> [flags]");
    let output = output.expect("-o <output.vindex> is required");
    let model_name = model_name.unwrap_or_else(|| {
        model_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    });

    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_file(&tok_path)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", tok_path.display()));

    let opts = larql_vindex::StreamingOpts { consume_shards, stage_control };
    let start = std::time::Instant::now();
    // Run on a big-stack worker: BLAS static-init and the extraction's deep
    // call paths overflow the 1 MB MSVC main thread on Windows (Linux's 8 MB
    // main thread and cargo-test's 8 MB worker threads never hit this).
    let output_path = output.clone();
    let worker = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
        .spawn(move || {
            larql_vindex::build_vindex_streaming_opts(
                &model_dir,
                &tokenizer,
                &model_name,
                &output,
                down_top_k,
                level,
                dtype,
                &mut larql_vindex::SilentBuildCallbacks,
                &opts,
            )
            .unwrap_or_else(|e| panic!("extraction failed: {e}"));
        })
        .expect("spawn extraction worker");
    worker.join().expect("extraction worker panicked");
    eprintln!("done in {:.1}s → {}", start.elapsed().as_secs_f64(), output_path.display());
}
