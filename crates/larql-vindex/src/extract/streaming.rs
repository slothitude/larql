//! Streaming vindex extraction — build from safetensors without loading the full model.
//!
//! Instead of loading all weights into ModelWeights (which requires the entire model
//! in RAM), this module mmaps safetensors files and processes one layer at a time.
//! Peak memory = 1 layer's tensors + embeddings, not the full model.
//!
//! For a 120B MoE model: ~120 GB as ModelWeights vs ~2 GB streaming.

use std::collections::{HashMap, HashSet};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use ndarray::Array2;

use crate::config::dtype::StorageDtype;
use crate::config::{VindexConfig, VindexLayerInfo, VindexModelConfig};
use crate::error::VindexError;
use crate::extract::callbacks::IndexBuildCallbacks;

/// Mmap'd safetensors file — held open until released (consume-shards) or
/// until the end of extraction.
struct MmapShard {
    _file: std::fs::File,
    mmap: memmap2::Mmap,
}

/// One safetensors shard slot. `open: None` once the mmap has been dropped
/// (consume-shards release) — any later read is a hard error instead of a
/// silent stale-page read.
struct ShardSlot {
    path: PathBuf,
    open: Option<MmapShard>,
}

/// All source shards of a streaming extraction. The slot paths are the exact
/// downloaded shard files — `consume_shards` deletes exactly those paths and
/// nothing else (config.json / tokenizer.json / aux files are never touched).
struct ShardStore {
    slots: Vec<ShardSlot>,
}

impl ShardStore {
    fn open_all(paths: Vec<PathBuf>) -> Result<Self, VindexError> {
        let slots = paths
            .into_iter()
            .map(|path| {
                let file = std::fs::File::open(&path)?;
                // SAFETY: read-only mapping of a regular file; pages are
                // touched lazily via ranged reads (never a whole-shard copy).
                let mmap = unsafe { memmap2::Mmap::map(&file)? };
                Ok(ShardSlot {
                    path,
                    open: Some(MmapShard { _file: file, mmap }),
                })
            })
            .collect::<Result<Vec<_>, VindexError>>()?;
        Ok(ShardStore { slots })
    }

    fn bytes(&self, idx: usize) -> Result<&[u8], VindexError> {
        self.slots[idx]
            .open
            .as_ref()
            .map(|s| s.mmap.as_ref())
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "shard {} ({}) was already consumed and deleted",
                    idx + 1,
                    self.slots[idx].path.display()
                ))
            })
    }

    fn path(&self, idx: usize) -> &Path {
        &self.slots[idx].path
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    /// Drop the mmap + file handle (required before deletion on Windows), then
    /// delete the shard file. Deletion failures are reported but never abort an
    /// otherwise-successful extraction.
    fn release_and_delete(&mut self, idx: usize) {
        self.slots[idx].open = None;
        match std::fs::remove_file(&self.slots[idx].path) {
            Ok(()) => eprintln!(
                "  consume-shards: deleted {}",
                self.slots[idx].path.display()
            ),
            Err(e) => eprintln!(
                "  consume-shards: WARNING could not delete {}: {e}",
                self.slots[idx].path.display()
            ),
        }
    }
}

/// Which tensors each shard has served so far (consume-shards accounting).
#[derive(Default)]
struct Consumption {
    by_shard: Vec<HashSet<String>>,
}

impl Consumption {
    fn record(&mut self, shard_idx: usize, tensor_name: &str) {
        if self.by_shard.len() <= shard_idx {
            self.by_shard.resize(shard_idx + 1, HashSet::new());
        }
        self.by_shard[shard_idx].insert(tensor_name.to_string());
    }

    fn served(&self, shard_idx: usize) -> &HashSet<String> {
        static EMPTY: std::sync::OnceLock<HashSet<String>> = std::sync::OnceLock::new();
        self.by_shard.get(shard_idx).unwrap_or_else(|| EMPTY.get_or_init(HashSet::new))
    }
}

/// Options for `build_vindex_streaming_opts`.
#[derive(Debug, Clone, Default)]
pub struct StreamingOpts {
    /// Delete each source safetensors shard once extraction has consumed every
    /// tensor it needs from it — the disk-recovery mode for tight-disk hosts
    /// (51.6 GB checkpoint → ~17 GB vindex on a 78 GB Kaggle/Colab disk).
    ///
    /// Guards: only the exact shard files that were opened are ever removed
    /// (config.json, tokenizer.json and any other file in the model directory
    /// survive); browse-level deletions happen only after the down-meta stage,
    /// when the full browse consumption plan is known to be served; higher
    /// extract levels delete only after the model-weights stage, so the weight
    /// pass can never hit a closed shard.
    pub consume_shards: bool,
    /// Two-session split control (M2 step-8). Default = classic single-session
    /// run with unchanged behavior.
    pub stage_control: StageControl,
}

/// Stage-control for the two-session split (M2 step-8).
///
/// A session runs a prefix of the extraction pipeline, persists its outputs
/// plus a `stage-manifest.json`, and exits; the next session resumes from that
/// manifest. `StageControl::default()` keeps the classic single-session run:
/// no manifest, no skip, serial down stage — byte-identical to previous
/// versions.
#[derive(Debug, Clone, Default)]
pub struct StageControl {
    /// Path of the stage manifest (written on each stage completion and
    /// consumed on load). When `Some`, stages recorded as complete are
    /// skipped and every stage completed during this run is recorded with its
    /// output file's sha256 + byte size.
    pub manifest_path: Option<PathBuf>,
    /// Stop cleanly after this stage completes (manifest updated, open shards
    /// released under `consume_shards`, then `Ok(())`). Stage names:
    /// "gate_vectors" | "router_weights" | "embeddings" | "down_meta".
    pub stop_after: Option<String>,
    /// Worker threads for the down_meta stage. `0` = one thread (serial code
    /// path, unchanged behavior); `>1` splits each layer's feature space
    /// across N `std::thread`s. Per-feature math and output ordering are
    /// unchanged — the split is validated byte-identical against the serial
    /// path on the gemma4 fixture.
    pub down_threads: usize,
}

/// One completed stage's persistence record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageRecord {
    /// Output file name relative to the vindex output dir.
    pub output: String,
    pub bytes: u64,
    pub sha256: String,
    /// down_meta only: number of layers fully processed when this checkpoint
    /// was written (partial checkpoint — a later final write replaces it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub done_layers: Option<usize>,
}

/// Cross-session stage manifest (`stage-manifest.json`).
#[derive(Clone, Serialize, Deserialize)]
pub struct StageManifest {
    pub format: String,
    pub model_name: String,
    pub down_top_k: usize,
    pub extract_level: crate::ExtractLevel,
    pub dtype: StorageDtype,
    pub vocab_size: usize,
    pub stages: HashMap<String, StageRecord>,
    /// Gate-stage product: per-layer feature inventory, required to assemble
    /// index.json in the resuming session without re-reading the gate tensors.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub layer_infos: Vec<VindexLayerInfo>,
}

impl StageManifest {
    const FORMAT: &'static str = "larql-stage-manifest/1";
    fn load(path: &Path) -> Option<StageManifest> {
        let text = std::fs::read_to_string(path).ok()?;
        match serde_json::from_str::<StageManifest>(&text) {
            Ok(m) if m.format == Self::FORMAT => Some(m),
            Ok(m) => {
                eprintln!(
                    "  stage-manifest: ignoring {} (format {:?} != {:?})",
                    path.display(),
                    m.format,
                    Self::FORMAT
                );
                None
            }
            Err(e) => {
                eprintln!("  stage-manifest: ignoring {} (parse: {e})", path.display());
                None
            }
        }
    }

    fn stage_done(&self, name: &str) -> bool {
        self.stages.contains_key(name)
    }
}

/// Run-identity fields mirrored into the manifest so a resuming session can
/// cross-check that it was handed the same run parameters.
#[derive(Debug, Clone)]
struct StageIdentity {
    model_name: String,
    down_top_k: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
}

/// sha256 + byte size of a file (manifest integrity record for the
/// persist→restore copy of each stage product).
fn sha256_file(path: &Path) -> Result<(u64, String), VindexError> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = std::io::Read::read(&mut file, &mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok((total, format!("{:x}", hasher.finalize())))
}

/// Build a vindex by streaming from safetensors files (no full model load).
///
/// Peak memory: embeddings + 1 layer of gate/down weights at a time.
#[allow(clippy::too_many_arguments)]
pub fn build_vindex_streaming(
    model_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    down_top_k: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    build_vindex_streaming_opts(
        model_dir,
        tokenizer,
        model_name,
        output_dir,
        down_top_k,
        extract_level,
        dtype,
        callbacks,
        &StreamingOpts::default(),
    )
}

/// `build_vindex_streaming` with `StreamingOpts` (consume-shards et al).
#[allow(clippy::too_many_arguments)]
pub fn build_vindex_streaming_opts(
    model_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    down_top_k: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
    callbacks: &mut dyn IndexBuildCallbacks,
    opts: &StreamingOpts,
) -> Result<(), VindexError> {
    std::fs::create_dir_all(output_dir)?;

    // Detect architecture
    let arch = larql_models::detect_architecture(model_dir)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    let prefixes = arch.key_prefixes_to_strip();
    let cfg = arch.config();

    let num_layers = cfg.num_layers;
    let hidden_size = cfg.hidden_size;
    let intermediate_size = cfg.intermediate_size;
    let embed_scale = arch.embed_scale();
    let is_moe = arch.is_moe();
    let n_experts = arch.num_experts();

    // Mmap all safetensors files
    let mut st_files: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
        .collect();
    if st_files.is_empty() {
        let weights_dir = model_dir.join("weights");
        if weights_dir.is_dir() {
            st_files = std::fs::read_dir(&weights_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|ext| ext == "safetensors"))
                .collect();
        }
    }
    st_files.sort();

    if st_files.is_empty() {
        return Err(VindexError::NoSafetensors(model_dir.to_path_buf()));
    }

    callbacks.on_stage("loading");
    eprintln!("  Streaming mode: {} safetensors shards (mmap'd, not loaded)", st_files.len());

    // (shards vec was for an earlier design — tensor_index + ShardStore is the actual approach)

    // SAFETY: We need to hold both the mmap and the SafeTensors that borrows from it.
    // We use a two-phase approach: first mmap all files, then deserialize.
    // The mmaps are kept alive in the ShardStore until released (consume-shards)
    // or until the end of the function.
    let mut shard_store = ShardStore::open_all(st_files.clone())?;
    let mut consumed = Consumption::default();

    // Build a tensor index: key → (shard_idx, tensor_name)
    // We need to find which shard contains each tensor.
    let mut tensor_index: HashMap<String, (usize, String)> = HashMap::new();
    for shard_idx in 0..shard_store.len() {
        let st = safetensors::SafeTensors::deserialize(shard_store.bytes(shard_idx)?)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        for name in st.names() {
            let key = normalize_key(name, prefixes);
            tensor_index.insert(key.clone(), (shard_idx, name.to_string()));
        }
    }

    // consume-shards accounting: the exact key set a shard must have served
    // before it may be deleted ("removed once every needed tensor is consumed").
    let plan_keys: Vec<String> = extraction_plan_keys(arch.as_ref(), num_layers, prefixes);
    let mut plan_by_shard: Vec<HashSet<String>> = vec![HashSet::new(); shard_store.len()];
    for key in &plan_keys {
        if let Some((shard_idx, name)) = tensor_index.get(key) {
            plan_by_shard[*shard_idx].insert(name.clone());
        }
    }
    if opts.consume_shards {
        let planned: usize = plan_by_shard.iter().map(|s| s.len()).sum();
        eprintln!(
            "  consume-shards: {} shard(s), {planned} planned tensor(s)",
            shard_store.len()
        );
    }

    callbacks.on_stage_done("loading", 0.0);

    // ── Stage-resume state (two-session split) ──
    // Load the manifest from a previous session, if any. Completed stages are
    // skipped; their tensor reads are pre-marked as consumed so consume-shards
    // can still release shards whose remaining plan keys are all served.
    let stage_control = &opts.stage_control;
    let mut manifest = stage_control
        .manifest_path
        .as_ref()
        .and_then(|p| StageManifest::load(p));
    if let Some(m) = &manifest {
        let done: Vec<&String> = m.stages.keys().collect();
        eprintln!("  stage-resume: manifest loaded, completed stages: {done:?}");
        if !m.layer_infos.is_empty() {
            for key in gate_stage_keys(arch.as_ref(), num_layers, &prefixes) {
                if let Some((shard_idx, name)) = tensor_index.get(&key) {
                    consumed.record(*shard_idx, name);
                }
            }
        }
        if m.stage_done("router_weights") {
            for key in router_stage_keys(arch.as_ref(), num_layers, &prefixes) {
                if let Some((shard_idx, name)) = tensor_index.get(&key) {
                    consumed.record(*shard_idx, name);
                }
            }
        }
        if m.stage_done("embeddings") {
            let embed_key = normalize_key(arch.embed_key(), prefixes);
            if let Some((shard_idx, name)) = tensor_index.get(&embed_key) {
                consumed.record(*shard_idx, name);
            }
        }
    }

    // Persist one stage product into the manifest and rewrite the manifest file.
    fn persist_stage(
        manifest: &mut Option<StageManifest>,
        manifest_path: Option<&Path>,
        output_dir: &Path,
        stage: &str,
        file: &str,
        layer_infos: Option<&[VindexLayerInfo]>,
        vocab_size: Option<usize>,
        done_layers: Option<usize>,
        identity: &StageIdentity,
    ) {
        let Some(mp) = manifest_path else { return };
        let m = manifest.get_or_insert_with(|| StageManifest {
            format: StageManifest::FORMAT.to_string(),
            model_name: identity.model_name.clone(),
            down_top_k: identity.down_top_k,
            extract_level: identity.extract_level,
            dtype: identity.dtype,
            vocab_size: 0,
            stages: HashMap::new(),
            layer_infos: Vec::new(),
        });
        if let Some(v) = vocab_size {
            m.vocab_size = v;
        }
        if let Some(li) = layer_infos {
            m.layer_infos = li.to_vec();
        }
        let fp = output_dir.join(file);
        let recorded = match sha256_file(&fp) {
            Ok((bytes, sha)) => {
                m.stages.insert(
                    stage.to_string(),
                    StageRecord { output: file.to_string(), bytes, sha256: sha, done_layers },
                );
                true
            }
            Err(e) => {
                eprintln!("  stage-resume: WARNING could not hash {}: {e}", fp.display());
                false
            }
        };
        if recorded {
            match std::fs::File::create(mp).and_then(|f| {
                serde_json::to_writer_pretty(f, &*m).map_err(|e| std::io::Error::new(
                    std::io::ErrorKind::Other, e.to_string(),
                ))
            }) {
                Ok(()) => eprintln!("  stage-resume: recorded stage {stage:?} -> {}", mp.display()),
                Err(e) => eprintln!("  stage-resume: WARNING manifest write failed: {e}"),
            }
        }
    }

    // Clean stop: release open shards when asked, exit Ok.
    macro_rules! stop_early {
        () => {{
            if opts.consume_shards {
                for shard_idx in 0..shard_store.len() {
                    if shard_store.slots[shard_idx].open.is_some() {
                        shard_store.release_and_delete(shard_idx);
                    }
                }
            }
            return Ok(());
        }};
    }

    // ── 1. Gate vectors (streaming, one layer at a time) ──
    let expert_format = arch.expert_format();
    let identity = StageIdentity {
        model_name: model_name.to_string(),
        down_top_k,
        extract_level,
        dtype,
    };
    // A stop_after target that is ALREADY complete means this run would fall
    // through the per-stage stop checks (skipped stages never complete here)
    // and continue into later stages — stop now instead, with the same clean
    // exit contract as a fresh stop.
    if let Some(sa) = stage_control.stop_after.as_deref() {
        if manifest.as_ref().is_some_and(|m| m.stage_done(sa)) {
            eprintln!("  stage-resume: stop-after target {sa:?} already complete; stopping");
            stop_early!();
        }
    }
    let gate_done = manifest.as_ref().is_some_and(|m| m.stage_done("gate_vectors"));
    let mut layer_infos: Vec<VindexLayerInfo> = Vec::new();
    if gate_done {
        layer_infos = manifest
            .as_ref()
            .map(|m| m.layer_infos.clone())
            .unwrap_or_default();
        if layer_infos.is_empty() {
            return Err(VindexError::Parse(
                "stage-manifest records gate_vectors complete but carries no layer_infos".to_string(),
            ));
        }
        let gate_file = output_dir.join("gate_vectors.bin");
        if !gate_file.exists() {
            return Err(VindexError::Parse(format!(
                "stage-resume: gate_vectors complete but {} is missing — restore it into the output dir before resuming",
                gate_file.display()
            )));
        }
        eprintln!(
            "  stage-resume: gate_vectors complete ({} layer infos); skipping gate stage",
            layer_infos.len()
        );
        callbacks.on_stage_done("gate_vectors", 0.0);
    } else {
    callbacks.on_stage("gate_vectors");
    let gate_path = output_dir.join("gate_vectors.bin");
    let mut gate_file = BufWriter::new(std::fs::File::create(&gate_path)?);
    let mut offset: u64 = 0;

    // Check expert format from the architecture
    let expert_format = arch.expert_format();

    for layer in 0..num_layers {
        callbacks.on_layer_start("gate", layer, num_layers);
        let start = std::time::Instant::now();

        if expert_format == larql_models::ExpertFormat::PackedMxfp4 {
            // MXFP4 packed experts: dequantize gate_up_proj_blocks per layer
            // The fused tensor is [num_experts, 2*intermediate, groups, 16]
            // First half of output features = gate, second half = up
            let blocks_key = arch.packed_gate_up_blocks_key(layer).unwrap_or_default();
            let scales_key = arch.packed_gate_up_scales_key(layer).unwrap_or_default();

            if let (Some(blocks_info), Some(scales_info)) = (
                tensor_index.get(&blocks_key),
                tensor_index.get(&scales_key),
            ) {
                let blocks_st = safetensors::SafeTensors::deserialize(shard_store.bytes(blocks_info.0)?)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let scales_st = safetensors::SafeTensors::deserialize(shard_store.bytes(scales_info.0)?)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;

                let blocks_view = blocks_st.tensor(&blocks_info.1)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let scales_view = scales_st.tensor(&scales_info.1)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;

                let shape = blocks_view.shape();
                let n_exp = shape[0];
                let out_features = shape[1]; // 2 * intermediate (fused gate+up)
                let groups = shape[2];
                let in_features = groups * 32;
                let half = out_features / 2; // gate portion

                consumed.record(blocks_info.0, &blocks_info.1);
                consumed.record(scales_info.0, &scales_info.1);

                let experts = crate::format::quant::mxfp4::dequantize_all_experts(
                    blocks_view.data(), scales_view.data(), n_exp, out_features, groups,
                );

                let mut total_features = 0usize;
                let mut layer_bytes = 0u64;

                for expert_data in &experts {
                    // Extract gate portion (first half rows)
                    let gate_data = &expert_data[..half * in_features];
                    layer_bytes += write_floats(&mut gate_file, gate_data, dtype)?;
                    total_features += half;
                }

                if total_features > 0 {
                    layer_infos.push(VindexLayerInfo {
                        layer, num_features: total_features, offset, length: layer_bytes,
                        num_experts: Some(n_exp),
                        num_features_per_expert: Some(half),
                    });
                    offset += layer_bytes;
                }
            }
        } else if expert_format == larql_models::ExpertFormat::PackedBf16
            && is_moe
            && n_experts > 0
        {
            // Packed BF16 fused experts (Gemma 4 26B-A4B): stream the fused
            // tensor one expert slice at a time. Only the GATE half of each
            // expert's [2*mi, hidden] block is written (MXFP4 convention).
            let gate_up_key = arch
                .packed_experts_gate_up_key(layer)
                .map(|k| normalize_key(&k, prefixes))
                .ok_or_else(|| {
                    VindexError::Parse(format!(
                        "PackedBf16 model but packed_experts_gate_up_key({layer}) is None"
                    ))
                })?;

            let info = packed_tensor_info(&shard_store, &tensor_index, &gate_up_key)?
                .ok_or_else(|| VindexError::MissingTensor(gate_up_key.clone()))?;
            if info.num_experts != n_experts {
                return Err(VindexError::Parse(format!(
                    "{gate_up_key}: dim0 = {} experts, config declares {n_experts}",
                    info.num_experts
                )));
            }
            let moe_intermediate = packed_moe_intermediate(&gate_up_key, &info, arch.as_ref())?;

            let mut total_features = 0usize;
            let mut layer_bytes = 0u64;
            for expert in 0..n_experts {
                // Ranged read + lazy BF16→f32: rows [0..moe_intermediate] of
                // the expert's [2*moe_intermediate, hidden] block.
                let gate = packed_expert_rows_f32(
                    &shard_store,
                    &info,
                    expert,
                    0,
                    moe_intermediate,
                    &mut consumed,
                )?;
                layer_bytes += write_floats(&mut gate_file, gate.as_slice().unwrap(), dtype)?;
                total_features += moe_intermediate;
            }

            layer_infos.push(VindexLayerInfo {
                layer,
                num_features: total_features,
                offset,
                length: layer_bytes,
                num_experts: Some(n_experts),
                num_features_per_expert: Some(moe_intermediate),
            });
            offset += layer_bytes;
        } else if is_moe && n_experts > 0 {
            // Standard MoE (Mixtral): per-expert gate tensors
            let mut total_features = 0usize;
            let mut layer_bytes = 0u64;
            let mut features_per_expert = 0usize;

            for expert in 0..n_experts {
                let gate_key = match arch.expert_ffn_gate_key(layer, expert) {
                    Some(k) => normalize_key(&k, prefixes),
                    None => continue,
                };

                if let Some(tensor) = get_tensor_f32(&shard_store, &tensor_index, &gate_key, &mut consumed)? {
                    features_per_expert = tensor.shape()[0];
                    total_features += features_per_expert;
                    let data = tensor.as_slice().unwrap();
                    layer_bytes += write_floats(&mut gate_file, data, dtype)?;
                }
            }

            if total_features > 0 {
                layer_infos.push(VindexLayerInfo {
                    layer, num_features: total_features, offset, length: layer_bytes,
                    num_experts: Some(n_experts),
                    num_features_per_expert: Some(features_per_expert),
                });
                offset += layer_bytes;
            }
        } else {
            // Dense: single gate matrix per layer
            let gate_key = normalize_key(&arch.ffn_gate_key(layer), prefixes);
            if let Some(tensor) = get_tensor_f32(&shard_store, &tensor_index, &gate_key, &mut consumed)? {
                let num_features = tensor.shape()[0];
                let data = tensor.as_slice().unwrap();
                let length = write_floats(&mut gate_file, data, dtype)?;
                layer_infos.push(VindexLayerInfo {
                    layer, num_features, offset, length,
                    num_experts: None, num_features_per_expert: None,
                });
                offset += length;
            }
        }

        callbacks.on_layer_done("gate", layer, start.elapsed().as_secs_f64() * 1000.0);
    }
    gate_file.flush()?;
    callbacks.on_stage_done("gate_vectors", 0.0);
    persist_stage(
        &mut manifest,
        stage_control.manifest_path.as_deref(),
        output_dir,
        "gate_vectors",
        "gate_vectors.bin",
        Some(&layer_infos),
        None,
        None,
        &identity,
    );
    if stage_control.stop_after.as_deref() == Some("gate_vectors") {
        stop_early!();
    }
    } // end gate else (not resumed-complete)

    // ── 1b. Router weights (MoE models only) ──
    if is_moe && !manifest.as_ref().is_some_and(|m| m.stage_done("router_weights")) {
        callbacks.on_stage("router_weights");
        let router_path = output_dir.join("router_weights.bin");
        let mut router_file = BufWriter::new(std::fs::File::create(&router_path)?);

        for layer in 0..num_layers {
            let router_key = arch.moe_router_key(layer)
                .map(|k| normalize_key(&k, prefixes))
                .unwrap_or_default();

            if let Some(tensor) = get_tensor_f32(&shard_store, &tensor_index, &router_key, &mut consumed)? {
                let data = tensor.as_slice().unwrap();
                let bytes = crate::config::dtype::encode_floats(data, dtype);
                router_file.write_all(&bytes)?;
            }

            // Also try router bias
            let bias_key = router_key.replace(".weight", ".bias");
            if let Some(tensor) = get_tensor_f32(&shard_store, &tensor_index, &bias_key, &mut consumed)? {
                let data = tensor.as_slice().unwrap();
                let bytes = crate::config::dtype::encode_floats(data, dtype);
                // Write bias after weight for each layer
                router_file.write_all(&bytes)?;
            }
        }
        router_file.flush()?;
        callbacks.on_stage_done("router_weights", 0.0);
        persist_stage(
            &mut manifest,
            stage_control.manifest_path.as_deref(),
            output_dir,
            "router_weights",
            "router_weights.bin",
            None,
            None,
            None,
            &identity,
        );
        if stage_control.stop_after.as_deref() == Some("router_weights") {
            stop_early!();
        }
    } else if is_moe {
        // router_weights recorded complete in the manifest — output restored
        // by the resuming session.
        eprintln!("  stage-resume: router_weights complete; skipping router stage");
        callbacks.on_stage_done("router_weights", 0.0);
    }

    // ── 2. Embeddings ──
    let embed_done = manifest.as_ref().is_some_and(|m| m.stage_done("embeddings"));
    let mut vocab_size = manifest.as_ref().map(|m| m.vocab_size).unwrap_or(0);
    // The down_meta stage multiplies against the embedding matrix. When the
    // embeddings stage is skipped (its product restored from the previous
    // session), the embed tensor is re-read from the shard right before the
    // down stage — same tensor, same decode, so down-stage logits are
    // byte-identical to a single-session run.
    let mut embed: Option<Array2<f32>> = None;
    if embed_done {
        if !output_dir.join("embeddings.bin").exists() {
            return Err(VindexError::Parse(
                "stage-resume: embeddings complete but embeddings.bin is missing — restore it into the output dir before resuming".to_string(),
            ));
        }
        eprintln!(
            "  stage-resume: embeddings complete; skipping embeddings stage (embed tensor re-read from shard for the down stage)"
        );
        callbacks.on_stage_done("embeddings", 0.0);
    } else {
    callbacks.on_stage("embeddings");
    let embed_key = normalize_key(arch.embed_key(), prefixes);
    let e = get_tensor_f32(&shard_store, &tensor_index, &embed_key, &mut consumed)?
        .ok_or_else(|| VindexError::MissingTensor(embed_key.clone()))?;
    vocab_size = e.shape()[0];
    let embed_data = e.as_slice().unwrap();
    let embed_bytes = crate::config::dtype::encode_floats(embed_data, dtype);
    std::fs::write(output_dir.join("embeddings.bin"), &embed_bytes)?;
    embed = Some(e);
    callbacks.on_stage_done("embeddings", 0.0);
    persist_stage(
        &mut manifest,
        stage_control.manifest_path.as_deref(),
        output_dir,
        "embeddings",
        "embeddings.bin",
        None,
        Some(vocab_size),
        None,
        &identity,
    );
    if stage_control.stop_after.as_deref() == Some("embeddings") {
        stop_early!();
    }
    }

    // ── 3. Down meta (streaming; threaded + per-layer resumable) ──
    let down_done_layers = manifest
        .as_ref()
        .and_then(|m| m.stages.get("down_meta"))
        .and_then(|r| r.done_layers)
        .unwrap_or(0);
    let down_done =
        manifest.as_ref().is_some_and(|m| m.stage_done("down_meta")) && down_done_layers >= num_layers;
    if down_done {
        if !output_dir.join("down_meta.bin").exists() {
            return Err(VindexError::Parse(
                "stage-resume: down_meta complete but down_meta.bin is missing — restore it into the output dir before resuming".to_string(),
            ));
        }
        if vocab_size == 0 {
            return Err(VindexError::Parse(
                "stage-resume: down_meta complete but the manifest carries no vocab_size".to_string(),
            ));
        }
        eprintln!("  stage-resume: down_meta complete ({num_layers}/{num_layers} layers); skipping down stage");
        callbacks.on_stage_done("down_meta", 0.0);
    } else {
    callbacks.on_stage("down_meta");
    // The down stage multiplies against the embedding matrix. On resume the
    // embeddings stage was skipped, so re-read the embed tensor from its
    // shard (same tensor, same decode → identical logits).
    let embed = match embed.take() {
        Some(e) => e,
        None => {
            let embed_key = normalize_key(arch.embed_key(), prefixes);
            get_tensor_f32(&shard_store, &tensor_index, &embed_key, &mut consumed)?
                .ok_or_else(|| VindexError::MissingTensor(embed_key.clone()))?
        }
    };
    vocab_size = embed.shape()[0];
    let mut all_down_meta: Vec<Option<Vec<Option<crate::FeatureMeta>>>> = vec![None; num_layers];

    // Per-layer checkpoint resume: reload records for the layers an
    // interrupted run already finished (same down_meta.bin format, partial
    // file — not-yet-computed layers read back as None).
    let mut start_layer = 0usize;
    if down_done_layers > 0 && crate::format::down_meta::has_binary(output_dir) {
        let (partial, _total, _tk) = crate::format::down_meta::read_binary_raw(output_dir, num_layers)?;
        let keep = down_done_layers.min(num_layers).min(partial.len());
        for (i, v) in partial.into_iter().take(keep).enumerate() {
            all_down_meta[i] = v;
        }
        start_layer = keep;
        eprintln!(
            "  stage-resume: down_meta checkpoint holds {start_layer}/{num_layers} layer(s); resuming from layer {start_layer}"
        );
    }

    // Build whole-word vocab once
    let (_ww_ids, _ww_embed) = super::build::build_whole_word_vocab(tokenizer, &embed, vocab_size, hidden_size);

    let down_threads = stage_control.down_threads;
    eprintln!(
        "  down_meta: {num_layers} layer(s) to compute (from {start_layer}), threads={down_threads}, top_k={down_top_k}"
    );
    for layer in start_layer..num_layers {
        let layer_down_meta = &mut all_down_meta[layer];
        callbacks.on_layer_start("down", layer, num_layers);
        let start = std::time::Instant::now();

        // Get down matrices for this layer
        let down_matrices: Vec<Array2<f32>> = if expert_format == larql_models::ExpertFormat::PackedMxfp4 {
            // MXFP4: dequantize down_proj_blocks
            let blocks_key = arch.packed_down_blocks_key(layer).unwrap_or_default();
            let scales_key = arch.packed_down_scales_key(layer).unwrap_or_default();
            if let (Some(bi), Some(si)) = (tensor_index.get(&blocks_key), tensor_index.get(&scales_key)) {
                let bst = safetensors::SafeTensors::deserialize(shard_store.bytes(bi.0)?)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let sst = safetensors::SafeTensors::deserialize(shard_store.bytes(si.0)?)
                    .map_err(|e| VindexError::Parse(e.to_string()))?;
                let bv = bst.tensor(&bi.1).map_err(|e| VindexError::Parse(e.to_string()))?;
                let sv = sst.tensor(&si.1).map_err(|e| VindexError::Parse(e.to_string()))?;
                let shape = bv.shape();
                let n_exp = shape[0];
                let out_features = shape[1];
                let groups = shape[2];
                let in_features = groups * 32;
                let experts = crate::format::quant::mxfp4::dequantize_all_experts(
                    bv.data(), sv.data(), n_exp, out_features, groups,
                );
                experts.into_iter().map(|data| {
                    Array2::from_shape_vec((out_features, in_features), data).unwrap()
                }).collect()
            } else {
                callbacks.on_layer_done("down", layer, 0.0); continue;
            }
        } else if expert_format == larql_models::ExpertFormat::PackedBf16
            && is_moe
            && n_experts > 0
        {
            // Packed BF16 fused experts (Gemma 4): slice the fused
            // [num_experts, hidden, moe_intermediate] tensor by dim0, one
            // expert at a time (ranged read + lazy BF16→f32). Expert-major
            // ordering matches the gate stage, keeping feature ids aligned.
            let down_key = arch
                .packed_experts_down_key(layer)
                .map(|k| normalize_key(&k, prefixes))
                .ok_or_else(|| {
                    VindexError::Parse(format!(
                        "PackedBf16 model but packed_experts_down_key({layer}) is None"
                    ))
                })?;

            let info = packed_tensor_info(&shard_store, &tensor_index, &down_key)?
                .ok_or_else(|| VindexError::MissingTensor(down_key.clone()))?;
            if info.num_experts != n_experts {
                return Err(VindexError::Parse(format!(
                    "{down_key}: dim0 = {} experts, config declares {n_experts}",
                    info.num_experts
                )));
            }
            let mi_cfg = arch.moe_intermediate_size();
            if mi_cfg > 0 && info.cols != mi_cfg {
                return Err(VindexError::Parse(format!(
                    "{down_key}: moe_intermediate = {}, config declares {mi_cfg}",
                    info.cols
                )));
            }
            if info.rows_per_expert != hidden_size {
                return Err(VindexError::Parse(format!(
                    "{down_key}: rows = {}, hidden_size = {hidden_size}",
                    info.rows_per_expert
                )));
            }

            let mut mats = Vec::with_capacity(n_experts);
            for expert in 0..n_experts {
                mats.push(packed_expert_rows_f32(
                    &shard_store,
                    &info,
                    expert,
                    0,
                    info.rows_per_expert,
                    &mut consumed,
                )?);
            }
            mats
        } else if is_moe && n_experts > 0 {
            let mut mats = Vec::new();
            for expert in 0..n_experts {
                if let Some(key) = arch.expert_ffn_down_key(layer, expert) {
                    let nk = normalize_key(&key, prefixes);
                    if let Some(t) = get_tensor_f32(&shard_store, &tensor_index, &nk, &mut consumed)? {
                        mats.push(t);
                    }
                }
            }
            mats
        } else {
            let down_key = normalize_key(&arch.ffn_down_key(layer), prefixes);
            match get_tensor_f32(&shard_store, &tensor_index, &down_key, &mut consumed)? {
                Some(t) => vec![t],
                None => { callbacks.on_layer_done("down", layer, 0.0); continue; }
            }
        };

        if down_matrices.is_empty() {
            callbacks.on_layer_done("down", layer, 0.0);
            continue;
        }

        if down_threads > 1 {
            // Threaded path: expert-major feature blocks are distributed
            // round-robin across workers; each block is computed with the
            // SAME matmul shapes and per-feature math as the serial path, so
            // the assembled layer records are byte-identical (validated on
            // the gemma4 fixture). Per-batch on_feature_progress callbacks
            // are not emitted on this path (SilentBuildCallbacks is the
            // remote-run caller; layer start/done still fire).
            let metas = compute_layer_down_meta_threaded(
                &embed, &down_matrices, down_top_k, down_threads, tokenizer, layer,
            );
            *layer_down_meta = Some(metas);
        } else {
        let mut feature_offset = 0usize;
        for w_down in &down_matrices {
            let num_features = w_down.shape()[1];
            let batch_size = 1024;

            for batch_start in (0..num_features).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(num_features);
                callbacks.on_feature_progress("down", layer, feature_offset + batch_start,
                    down_matrices.iter().map(|m| m.shape()[1]).sum());

                let w_chunk = w_down.slice(ndarray::s![.., batch_start..batch_end]).to_owned();
                let cpu = larql_compute::CpuBackend;
                use larql_compute::ComputeBackend;
                let chunk_logits = cpu.matmul(embed.view(), w_chunk.view());

                for feat in batch_start..batch_end {
                    let col = chunk_logits.column(feat - batch_start);
                    let meta = down_feature_meta(col, down_top_k, tokenizer);
                    let feat_idx = feature_offset + feat;
                    if layer_down_meta.is_none() {
                        *layer_down_meta = Some(Vec::new());
                    }
                    if let Some(ref mut metas) = layer_down_meta {
                        while metas.len() <= feat_idx { metas.push(None); }
                        metas[feat_idx] = Some(meta);
                    }
                }
            }
            feature_offset += num_features;
        }
        } // end serial else

        callbacks.on_layer_done("down", layer, start.elapsed().as_secs_f64() * 1000.0);

        // Per-layer checkpoint (manifest mode only): rewrite down_meta.bin in
        // the unchanged binary format (uncomputed layers are None records)
        // and record the layer watermark, so an interrupted session resumes
        // from the next layer instead of restarting the whole stage.
        if stage_control.manifest_path.is_some() {
            crate::format::down_meta::write_binary(output_dir, &all_down_meta, down_top_k)?;
            record_down_checkpoint(
                &mut manifest,
                stage_control.manifest_path.as_deref(),
                layer + 1,
            );
        }
    }

    crate::format::down_meta::write_binary(output_dir, &all_down_meta, down_top_k)?;
    callbacks.on_stage_done("down_meta", 0.0);
    persist_stage(
        &mut manifest,
        stage_control.manifest_path.as_deref(),
        output_dir,
        "down_meta",
        "down_meta.bin",
        None,
        Some(vocab_size),
        Some(num_layers),
        &identity,
    );
    if stage_control.stop_after.as_deref() == Some("down_meta") {
        stop_early!();
    }
    } // end down else (not resumed-complete)

    // ── consume-shards (browse level): the full browse consumption plan has
    // been served by now, so every shard whose planned tensors are all
    // recorded is safe to drop and delete. At Inference/All levels this
    // release is deferred past the model-weights stage.
    if opts.consume_shards && extract_level == crate::ExtractLevel::Browse {
        for shard_idx in 0..shard_store.len() {
            let pending: Vec<&String> = plan_by_shard[shard_idx]
                .iter()
                .filter(|name| !consumed.served(shard_idx).contains(*name))
                .collect();
            if pending.is_empty() {
                shard_store.release_and_delete(shard_idx);
            } else {
                eprintln!(
                    "  consume-shards: keeping {} ({} planned tensor(s) unserved)",
                    shard_store.path(shard_idx).display(),
                    pending.len()
                );
            }
        }
    }

    // ── 4. Tokenizer ──
    callbacks.on_stage("tokenizer");
    let tokenizer_json = tokenizer.to_string(true)
        .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
    std::fs::write(output_dir.join("tokenizer.json"), tokenizer_json)?;
    callbacks.on_stage_done("tokenizer", 0.0);

    // ── 5. Config ──
    let family = arch.family().to_string();
    let config = VindexConfig {
        version: 2,
        model: model_name.to_string(),
        family: family.clone(),
        num_layers, hidden_size, intermediate_size, vocab_size,
        embed_scale,
        layers: layer_infos,
        down_top_k,
        has_model_weights: false,
        source: Some(crate::VindexSource {
            huggingface_repo: Some(model_name.to_string()),
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: super::build::chrono_now(),
            larql_version: env!("CARGO_PKG_VERSION").to_string(),
        }),
        checksums: None,
        extract_level,
        dtype,
        layer_bands: crate::LayerBands::for_family(&family, num_layers),
        model_config: Some(VindexModelConfig {
            model_type: cfg.model_type.clone(),
            head_dim: cfg.head_dim,
            num_q_heads: cfg.num_q_heads,
            num_kv_heads: cfg.num_kv_heads,
            rope_base: cfg.rope_base,
            sliding_window: cfg.sliding_window,
            moe: if is_moe {
                Some(crate::MoeConfig {
                    num_experts: n_experts,
                    top_k: arch.num_experts_per_token(),
                    shared_expert: arch.num_shared_experts() > 0,
                    router_type: "top_k_softmax".to_string(),
                })
            } else { None },
            // Per-layer geometry (Gemma 4)
            global_head_dim: cfg.global_head_dim,
            num_global_kv_heads: cfg.num_global_kv_heads,
            partial_rotary_factor: cfg.partial_rotary_factor,
            sliding_window_pattern: cfg.sliding_window_pattern,
            layer_types: cfg.layer_types.clone(),
            attention_k_eq_v: cfg.attention_k_eq_v,
            num_kv_shared_layers: cfg.num_kv_shared_layers,
            per_layer_embed_dim: cfg.per_layer_embed_dim,
            rope_local_base: cfg.rope_local_base,
            query_pre_attn_scalar: cfg.query_pre_attn_scalar,
        }),
    };

    // Write preliminary index.json (needed by write_model_weights which reads dtype from it)
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join("index.json"), config_json)?;

    // ── 6. Model weights (if extract level requires them) ──
    if extract_level != crate::ExtractLevel::Browse {
        // Note: consume-shards never releases before this point above browse
        // level, so every slot is guaranteed open here.
        let mut shard_refs: Vec<&[u8]> = Vec::with_capacity(shard_store.len());
        for shard_idx in 0..shard_store.len() {
            shard_refs.push(shard_store.bytes(shard_idx)?);
        }
        let streaming_source = crate::format::weights::StreamingWeights {
            shard_mmaps: &shard_refs,
            tensor_index: &tensor_index,
            arch: &*arch,
            num_layers,
        };
        crate::format::weights::write_model_weights(&streaming_source, output_dir, callbacks)?;
        // write_model_weights updates index.json with has_model_weights=true
    }

    // Final checksums
    let config_text = std::fs::read_to_string(output_dir.join("index.json"))?;
    let mut config: VindexConfig = serde_json::from_str(&config_text)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    config.checksums = crate::format::checksums::compute_checksums(output_dir).ok();
    let config_json = serde_json::to_string_pretty(&config)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    std::fs::write(output_dir.join("index.json"), config_json)?;

    // ── consume-shards (final release): nothing reads the shards past this
    // point, so any still-open shard (higher extract levels, or shards whose
    // plan keys were absent from the checkpoint) is now safe to drop+delete.
    if opts.consume_shards {
        for shard_idx in 0..shard_store.len() {
            if shard_store.slots[shard_idx].open.is_some() {
                shard_store.release_and_delete(shard_idx);
            }
        }
    }

    Ok(())
}

/// Item size in bytes for the float dtypes the extractor consumes.
fn dtype_itemsize(dtype: safetensors::Dtype) -> Option<usize> {
    match dtype {
        safetensors::Dtype::F32 => Some(4),
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 => Some(2),
        _ => None,
    }
}

/// Decode a raw byte slice of the given safetensors dtype to f32.
fn decode_to_f32(dtype: safetensors::Dtype, data: &[u8]) -> Option<Vec<f32>> {
    match dtype {
        safetensors::Dtype::F32 => Some(
            data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        ),
        safetensors::Dtype::F16 => Some(crate::format::quant::half::decode_f16(data)),
        safetensors::Dtype::BF16 => Some(crate::format::quant::half::decode_bf16(data)),
        _ => None,
    }
}

/// Header of a packed fused expert tensor, peeked from its shard.
#[derive(Debug, Clone)]
struct PackedInfo {
    shard_idx: usize,
    tensor_name: String,
    num_experts: usize,
    rows_per_expert: usize,
    cols: usize,
    dtype: safetensors::Dtype,
}

/// Peek a packed fused tensor `[num_experts, rows_per_expert, cols]` in the
/// mmap'd shards (zero copy — SafeTensors borrows the mmap).
fn packed_tensor_info(
    store: &ShardStore,
    index: &HashMap<String, (usize, String)>,
    key: &str,
) -> Result<Option<PackedInfo>, VindexError> {
    let Some(&(shard_idx, ref tensor_name)) = index.get(key) else {
        return Ok(None);
    };
    let st = safetensors::SafeTensors::deserialize(store.bytes(shard_idx)?)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    let view = st
        .tensor(tensor_name)
        .map_err(|e| VindexError::Parse(format!("{key}: {e}")))?;
    let shape = view.shape();
    if shape.len() != 3 {
        return Err(VindexError::Parse(format!(
            "{key}: expected 3-D [num_experts, rows, cols], got {shape:?}"
        )));
    }
    if dtype_itemsize(view.dtype()).is_none() {
        return Err(VindexError::UnsupportedDtype(format!("{key}: {:?}", view.dtype())));
    }
    Ok(Some(PackedInfo {
        shard_idx,
        tensor_name: tensor_name.clone(),
        num_experts: shape[0],
        rows_per_expert: shape[1],
        cols: shape[2],
        dtype: view.dtype(),
    }))
}

/// Slice ONE expert out of a packed fused tensor as `[row_end - row_begin, cols]`
/// f32. Reads only that expert's byte range from the mmap (ranged read) and
/// decodes dtype→f32 lazily for exactly that slice.
fn packed_expert_rows_f32(
    store: &ShardStore,
    info: &PackedInfo,
    expert: usize,
    row_begin: usize,
    row_end: usize,
    consumed: &mut Consumption,
) -> Result<Array2<f32>, VindexError> {
    if expert >= info.num_experts {
        return Err(VindexError::Parse(format!(
            "{}: expert {expert} out of range (dim0 = {})",
            info.tensor_name, info.num_experts
        )));
    }
    if row_end > info.rows_per_expert || row_begin >= row_end {
        return Err(VindexError::Parse(format!(
            "{}: bad row range {row_begin}..{row_end} (rows = {})",
            info.tensor_name, info.rows_per_expert
        )));
    }
    let itemsize = dtype_itemsize(info.dtype)
        .ok_or_else(|| VindexError::UnsupportedDtype(format!("{:?}", info.dtype)))?;

    let st = safetensors::SafeTensors::deserialize(store.bytes(info.shard_idx)?)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    let view = st
        .tensor(&info.tensor_name)
        .map_err(|e| VindexError::Parse(format!("{}: {e}", info.tensor_name)))?;

    // Expert e occupies dim0 slice e: rows [e*rows_per_expert, (e+1)*rows_per_expert)
    // of the flattened [num_experts * rows_per_expert, cols] matrix.
    let row_stride = info.cols * itemsize;
    let begin = (expert * info.rows_per_expert + row_begin) * row_stride;
    let end = begin + (row_end - row_begin) * row_stride;
    let data = &view.data()[begin..end];
    let floats = decode_to_f32(info.dtype, data)
        .ok_or_else(|| VindexError::UnsupportedDtype(format!("{:?}", info.dtype)))?;

    consumed.record(info.shard_idx, &info.tensor_name);
    Array2::from_shape_vec((row_end - row_begin, info.cols), floats)
        .map_err(|e| VindexError::Parse(e.to_string()))
}

/// Cross-check the fused gate_up geometry (rows = 2 * moe_intermediate, gate
/// rows first) against the config-declared `moe_intermediate_size`. Catches a
/// wrong or truncated download before megabytes of output are written.
fn packed_moe_intermediate(
    key: &str,
    info: &PackedInfo,
    arch: &dyn larql_models::ModelArchitecture,
) -> Result<usize, VindexError> {
    if info.rows_per_expert % 2 != 0 {
        return Err(VindexError::Parse(format!(
            "{key}: rows_per_expert {} is not 2 * moe_intermediate",
            info.rows_per_expert
        )));
    }
    let derived = info.rows_per_expert / 2;
    let declared = arch.moe_intermediate_size();
    if declared > 0 && declared != derived {
        return Err(VindexError::Parse(format!(
            "{key}: rows_per_expert {} implies moe_intermediate {derived}, config declares {declared}",
            info.rows_per_expert
        )));
    }
    Ok(derived)
}

/// The exact key set the extraction pipeline requests at browse level — the
/// tensors a shard must have served before consume-shards may delete it.
/// (Attention / vision tensors are intentionally absent: browse never reads
/// them, so they must never block a shard release.)
fn extraction_plan_keys(
    arch: &dyn larql_models::ModelArchitecture,
    num_layers: usize,
    prefixes: &[&str],
) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for layer in 0..num_layers {
        match arch.expert_format() {
            larql_models::ExpertFormat::PackedBf16 => {
                if let Some(k) = arch.packed_experts_gate_up_key(layer) {
                    keys.push(k);
                }
                if let Some(k) = arch.packed_experts_down_key(layer) {
                    keys.push(k);
                }
            }
            larql_models::ExpertFormat::PackedMxfp4 => {
                for k in [
                    arch.packed_gate_up_blocks_key(layer),
                    arch.packed_gate_up_scales_key(layer),
                    arch.packed_down_blocks_key(layer),
                    arch.packed_down_scales_key(layer),
                ]
                .into_iter()
                .flatten()
                {
                    keys.push(k);
                }
            }
            _ if arch.is_moe() && arch.num_experts() > 0 => {
                for expert in 0..arch.num_experts() {
                    if let Some(k) = arch.expert_ffn_gate_key(layer, expert) {
                        keys.push(k);
                    }
                    if let Some(k) = arch.expert_ffn_down_key(layer, expert) {
                        keys.push(k);
                    }
                }
            }
            _ => {
                keys.push(arch.ffn_gate_key(layer));
                keys.push(arch.ffn_down_key(layer));
            }
        }
        if arch.is_moe() {
            if let Some(router_key) = arch.moe_router_key(layer) {
                keys.push(router_key.clone());
                // The router stage also probes a `.bias` variant of the router key.
                keys.push(router_key.replace(".weight", ".bias"));
            }
        }
    }
    keys.push(arch.embed_key().to_string());
    keys.iter().map(|k| normalize_key(k, prefixes)).collect()
}

/// Get a 2D tensor from mmap'd safetensors, dequantizing to f32.
fn get_tensor_f32(
    store: &ShardStore,
    index: &HashMap<String, (usize, String)>,
    key: &str,
    consumed: &mut Consumption,
) -> Result<Option<Array2<f32>>, VindexError> {
    let (shard_idx, tensor_name) = match index.get(key) {
        Some(v) => v,
        None => return Ok(None),
    };

    let st = safetensors::SafeTensors::deserialize(store.bytes(*shard_idx)?)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let view = st.tensor(tensor_name)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let shape = view.shape();
    if shape.len() != 2 { return Ok(None); }

    let data = match view.dtype() {
        safetensors::Dtype::F32 => {
            view.data().chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        }
        safetensors::Dtype::F16 => crate::format::quant::half::decode_f16(view.data()),
        safetensors::Dtype::BF16 => crate::format::quant::half::decode_bf16(view.data()),
        _ => return Ok(None), // skip non-float
    };

    let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    consumed.record(*shard_idx, tensor_name);
    Ok(Some(arr))
}

fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
}

fn write_floats(w: &mut impl Write, data: &[f32], dtype: StorageDtype) -> Result<u64, VindexError> {
    let bytes = crate::config::dtype::encode_floats(data, dtype);
    w.write_all(&bytes)?;
    Ok(bytes.len() as u64)
}

/// Update the down_meta layer watermark in the manifest WITHOUT re-hashing the
/// checkpoint (intermediate per-layer checkpoints are same-machine temp state;
/// the final stage record carries the sha256).
fn record_down_checkpoint(
    manifest: &mut Option<StageManifest>,
    manifest_path: Option<&Path>,
    done_layers: usize,
) {
    let Some(mp) = manifest_path else { return };
    let Some(m) = manifest.as_mut() else { return };
    let entry = m.stages.entry("down_meta".to_string()).or_insert_with(|| StageRecord {
        output: "down_meta.bin".to_string(),
        bytes: 0,
        sha256: String::new(),
        done_layers: None,
    });
    entry.done_layers = Some(done_layers);
    let write = std::fs::File::create(mp).and_then(|f| {
        serde_json::to_writer_pretty(f, &*m).map_err(|e| std::io::Error::new(
            std::io::ErrorKind::Other, e.to_string(),
        ))
    });
    if let Err(e) = write {
        eprintln!("  stage-resume: WARNING manifest write failed: {e}");
    }
}

/// Per-feature down_meta record — the EXACT math of the serial path
/// (select_nth partial top-k → truncate → sort → tokenizer decode filter),
/// factored out so the threaded path computes byte-identical records.
fn down_feature_meta(
    logits_col: ndarray::ArrayView1<f32>,
    down_top_k: usize,
    tokenizer: &tokenizers::Tokenizer,
) -> crate::FeatureMeta {
    let mut scores: Vec<(usize, f32)> = logits_col.iter().copied().enumerate().collect();
    let k = down_top_k.min(scores.len());
    if k > 0 && k < scores.len() {
        scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
    }
    scores.truncate(k);
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_k_entries: Vec<larql_models::TopKEntry> = scores
        .into_iter()
        .filter_map(|(idx, logit)| {
            tokenizer
                .decode(&[idx as u32], true)
                .ok()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .map(|token| larql_models::TopKEntry { token, token_id: idx as u32, logit })
        })
        .collect();

    let (top_token, top_token_id, c_score) = if let Some(first) = top_k_entries.first() {
        (first.token.clone(), first.token_id, first.logit)
    } else {
        (String::new(), 0, 0.0)
    };

    crate::FeatureMeta { top_token, top_token_id, c_score, top_k: top_k_entries }
}

/// Threaded down_meta for one layer: the layer's expert-major feature blocks
/// (one block per down matrix, contiguous global feature ids) are distributed
/// round-robin across workers. Each block is computed with the SAME matmul
/// shapes (batch 1024 within a matrix, identical to the serial path) and the
/// same per-feature math, so output is byte-identical to the serial path —
/// only the wall-clock distribution changes.
///
/// Progress: each worker throttles its own rate line to ≤1 per 30 s; the
/// per-batch `on_feature_progress` callback is not used on this path.
fn compute_layer_down_meta_threaded(
    embed: &Array2<f32>,
    down_matrices: &[Array2<f32>],
    down_top_k: usize,
    n_threads: usize,
    tokenizer: &tokenizers::Tokenizer,
    layer: usize,
) -> Vec<Option<crate::FeatureMeta>> {
    use larql_compute::ComputeBackend;

    // Contiguous global feature offset of each matrix (expert-major order,
    // matching the serial feature_offset accumulation).
    let mut offsets = Vec::with_capacity(down_matrices.len());
    let mut total = 0usize;
    for m in down_matrices {
        offsets.push(total);
        total += m.shape()[1];
    }

    let n_threads = n_threads.max(1).min(down_matrices.len().max(1));
    let batch_size = 1024usize;
    let cpu = &larql_compute::CpuBackend;
    let offsets = &offsets;
    let started = std::time::Instant::now();

    let results: Vec<Vec<Option<crate::FeatureMeta>>> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..n_threads {
            // Round-robin whole-matrix assignment keeps every matmul at the
            // serial path's exact shape ([vocab, mat_features] per batch).
            let mats: Vec<usize> = (t..down_matrices.len()).step_by(n_threads).collect();
            handles.push(s.spawn(move || {
                let mut out: Vec<Option<crate::FeatureMeta>> = Vec::new();
                let mut last_log = std::time::Instant::now();
                let mut done = 0usize;
                for mi in mats {
                    let w_down = &down_matrices[mi];
                    let num_features = w_down.shape()[1];
                    for batch_start in (0..num_features).step_by(batch_size) {
                        let batch_end = (batch_start + batch_size).min(num_features);
                        let w_chunk =
                            w_down.slice(ndarray::s![.., batch_start..batch_end]).to_owned();
                        let chunk_logits = cpu.matmul(embed.view(), w_chunk.view());
                        for feat in batch_start..batch_end {
                            let col = chunk_logits.column(feat - batch_start);
                            let meta = down_feature_meta(col, down_top_k, tokenizer);
                            out.push(Some(meta));
                        }
                        done += batch_end - batch_start;
                        if last_log.elapsed().as_secs() >= 30 {
                            let rate = done as f64 / last_log.elapsed().as_secs_f64();
                            eprintln!(
                                "  down_meta: layer {layer} thread {t}: mat {mi} batch {batch_start}..{batch_end} done, {rate:.0} feat/s"
                            );
                            last_log = std::time::Instant::now();
                            done = 0;
                        }
                    }
                }
                out
            }));
        }
        handles.into_iter().map(|h| h.join().expect("down_meta worker panicked")).collect()
    });

    // Reassemble global order: thread t owns mats {t, t+n, ...}; each worker's
    // output list is in its own mat order, so place each run of records at its
    // matrices' global offsets.
    let mut metas: Vec<Option<crate::FeatureMeta>> = (0..total).map(|_| None).collect();
    for (t, worker_out) in results.into_iter().enumerate() {
        let mut cursor = 0usize;
        for mi in (t..down_matrices.len()).step_by(n_threads) {
            let nf = down_matrices[mi].shape()[1];
            let base = offsets[mi];
            for i in 0..nf {
                metas[base + i] = worker_out[cursor + i].clone();
            }
            cursor += nf;
        }
        debug_assert_eq!(cursor, worker_out.len());
    }
    eprintln!(
        "  down_meta: layer {layer} complete: {total} features in {:.1}s ({:.0} feat/s, {n_threads} threads)",
        started.elapsed().as_secs_f64(),
        total as f64 / started.elapsed().as_secs_f64().max(1e-9),
    );
    metas
}

/// The exact tensor keys the GATE stage reads — pre-marked consumed on
/// stage-resume so consume-shards can still release shards that hold only
/// gate tensors.
fn gate_stage_keys(
    arch: &dyn larql_models::ModelArchitecture,
    num_layers: usize,
    prefixes: &[&str],
) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    for layer in 0..num_layers {
        match arch.expert_format() {
            larql_models::ExpertFormat::PackedBf16 => {
                if let Some(k) = arch.packed_experts_gate_up_key(layer) {
                    keys.push(k);
                }
            }
            larql_models::ExpertFormat::PackedMxfp4 => {
                for k in [arch.packed_gate_up_blocks_key(layer), arch.packed_gate_up_scales_key(layer)]
                    .into_iter()
                    .flatten()
                {
                    keys.push(k);
                }
            }
            _ if arch.is_moe() && arch.num_experts() > 0 => {
                for expert in 0..arch.num_experts() {
                    if let Some(k) = arch.expert_ffn_gate_key(layer, expert) {
                        keys.push(k);
                    }
                }
            }
            _ => keys.push(arch.ffn_gate_key(layer)),
        }
    }
    keys.iter().map(|k| normalize_key(k, prefixes)).collect()
}

/// The exact tensor keys the ROUTER stage reads (router weights + probed bias).
fn router_stage_keys(
    arch: &dyn larql_models::ModelArchitecture,
    num_layers: usize,
    prefixes: &[&str],
) -> Vec<String> {
    let mut keys: Vec<String> = Vec::new();
    if arch.is_moe() {
        for layer in 0..num_layers {
            if let Some(router_key) = arch.moe_router_key(layer) {
                keys.push(router_key.clone());
                keys.push(router_key.replace(".weight", ".bias"));
            }
        }
    }
    keys.iter().map(|k| normalize_key(k, prefixes)).collect()
}
