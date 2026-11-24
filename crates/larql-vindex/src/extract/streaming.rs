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

    // ── 1. Gate vectors (streaming, one layer at a time) ──
    callbacks.on_stage("gate_vectors");
    let gate_path = output_dir.join("gate_vectors.bin");
    let mut gate_file = BufWriter::new(std::fs::File::create(&gate_path)?);
    let mut layer_infos: Vec<VindexLayerInfo> = Vec::new();
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

    // ── 1b. Router weights (MoE models only) ──
    if is_moe {
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
    }

    // ── 2. Embeddings ──
    callbacks.on_stage("embeddings");
    let embed_key = normalize_key(arch.embed_key(), prefixes);
    let embed = get_tensor_f32(&shard_store, &tensor_index, &embed_key, &mut consumed)?
        .ok_or_else(|| VindexError::MissingTensor(embed_key.clone()))?;
    let vocab_size = embed.shape()[0];
    let embed_data = embed.as_slice().unwrap();
    let embed_bytes = crate::config::dtype::encode_floats(embed_data, dtype);
    std::fs::write(output_dir.join("embeddings.bin"), &embed_bytes)?;
    callbacks.on_stage_done("embeddings", 0.0);

    // ── 3. Down meta (streaming) ──
    callbacks.on_stage("down_meta");
    let mut all_down_meta: Vec<Option<Vec<Option<crate::FeatureMeta>>>> = vec![None; num_layers];

    // Build whole-word vocab once
    let (_ww_ids, _ww_embed) = super::build::build_whole_word_vocab(tokenizer, &embed, vocab_size, hidden_size);

    for (layer, layer_down_meta) in all_down_meta.iter_mut().enumerate().take(num_layers) {
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
                    let mut scores: Vec<(usize, f32)> = col.iter().copied().enumerate().collect();
                    let k = down_top_k.min(scores.len());
                    if k > 0 && k < scores.len() {
                        scores.select_nth_unstable_by(k, |a, b| b.1.partial_cmp(&a.1).unwrap());
                    }
                    scores.truncate(k);
                    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                    let top_k_entries: Vec<larql_models::TopKEntry> = scores.into_iter()
                        .filter_map(|(idx, logit)| {
                            tokenizer.decode(&[idx as u32], true).ok()
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

                    let feat_idx = feature_offset + feat;
                    if layer_down_meta.is_none() {
                        *layer_down_meta = Some(Vec::new());
                    }
                    if let Some(ref mut metas) = layer_down_meta {
                        while metas.len() <= feat_idx { metas.push(None); }
                        metas[feat_idx] = Some(crate::FeatureMeta {
                            top_token, top_token_id, c_score, top_k: top_k_entries,
                        });
                    }
                }
            }
            feature_offset += num_features;
        }

        callbacks.on_layer_done("down", layer, start.elapsed().as_secs_f64() * 1000.0);
    }

    crate::format::down_meta::write_binary(output_dir, &all_down_meta, down_top_k)?;
    callbacks.on_stage_done("down_meta", 0.0);

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
