//! PackedBf16 load-branch tests (Gemma 4 fused experts, streaming slices,
//! v_shares_k honored) against a tiny synthetic checkpoint — real safetensors,
//! zero network access. Fixture: tests-fixtures/gemma4-fixture (single source
//! of truth for the checkpoint AND its expected values).

use std::path::PathBuf;

use gemma4_fixture::{write_checkpoint, write_plan, FixtureSpec};
use ndarray::Array2;
use safetensors::Dtype;

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("larql_packedbf16_load_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn decode_bf16_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

fn as_arr(rows: usize, cols: usize, data: &[f32]) -> Array2<f32> {
    Array2::from_shape_vec((rows, cols), data.to_vec()).unwrap()
}

/// The load branch materializes per-expert slices `layers.{i}.experts.gate_up_proj.{e}`
/// ([2*mi, hidden]) and `layers.{i}.experts.down_proj.{e}` ([hidden, mi]); every
/// slice must equal the fixture generator's byte-exact expectation, and the rest
/// of the model (embed, router, dense mlp, norms, attention) must load normally.
#[test]
fn packedbf16_load_materializes_per_expert_slices() {
    let spec = FixtureSpec::default();
    let model_dir = temp_dir("slices");
    let _shards = write_checkpoint(&model_dir, &spec, 2).expect("fixture");

    let weights = larql_models::load_model_dir(&model_dir).expect("load");

    // Architecture surface.
    assert_eq!(weights.arch.family(), "gemma4");
    assert_eq!(weights.arch.expert_format(), larql_models::ExpertFormat::PackedBf16);
    assert_eq!(weights.arch.num_experts(), 4);
    assert_eq!(weights.arch.moe_intermediate_size(), 16);
    assert!(weights.arch.v_shares_k(1), "layer 1 is the global layer");
    assert!(!weights.arch.v_shares_k(0), "layer 0 is sliding");

    // Embedding + lm_head (tie_word_embeddings → lm_head = embed).
    let expected_embed = as_arr(spec.vocab_size, spec.hidden_size, &gemma4_fixture::expected_embed(&spec));
    assert_eq!(weights.embed, expected_embed);
    assert_eq!(weights.vocab_size, spec.vocab_size);

    // Per-expert slices — exact equality against the generator's own math.
    for layer in 0..spec.num_layers {
        for expert in 0..spec.num_experts {
            let gu_key = format!("layers.{layer}.experts.gate_up_proj.{expert}");
            let got = weights
                .tensors
                .get(&gu_key)
                .unwrap_or_else(|| panic!("missing {gu_key}"));
            let want = as_arr(
                2 * spec.moe_intermediate,
                spec.hidden_size,
                &gemma4_fixture::expected_gate_up_expert(&spec, layer, expert),
            );
            assert_eq!(got, &want, "{gu_key} must match the fixture bytes");

            let dn_key = format!("layers.{layer}.experts.down_proj.{expert}");
            let got = weights
                .tensors
                .get(&dn_key)
                .unwrap_or_else(|| panic!("missing {dn_key}"));
            let want = as_arr(
                spec.hidden_size,
                spec.moe_intermediate,
                &gemma4_fixture::expected_down_expert(&spec, layer, expert),
            );
            assert_eq!(got, &want, "{dn_key} must match the fixture bytes");
        }
    }

    // Expert-major ordering sanity: expert 1's slice must differ from expert 0's.
    assert_ne!(
        weights.tensors["layers.0.experts.gate_up_proj.0"],
        weights.tensors["layers.0.experts.gate_up_proj.1"]
    );

    // Router + dense mlp + attention + norms all present.
    assert!(weights.tensors.contains_key("layers.0.router.proj.weight"));
    assert!(weights.vectors.contains_key("layers.0.router.scale"));
    assert!(weights.vectors.contains_key("layers.0.router.per_expert_scale"));
    assert!(weights.tensors.contains_key("layers.1.mlp.gate_proj.weight"));
    assert!(weights.tensors.contains_key("layers.1.self_attn.k_proj.weight"));
    assert!(weights.vectors.contains_key("layers.0.input_layernorm.weight"));
    assert!(weights.vectors.contains_key("layers.0.layer_scalar"));

    // v_shares_k truth table in the loaded map: v_proj on sliding only.
    assert!(weights.tensors.contains_key("layers.0.self_attn.v_proj.weight"));
    assert!(
        !weights.tensors.contains_key("layers.1.self_attn.v_proj.weight"),
        "global layer has no v_proj tensor and must load without MissingTensor"
    );

    let _ = std::fs::remove_dir_all(&model_dir);
}

/// A fixture variant with the sliding layer's v_proj removed must FAIL the
/// load with MissingTensor("layers.0.self_attn.v_proj.weight") — sliding
/// layers require v_proj.
#[test]
fn packedbf16_load_requires_v_proj_on_sliding_layer() {
    let spec = FixtureSpec::default();
    let model_dir = temp_dir("missing_v_sliding");
    let filtered: Vec<_> = gemma4_fixture::tensor_plan(&spec)
        .into_iter()
        .filter(|t| t.name != "model.language_model.layers.0.self_attn.v_proj.weight")
        .collect();
    let _shards = write_plan(&model_dir, &spec, &filtered, 2).expect("fixture");

    let err = match larql_models::load_model_dir(&model_dir) {
        Err(e) => e,
        Ok(_) => panic!("load must fail: sliding layer's v_proj is missing"),
    };
    match err {
        larql_models::ModelError::MissingTensor(key) => {
            assert_eq!(key, "layers.0.self_attn.v_proj.weight");
        }
        other => panic!("expected MissingTensor, got: {other:?}"),
    }

    let _ = std::fs::remove_dir_all(&model_dir);
}

/// Dropping the GLOBAL layer's attention tensors entirely (no v_proj is the
/// normal case) still loads: global layers must never trigger a v_proj lookup.
#[test]
fn packedbf16_load_global_layer_absent_v_proj_ok() {
    let spec = FixtureSpec::default();
    let model_dir = temp_dir("global_no_v");
    let filtered: Vec<_> = gemma4_fixture::tensor_plan(&spec)
        .into_iter()
        .filter(|t| {
            // Strip every layer-1 attention tensor: the extreme case of
            // v_proj absence must not error either.
            !t.name.starts_with("model.language_model.layers.1.self_attn.")
        })
        .collect();
    let _shards = write_plan(&model_dir, &spec, &filtered, 1).expect("fixture");

    let weights = larql_models::load_model_dir(&model_dir).expect("load must succeed");
    assert_eq!(weights.tensors["layers.1.experts.down_proj.3"].shape(), &[64, 16]);
    let _ = std::fs::remove_dir_all(&model_dir);
}

/// dtype provenance: the fixture is BF16; the f32 slices must be exact bf16
/// decodes (no rounding surprises) — spot-check one byte-adjacent pair.
#[test]
fn packedbf16_load_decodes_bf16_exactly() {
    let spec = FixtureSpec::default();
    let model_dir = temp_dir("bf16_exact");
    let plan = gemma4_fixture::tensor_plan(&spec);
    let gu = plan
        .iter()
        .find(|t| t.name == "model.language_model.layers.0.experts.gate_up_proj")
        .unwrap();
    assert_eq!(gu.dtype, Dtype::BF16);
    let _shards = write_plan(&model_dir, &spec, &plan, 2).expect("fixture");

    let weights = larql_models::load_model_dir(&model_dir).unwrap();
    let expert0_raw = decode_bf16_f32(&gu.data[0..2 * spec.moe_intermediate * spec.hidden_size * 2]);
    let loaded = weights.tensors["layers.0.experts.gate_up_proj.0"].as_slice().unwrap();
    assert_eq!(expert0_raw, loaded);
    let _ = std::fs::remove_dir_all(&model_dir);
}
