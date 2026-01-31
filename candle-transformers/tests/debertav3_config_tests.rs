use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::debertav2::{Config, Id2Label};
use std::collections::HashMap;

// Test-only defaults mirroring the small config used in this file.
const DEFAULT_POSITION_BUCKETS: usize = 8;

fn base_config_json(pos_att_type: &str) -> String {
    format!(
        r#"
    {{
      "model_type": "deberta-v2",
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 8,
      "initializer_range": 0.02,
      "intermediate_size": 32,
      "max_position_embeddings": 16,
      "relative_attention": true,
      "position_buckets": {DEFAULT_POSITION_BUCKETS},
      "norm_rel_ebd": "layer_norm",
      "share_att_key": true,
      "pos_att_type": {pos_att_type},
      "layer_norm_eps": 1e-7,
      "max_relative_positions": -1,
      "position_biased_input": false,
      "num_attention_heads": 2,
      "num_hidden_layers": 2,
      "type_vocab_size": 0,
      "vocab_size": 32
    }}
    "#
    )
}

#[test]
fn deberta_v3_defaults_for_pooler_and_pos_att_type() {
    let config_json = base_config_json("\"p2c|c2p\"");

    let config: Config = serde_json::from_str(&config_json).expect("config should parse");

    assert_eq!(
        config.pos_att_type,
        Some(vec!["p2c".to_string(), "c2p".to_string()])
    );
    assert_eq!(config.pooler_hidden_size, None);
    assert_eq!(config.pooler_dropout, None);
    assert_eq!(config.pooler_hidden_act, None);

    let device = Device::Cpu;
    let mut tensors = HashMap::new();
    tensors.insert(
        "embeddings.word_embeddings.weight".to_string(),
        Tensor::zeros((config.vocab_size, config.hidden_size), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "embeddings.LayerNorm.weight".to_string(),
        Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "embeddings.LayerNorm.bias".to_string(),
        Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
    );
    let rel_embedding_size = config
        .relative_embeddings_size()
        .expect("relative embeddings size should be available with relative attention");
    assert_eq!(rel_embedding_size, DEFAULT_POSITION_BUCKETS * 2);
    tensors.insert(
        "encoder.rel_embeddings.weight".to_string(),
        Tensor::zeros((rel_embedding_size, config.hidden_size), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "encoder.LayerNorm.weight".to_string(),
        Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "encoder.LayerNorm.bias".to_string(),
        Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "pooler.dense.weight".to_string(),
        Tensor::zeros((config.hidden_size, config.hidden_size), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "pooler.dense.bias".to_string(),
        Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "classifier.weight".to_string(),
        Tensor::zeros((2, config.hidden_size), DType::F32, &device).unwrap(),
    );
    tensors.insert(
        "classifier.bias".to_string(),
        Tensor::zeros(2, DType::F32, &device).unwrap(),
    );

    for layer_idx in 0..config.num_hidden_layers {
        let prefix = format!("encoder.layer.{layer_idx}");
        tensors.insert(
            format!("{prefix}.attention.self.query_proj.weight"),
            Tensor::zeros((config.hidden_size, config.hidden_size), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.attention.self.query_proj.bias"),
            Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.attention.self.key_proj.weight"),
            Tensor::zeros((config.hidden_size, config.hidden_size), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.attention.self.key_proj.bias"),
            Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.attention.self.value_proj.weight"),
            Tensor::zeros((config.hidden_size, config.hidden_size), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.attention.self.value_proj.bias"),
            Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.attention.output.dense.weight"),
            Tensor::zeros((config.hidden_size, config.hidden_size), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.attention.output.dense.bias"),
            Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.attention.output.LayerNorm.weight"),
            Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.attention.output.LayerNorm.bias"),
            Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.intermediate.dense.weight"),
            Tensor::zeros((config.intermediate_size, config.hidden_size), DType::F32, &device)
                .unwrap(),
        );
        tensors.insert(
            format!("{prefix}.intermediate.dense.bias"),
            Tensor::zeros(config.intermediate_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.output.dense.weight"),
            Tensor::zeros((config.hidden_size, config.intermediate_size), DType::F32, &device)
                .unwrap(),
        );
        tensors.insert(
            format!("{prefix}.output.dense.bias"),
            Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.output.LayerNorm.weight"),
            Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.output.LayerNorm.bias"),
            Tensor::zeros(config.hidden_size, DType::F32, &device).unwrap(),
        );
    }

    let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
    let id_to_label: Id2Label = [(0u32, "safe".to_string()), (1, "unsafe".to_string())]
        .into_iter()
        .collect();
    let model = candle_transformers::models::debertav2::DebertaV2SeqClassificationModel::load(
        vb,
        &config,
        Some(id_to_label),
    )
    .expect("model should load with defaults");

    assert!(model.device.is_cpu());
    assert_eq!(config.num_hidden_layers, 2);
    assert_eq!(config.hidden_size, 8);
}

#[test]
fn deberta_v3_pos_att_type_variants() {
    let config_missing = format!(
        r#"
    {{
      "model_type": "deberta-v2",
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 8,
      "initializer_range": 0.02,
      "intermediate_size": 32,
      "max_position_embeddings": 16,
      "relative_attention": true,
      "position_buckets": {DEFAULT_POSITION_BUCKETS},
      "norm_rel_ebd": "layer_norm",
      "share_att_key": true,
      "layer_norm_eps": 1e-7,
      "max_relative_positions": -1,
      "position_biased_input": false,
      "num_attention_heads": 2,
      "num_hidden_layers": 2,
      "type_vocab_size": 0,
      "vocab_size": 32
    }}
    "#
    );

    let config: Config = serde_json::from_str(&config_missing).expect("config should parse");
    assert!(config.pos_att_type.is_none());

    let config_json = base_config_json("[\"p2c\", \"c2p\"]");
    let config: Config = serde_json::from_str(&config_json).expect("config should parse");
    assert_eq!(
        config.pos_att_type,
        Some(vec!["p2c".to_string(), "c2p".to_string()])
    );
}

#[test]
fn deberta_v3_norm_rel_ebd_normalization() {
    let config_json = r#"
    {
      "model_type": "deberta-v2",
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 8,
      "initializer_range": 0.02,
      "intermediate_size": 32,
      "max_position_embeddings": 16,
      "relative_attention": true,
      "position_buckets": 8,
      "norm_rel_ebd": "||layer_norm|",
      "share_att_key": true,
      "pos_att_type": "p2c|c2p",
      "layer_norm_eps": 1e-7,
      "max_relative_positions": -1,
      "position_biased_input": false,
      "num_attention_heads": 2,
      "num_hidden_layers": 2,
      "type_vocab_size": 0,
      "vocab_size": 32
    }
    "#;

    let config: Config = serde_json::from_str(config_json).expect("config should parse");
    let normalized = config
        .norm_rel_ebd
        .as_ref()
        .map(|value| {
            value
                .split('|')
                .map(|entry| entry.trim())
                .filter(|entry| !entry.is_empty())
                .collect::<Vec<_>>()
                .join("|")
        })
        .unwrap_or_default();
    assert_eq!(normalized, "layer_norm");
}
