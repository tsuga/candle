use candle_transformers::models::debertav2::{Config, HiddenAct};

#[test]
fn deberta_v3_defaults_for_pooler_and_pos_att_type() {
    let config_json = r#"
    {
      "model_type": "deberta-v2",
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "max_position_embeddings": 512,
      "relative_attention": true,
      "position_buckets": 256,
      "norm_rel_ebd": "layer_norm",
      "share_att_key": true,
      "pos_att_type": "p2c|c2p",
      "layer_norm_eps": 1e-7,
      "max_relative_positions": -1,
      "position_biased_input": false,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "type_vocab_size": 0,
      "vocab_size": 128100
    }
    "#;

    let config: Config = serde_json::from_str(config_json).expect("config should parse");

    assert_eq!(config.pos_att_type, vec!["p2c".to_string(), "c2p".to_string()]);
    assert_eq!(config.pooler_hidden_size, None);
    assert_eq!(config.pooler_dropout, None);
    assert_eq!(config.pooler_hidden_act, None);

    let pooler_hidden_size = config.pooler_hidden_size.unwrap_or(config.hidden_size);
    let pooler_dropout = config.pooler_dropout.unwrap_or(0.0);
    let pooler_hidden_act = config.pooler_hidden_act.unwrap_or(HiddenAct::Gelu);

    assert_eq!(pooler_hidden_size, 768);
    assert_eq!(pooler_dropout, 0.0);
    assert_eq!(pooler_hidden_act, HiddenAct::Gelu);
}
