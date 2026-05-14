export type FrontendConfig = {
  backend_command: string[];
  cwd?: string;
};

export type ModelType = 'lm' | 'vlm' | 'rlm';

export type BackendEvent =
  | { type: 'ready'; data?: { model_types: string[] } }
  | { type: 'model_list'; data?: ModelSnapshot[] }
  | { type: 'model_loaded'; message: string }
  | { type: 'token'; message: string }
  | { type: 'done'; message?: string }
  | { type: 'error'; message: string }
  | { type: 'metrics'; data?: Record<string, unknown> }
  | { type: 'eda_result'; data?: DatasetEntry[] }
  | { type: 'eda_stats_result'; data?: DatasetStats }
  | { type: 'eda_sample_result'; data?: SampleResult }
  | { type: 'bench_result'; data?: Record<string, unknown> }
  | { type: 'train_log'; message: string }
  | { type: 'train_status'; data?: Record<string, unknown> }
  | { type: 'shutdown' };

export type FrontendRequest = {
  type:
    | 'discover'
    | 'load_model'
    | 'generate'
    | 'stop_generation'
    | 'eda_scan'
    | 'eda_stats'
    | 'eda_sample'
    | 'train_launch'
    | 'train_stop'
    | 'train_status'
    | 'bench_run'
    | 'cancel'
    | 'shutdown';
  model_type?: ModelType;
  weight?: string;
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  intermediate_size?: number;
  use_moe?: boolean;
  dim?: number;
  prompt?: string;
  image_path?: string;
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  data_dir?: string;
  file_path?: string;
  index?: number;
  bench_name?: string;
  train_cmd?: string[];
};

export type ModelSnapshot = {
  name: string;
  path: string;
  type: ModelType;
  hidden_size: number;
  size_mb: number;
};

export type DatasetEntry = {
  path: string;
  name: string;
  type: 'jsonl' | 'parquet';
  size_mb: number;
};

export type DatasetStats = {
  rows?: number;
  keys?: string[];
  avg_text_length?: number;
  max_text_length?: number;
  min_text_length?: number;
  size_mb?: number;
  format?: string;
  columns?: string[];
  num_columns?: number;
  error?: string;
};

export type SampleResult = {
  index?: number;
  data?: Record<string, unknown>;
  error?: string;
};

export type BenchmarkResult = {
  benchmark: string;
  model_type: string;
  weight: string;
  status: string;
  loss?: number;
  perplexity?: number;
  accuracy?: number;
};
