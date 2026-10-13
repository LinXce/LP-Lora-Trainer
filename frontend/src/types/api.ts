/**
 * Wire types for the local backend (`/api/v1`).
 * Enum values mirror `app/schemas/engine.py` and `app/schemas/training.py`.
 */

export type ManagementMode = 'user_managed' | 'application_managed' | 'external_reference'
export type InstallationState = 'discovered' | 'preparing' | 'ready' | 'failed' | 'missing'
export type VerificationState = 'unverified' | 'experimental' | 'verified'

export type TaskState =
  | 'draft'
  | 'validating'
  | 'queued'
  | 'preparing'
  | 'running'
  | 'stopping'
  | 'stopped'
  | 'succeeded'
  | 'failed'
  | 'connection_lost'

/** Loaded from local adapters/<engine>/adapter.py, never a frontend whitelist. */
export interface AdapterInfo {
  engine_id: string
  display_name: string
  adapter_version: string
  submittable: boolean
  environment_strategy: string
  installation_help: string
  installation_source_help: string
  default_installation_source: string
  /** How a PyTorch/CUDA wheel mirror is applied: named uv index, extra env index, or none. */
  torch_index_mode?: string
  /** Upstream reference a fresh clone should land on, declared by the adapter. */
  clone_url?: string
  clone_branch?: string
  clone_commit?: string
}

export interface AdapterCatalog {
  adapters: AdapterInfo[]
  errors: { directory: string; message: string }[]
}

export interface EngineRevision {
  source: string | null
  requested_ref: string | null
  commit: string | null
  fingerprint: string | null
}

export interface EngineInstallation {
  installation_id: string
  /** Adapter-detected engine type, e.g. `kohya`, `ai_toolkit`. Null when detection is ambiguous. */
  engine_id: string | null
  /** Instance label; the directory name under `engine/` by default. Not an identity. */
  label: string
  source_path: string
  management_mode: ManagementMode
  revision: EngineRevision
  state: InstallationState
  verification: VerificationState
  environment_id: string | null
  python_executable: string | null
  /** Existing Python interpreters detected inside this engine clone. */
  python_candidates?: string[]
  /** Installation sources declared by the adapter for this particular clone. */
  installation_sources?: string[]
  python_version: string | null
  torch_version: string | null
  cuda_wheel: string | null
  is_default: boolean
  /** Number of tasks pinned to this installation. */
  referenced_by: number
  /** Adapters that matched when detection is ambiguous. */
  candidate_engines: string[]
  issues: string[]
  /** Adapter-provided completion step for incomplete source, or null when complete. */
  source_repair?: string | null
  discovered_at: string
}

export interface InstallationSession {
  session_id: string
  /** 'clone' for a git pull into engine/; 'repair' for completing fetched source. */
  kind?: 'clone' | 'repair'
  repository_url?: string
  ref?: string | null
  installation_id: string
  label: string
  engine_id: string
  title: string
  cwd: string
  python_executable: string
  commands: string[][]
  command_index: number | null
  state: 'queued' | 'running' | 'stopping' | 'verifying' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted'
  started_at: string
  finished_at: string | null
  error: string | null
  diagnostic_error: string | null
  /** Selected PyPI index; null/absent uses the official installer default. */
  mirror_url?: string | null
  /** Notices about the run itself, e.g. the official-source fallback. */
  warnings?: SessionNotice[]
  /** Whether this run used the configured proxy (never the URL itself). */
  proxy?: boolean
}

export interface TerminalLogChunk {
  text: string
  offset: number
}

/** A title/detail/suggestion block; used for failure hints and run warnings. */
export interface SessionNotice {
  id: string
  title: string
  detail: string
  suggestion: string
  /** Matching log lines that triggered it, already trimmed. */
  evidence: string[]
}

export interface SessionDiagnosis {
  session_id: string
  state: InstallationSession['state']
  hints: SessionNotice[]
}

export interface EngineCloneOptions {
  url: string
  /** Folder under engine/; empty uses the repository name. */
  name?: string | null
  /** Branch or tag; empty clones the remote default branch. */
  ref?: string | null
  /** Commit to check out after cloning; pins the fetched version. */
  commit?: string | null
  /** Per-pull override; omitted follows the saved proxy setting. */
  use_proxy?: boolean
}

export interface InstallEnvironmentOptions {
  confirmed: true
  python_executable?: string
  torch_source: string
  mirror_url?: string | null
  /** Optional named PyTorch/CUDA wheel index override. */
  torch_mirror_url?: string | null
  /** Optional base URL for uv managed-Python downloads (defaults to GitHub). */
  python_mirror_url?: string | null
  /** Per-install override; omitted follows the saved proxy setting. */
  use_proxy?: boolean
}

export interface TaskProgress {
  /** Every field is null when the adapter cannot parse it reliably — display as unknown. */
  step: number | null
  total_steps: number | null
  epoch: number | null
  loss: number | null
  it_per_sec: number | null
  eta_seconds: number | null
}

export interface RecoveryCapabilities {
  graceful_stop: boolean
  resume_from_weights: boolean
  resume_full_state: boolean
}

export interface TaskSummary {
  history_deleted?: boolean
  task_id: string
  name: string
  state: TaskState
  engine_id: string
  installation_id: string
  installation_label: string
  architecture: string
  dataset_name: string | null
  output_dir: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  progress: TaskProgress
  error_summary: string | null
  recovery: RecoveryCapabilities
}

export interface MetricPoint {
  step: number
  loss: number
}

export interface DatasetIssueCount {
  kind: 'corrupt' | 'missing_caption' | 'odd_size' | 'duplicate'
  count: number
}

export interface Dataset {
  dataset_id: string
  name: string
  path: string
  image_count: number | null
  caption_count: number | null
  issues: DatasetIssueCount[]
  scanned_at: string | null
}

export interface DatasetImage {
  image_id: string
  file_name: string
  width: number
  height: number
  thumbnail_url: string | null
  caption: string | null
  issues: DatasetIssueCount['kind'][]
}

export interface Page<T> {
  items: T[]
  total: number
  offset: number
  limit: number
}

export interface Artifact {
  artifact_id: string
  task_id: string
  task_name: string
  kind: 'checkpoint' | 'sample'
  file_name: string
  path: string
  step: number | null
  /** The engine's final save: it carries no step in its name. */
  final?: boolean
  size_bytes: number | null
  /** False while the engine is still writing the file. */
  complete: boolean
  created_at: string
  preview_url: string | null
  architecture: string
  base_model: string | null
}

export interface GpuStatus {
  name: string
  memory_used_mb: number
  memory_total_mb: number
  utilization: number | null
  temperature_c: number | null
  memory_utilization: number | null
  power_draw_w: number | null
  power_limit_w: number | null
  fan_speed_percent: number | null
}

export interface SystemStatus {
  backend_version: string
  installation_workflow_version: number
  supervisor: 'running' | 'idle' | 'unreachable'
  /** Null when GPU monitoring is disabled or unavailable. */
  gpu: GpuStatus | null
}

export interface AppSettings {
  engine_root: string
  data_root: string
  comfyui_lora_dir: string | null
  gpu_monitor: boolean
  log_tail_lines: number
  /** HTTP(S) proxy for engine downloads; may contain credentials. */
  proxy_url: string | null
  proxy_enabled: boolean
}

/* ---------- Training form ---------- */

export type ParamGroup = 'basic' | 'advanced' | 'preview' | 'native'

export interface ParamSpec {
  key: string
  /** Native engine/config key used when importing or rendering engine files. */
  native_key?: string
  /** Native configuration block (for nested engine configs). */
  block?: string
  /** False for fields consumed by dataset generation or pipeline setup. */
  native?: boolean
  label: string
  type: 'int' | 'float' | 'string' | 'bool' | 'enum'
  group: ParamGroup
  /** Additional tabs where this same parameter is also displayed. */
  groups?: ParamGroup[]
  section: string
  /** Lower values are placed earlier within the parameter form. */
  display_order?: number
  default: string | number | boolean | null
  min?: number
  max?: number
  step?: number
  options?: { value: string; label: string }[]
  help?: string
  /** When set, the parameter is only shown for these base-model types. Empty/absent = all. */
  architectures?: string[]
  /** When set the parameter is shown disabled with this explanation. */
  unsupported_reason?: string
  /** Engine argument is a list (nargs="*"): entries may be separated by spaces or newlines. */
  list_arg?: boolean
  /** Render a multi-line input (list arguments, resolution sets, …). */
  multiline?: boolean
  /** Architectures for which a non-empty value is required before launch. */
  required_for?: string[]
  /** Square-resolution presets backed by the same multiline text value. */
  resolution_presets?: number[]
  /** Local path declared by the adapter; remote names/URLs are plain text. */
  path_kind?: 'file' | 'directory' | 'file_or_directory'
  file_types?: string[]
}

/** The staged training form as persisted by the backend (shared by all clients). */
export interface StoredTrainingDraft {
  draft: TrainingDraft | null
  params_by_architecture: Record<string, Record<string, string | number | boolean | null>>
  updated_at: string | null
}

export interface EngineCapabilities {
  installation_id: string
  architectures: string[]
  params: ParamSpec[]
  /** False when the engine only supports configuration preview, not submission. */
  submittable?: boolean
}

/** A base-model type ("底模类型") and the engines that can train it. */
export interface BaseModel {
  id: string
  label: string
  engines: string[]
}

/** Best-effort local hardware probe (never on the polled status path). */
export interface HardwareInfo {
  os: string
  arch: string
  gpu_name: string | null
  driver_version: string | null
  /** The driver's maximum supported CUDA runtime. */
  cuda_version: string | null
  compute_cap: string | null
  cuda_usable: boolean
}

export interface InstallOption {
  value: string
  label: string
  url: string | null
}

/** Machine-aware install recommendation for one engine installation. */
export interface InstallPlan {
  installation_id: string
  engine_id: string | null
  supported: boolean
  blocked_reason: string | null
  engine_type_confirmed: boolean
  engine_decides_cuda: boolean
  torch_source: string
  torch_source_options: string[]
  torch_source_reason: string
  cuda_version: string | null
  cuda_tag: string | null
  os: string
  gpu_name: string | null
  driver_version: string | null
  compute_cap: string | null
  mirror_preset: string
  mirror_presets: InstallOption[]
  torch_mirror_preset: string
  torch_mirror_presets: InstallOption[]
  python_mirror_preset: string
  python_mirror_presets: InstallOption[]
}

export interface TaskConfiguration extends TrainingDraft {
  format: 'lp-training-parameters'
  version: 1
  engine_id: string
  dataset_path: string | null
}

export interface TrainingDraft {
  name: string
  output_name: string
  installation_id: string
  architecture: string
  base_model_path: string
  dataset_id: string
  output_dir: string
  params: Record<string, string | number | boolean | null>
}

export interface TrainingExportSource extends TrainingDraft {
  engine_id: string | null
  dataset_path: string | null
}

export interface PromptTemplate {
  template_id: string
  name: string
  prompt: string
  created_at: string
  updated_at: string
}
export interface PromptTemplateInput { name: string; prompt: string }

export interface NativeParameterExport {
  engine_id: string
  engine_name: string
  native_format: 'toml' | 'yaml' | 'json'
  text: string
  warnings: string[]
  files?: { filename: string; text: string }[]
}

export interface ValidationIssue {
  field: string | null
  level: 'error' | 'warning'
  message: string
}
export interface ValidationResult {
  ok: boolean
  issues: ValidationIssue[]
  native_config: string | null
  native_format: 'toml' | 'yaml' | 'json' | null
  argv: string[] | null
  /** Ordered commands the supervisor will run (caching stages first, training last). */
  pipeline: string[][] | null
  /** False when the selected engine only supports configuration preview. */
  submittable: boolean
}

/* ---------- Events (SSE) ---------- */

export type ServerEvent =
  | { kind: 'task.deleted'; task_id: string }
  | { kind: 'task.updated'; task: TaskSummary }
  | { kind: 'task.metrics'; task_id: string; points: MetricPoint[] }
  | { kind: 'task.log'; task_id: string; lines: string[] }
  | { kind: 'engine.updated'; installation: EngineInstallation }
  | { kind: 'system.status'; status: SystemStatus }
