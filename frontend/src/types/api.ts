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
  python_version: string | null
  torch_version: string | null
  cuda_wheel: string | null
  is_default: boolean
  /** Number of tasks pinned to this installation. */
  referenced_by: number
  /** Adapters that matched when detection is ambiguous. */
  candidate_engines: string[]
  issues: string[]
  discovered_at: string
}

export interface InstallationSession {
  session_id: string
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
}

export interface TerminalLogChunk {
  text: string
  offset: number
}

export interface InstallEnvironmentOptions {
  confirmed: true
  python_executable?: string
  torch_source: 'cu124' | 'cu126' | 'cu128' | 'existing'
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
}

export interface SystemStatus {
  backend_version: string
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
}

/* ---------- Training form ---------- */

export type ParamGroup = 'basic' | 'advanced' | 'native'

export interface ParamSpec {
  key: string
  label: string
  type: 'int' | 'float' | 'string' | 'bool' | 'enum'
  group: ParamGroup
  section: string
  default: string | number | boolean | null
  min?: number
  max?: number
  step?: number
  options?: { value: string; label: string }[]
  help?: string
  /** When set the parameter is shown disabled with this explanation. */
  unsupported_reason?: string
}

export interface EngineCapabilities {
  installation_id: string
  architectures: string[]
  params: ParamSpec[]
}

export interface TrainingDraft {
  name: string
  installation_id: string
  architecture: string
  base_model_path: string
  dataset_id: string
  output_dir: string
  params: Record<string, string | number | boolean | null>
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
}

/* ---------- Events (SSE) ---------- */

export type ServerEvent =
  | { kind: 'task.updated'; task: TaskSummary }
  | { kind: 'task.metrics'; task_id: string; points: MetricPoint[] }
  | { kind: 'task.log'; task_id: string; lines: string[] }
  | { kind: 'engine.updated'; installation: EngineInstallation }
  | { kind: 'system.status'; status: SystemStatus }
