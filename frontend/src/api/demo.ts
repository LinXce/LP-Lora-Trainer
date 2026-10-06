/**
 * Sample data for UI development only. Enabled with `?demo` in a dev build;
 * production builds never enter this mode. Every screen shows a banner while
 * active and all write operations are rejected.
 */
import type {
  AppSettings,
  Artifact,
  Dataset,
  DatasetImage,
  EngineCapabilities,
  EngineInstallation,
  MetricPoint,
  Page,
  SystemStatus,
  TaskSummary,
  TrainingDraft,
  ValidationResult,
} from '@/types/api'

export function isDemoMode(): boolean {
  if (!import.meta.env.DEV) return false
  const q = window.location.search + window.location.hash
  return /[?&]demo\b/.test(q)
}

const now = Date.now()
const iso = (minutesAgo: number) => new Date(now - minutesAgo * 60_000).toISOString()

const engines: EngineInstallation[] = [
  {
    installation_id: 'inst-kohya-a1',
    engine_id: 'kohya',
    label: 'kohya_ss',
    source_path: 'D:\\SDaki\\LP-Lora-Trainer\\engine\\kohya_ss',
    management_mode: 'user_managed',
    revision: { source: 'git', requested_ref: 'v25.0.3', commit: '8f2c1d9e4b7a', fingerprint: null },
    state: 'ready',
    verification: 'experimental',
    environment_id: 'env-kohya-py310-cu124',
    python_executable: 'D:\\SDaki\\LP-Lora-Trainer\\data\\envs\\kohya\\py310-cu124\\Scripts\\python.exe',
    python_version: '3.10.11',
    torch_version: '2.5.1',
    cuda_wheel: 'cu124',
    is_default: true,
    referenced_by: 3,
    candidate_engines: [],
    issues: [],
    discovered_at: iso(60 * 24 * 3),
  },
  {
    installation_id: 'inst-kohya-old',
    engine_id: 'kohya',
    label: 'kohya_ss-old',
    source_path: 'D:\\SDaki\\LP-Lora-Trainer\\engine\\kohya_ss-old',
    management_mode: 'user_managed',
    revision: { source: null, requested_ref: null, commit: null, fingerprint: 'sha256:5e01a7…c2' },
    state: 'discovered',
    verification: 'unverified',
    environment_id: null,
    python_executable: null,
    python_version: null,
    torch_version: null,
    cuda_wheel: null,
    is_default: false,
    referenced_by: 0,
    candidate_engines: [],
    issues: ['未选择 Python 解释器', '复制来的 .venv 尚未核实'],
    discovered_at: iso(60 * 5),
  },
  {
    installation_id: 'inst-aitk',
    engine_id: null,
    label: 'ai-toolkit',
    source_path: 'D:\\SDaki\\LP-Lora-Trainer\\engine\\ai-toolkit',
    management_mode: 'user_managed',
    revision: { source: 'git', requested_ref: 'main', commit: '1b9a0c3f22de', fingerprint: null },
    state: 'discovered',
    verification: 'unverified',
    environment_id: null,
    python_executable: null,
    python_version: null,
    torch_version: null,
    cuda_wheel: null,
    is_default: false,
    referenced_by: 0,
    candidate_engines: ['ai_toolkit'],
    issues: ['需要确认引擎类型'],
    discovered_at: iso(30),
  },
]

const tasks: TaskSummary[] = [
  {
    task_id: 'job-0007',
    name: 'character_aki_v3',
    state: 'running',
    engine_id: 'kohya',
    installation_id: 'inst-kohya-a1',
    installation_label: 'kohya_ss',
    architecture: 'SDXL',
    dataset_name: 'aki_portraits',
    output_dir: 'D:\\lora_out\\character_aki_v3',
    created_at: iso(48),
    started_at: iso(46),
    finished_at: null,
    progress: { step: 1240, total_steps: 3000, epoch: 5, loss: 0.0842, it_per_sec: 1.62, eta_seconds: 1086 },
    error_summary: null,
    recovery: { graceful_stop: true, resume_from_weights: true, resume_full_state: false },
  },
  {
    task_id: 'job-0008',
    name: 'style_ink_wash',
    state: 'queued',
    engine_id: 'kohya',
    installation_id: 'inst-kohya-a1',
    installation_label: 'kohya_ss',
    architecture: 'SDXL',
    dataset_name: 'ink_wash_set',
    output_dir: 'D:\\lora_out\\style_ink_wash',
    created_at: iso(20),
    started_at: null,
    finished_at: null,
    progress: { step: null, total_steps: 2400, epoch: null, loss: null, it_per_sec: null, eta_seconds: null },
    error_summary: null,
    recovery: { graceful_stop: true, resume_from_weights: true, resume_full_state: false },
  },
  {
    task_id: 'job-0006',
    name: 'character_aki_v2',
    state: 'succeeded',
    engine_id: 'kohya',
    installation_id: 'inst-kohya-a1',
    installation_label: 'kohya_ss',
    architecture: 'SDXL',
    dataset_name: 'aki_portraits',
    output_dir: 'D:\\lora_out\\character_aki_v2',
    created_at: iso(60 * 26),
    started_at: iso(60 * 26 - 1),
    finished_at: iso(60 * 24),
    progress: { step: 2000, total_steps: 2000, epoch: 8, loss: 0.0791, it_per_sec: 1.58, eta_seconds: 0 },
    error_summary: null,
    recovery: { graceful_stop: true, resume_from_weights: true, resume_full_state: false },
  },
  {
    task_id: 'job-0005',
    name: 'bg_city_night',
    state: 'failed',
    engine_id: 'kohya',
    installation_id: 'inst-kohya-a1',
    installation_label: 'kohya_ss',
    architecture: 'SDXL',
    dataset_name: 'city_night',
    output_dir: 'D:\\lora_out\\bg_city_night',
    created_at: iso(60 * 50),
    started_at: iso(60 * 50 - 1),
    finished_at: iso(60 * 49),
    progress: { step: 312, total_steps: 2400, epoch: 1, loss: 0.1123, it_per_sec: null, eta_seconds: null },
    error_summary: 'torch.OutOfMemoryError: CUDA out of memory (step 312)',
    recovery: { graceful_stop: true, resume_from_weights: true, resume_full_state: false },
  },
]

const datasets: Dataset[] = [
  {
    dataset_id: 'ds-aki',
    name: 'aki_portraits',
    path: 'E:\\datasets\\aki_portraits',
    image_count: 86,
    caption_count: 83,
    issues: [
      { kind: 'missing_caption', count: 3 },
      { kind: 'odd_size', count: 2 },
    ],
    scanned_at: iso(90),
  },
  {
    dataset_id: 'ds-ink',
    name: 'ink_wash_set',
    path: 'E:\\datasets\\ink_wash',
    image_count: 142,
    caption_count: 142,
    issues: [{ kind: 'duplicate', count: 1 }],
    scanned_at: iso(60 * 8),
  },
  {
    dataset_id: 'ds-city',
    name: 'city_night',
    path: 'F:\\photos\\city_night',
    image_count: null,
    caption_count: null,
    issues: [],
    scanned_at: null,
  },
]

const tagPool = ['1girl', 'aki', 'solo', 'looking at viewer', 'short hair', 'upper body', 'smile', 'outdoors', 'jacket', 'night', 'portrait']

function images(id: string, offset: number, limit: number, issue?: string): Page<DatasetImage> {
  const ds = datasets.find((d) => d.dataset_id === id)
  const total = ds?.image_count ?? 0
  let all: DatasetImage[] = Array.from({ length: total }, (_, i) => {
    const missing = i % 31 === 7
    const odd = i % 43 === 11
    const w = odd ? 512 : [832, 896, 1024, 1152][i % 4]
    const h = odd ? 380 : [1216, 1152, 1024, 896][i % 4]
    const tags = tagPool.filter((_, t) => (i + t) % 3 !== 0).slice(0, 6)
    return {
      image_id: `${id}-${i}`,
      file_name: `${String(i + 1).padStart(4, '0')}.png`,
      width: w,
      height: h,
      thumbnail_url: null,
      caption: missing ? null : tags.join(', '),
      issues: [...(missing ? (['missing_caption'] as const) : []), ...(odd ? (['odd_size'] as const) : [])],
    }
  })
  if (issue) all = all.filter((im) => im.issues.includes(issue as DatasetImage['issues'][number]))
  return { items: all.slice(offset, offset + limit), total: all.length, offset, limit }
}

function metrics(id: string): MetricPoint[] {
  const t = tasks.find((x) => x.task_id === id)
  const last = t?.progress.step ?? 0
  const pts: MetricPoint[] = []
  for (let s = 10; s <= last; s += 10) {
    const base = 0.16 * Math.exp(-s / 700) + 0.075
    const noise = Math.sin(s * 0.37) * 0.012 + Math.sin(s * 0.051) * 0.006
    pts.push({ step: s, loss: +(base + noise).toFixed(4) })
  }
  return pts
}

function log(id: string): string[] {
  const t = tasks.find((x) => x.task_id === id)
  if (!t || t.state === 'queued') return []
  const lines = [
    `[supervisor] task ${id} bound to installation ${t.installation_id}`,
    '[supervisor] launching: python sdxl_train_network.py --config_file config.native.toml',
    'prepare tokenizers',
    'loading model for process 0/1',
    'load StableDiffusion checkpoint: D:/models/sdxl/sd_xl_base_1.0.safetensors',
    'building U-Net / text encoders',
    'enable LoRA for U-Net: 722 modules',
    'prepare optimizer, data loader etc.',
    'caching latents... 100%',
    'running training / 学習開始',
    '  num train images * repeats: 860',
    '  num epochs: 8',
  ]
  const step = t.progress.step ?? 0
  for (let s = Math.max(0, step - 300); s <= step; s += 20) {
    lines.push(`steps: ${s}/${t.progress.total_steps}  avr_loss=${(0.08 + Math.sin(s) * 0.01).toFixed(4)}`)
  }
  if (t.error_summary) lines.push('Traceback (most recent call last):', t.error_summary)
  return lines
}

function artifacts(taskId?: string): Artifact[] {
  const out: Artifact[] = []
  for (const t of tasks) {
    if (taskId && t.task_id !== taskId) continue
    const done = t.progress.step ?? 0
    for (let s = 500; s <= done; s += 500) {
      out.push({
        artifact_id: `${t.task_id}-ckpt-${s}`,
        task_id: t.task_id,
        task_name: t.name,
        kind: 'checkpoint',
        file_name: `${t.name}-${String(s).padStart(6, '0')}.safetensors`,
        path: `${t.output_dir}\\${t.name}-${String(s).padStart(6, '0')}.safetensors`,
        step: s,
        size_bytes: 228_501_120,
        complete: true,
        created_at: iso(10),
        preview_url: null,
        architecture: t.architecture,
        base_model: 'sd_xl_base_1.0.safetensors',
      })
      out.push({
        artifact_id: `${t.task_id}-sample-${s}`,
        task_id: t.task_id,
        task_name: t.name,
        kind: 'sample',
        file_name: `sample_${String(s).padStart(6, '0')}_0.png`,
        path: `${t.output_dir}\\sample\\sample_${String(s).padStart(6, '0')}_0.png`,
        step: s,
        size_bytes: 1_402_112,
        complete: true,
        created_at: iso(10),
        preview_url: null,
        architecture: t.architecture,
        base_model: 'sd_xl_base_1.0.safetensors',
      })
    }
  }
  return out
}

function capabilities(id: string): EngineCapabilities {
  return {
    installation_id: id,
    architectures: ['SDXL', 'SD1.5'],
    params: [
      { key: 'max_train_steps', label: '训练步数', type: 'int', group: 'basic', section: '训练', default: 3000, min: 1, max: 200000, step: 100 },
      { key: 'learning_rate', label: '学习率', type: 'float', group: 'basic', section: '训练', default: 0.0001, min: 0, step: 0.00001 },
      { key: 'network_dim', label: 'Rank (dim)', type: 'int', group: 'basic', section: '网络', default: 32, min: 1, max: 256 },
      { key: 'network_alpha', label: 'Alpha', type: 'int', group: 'basic', section: '网络', default: 16, min: 1, max: 256 },
      { key: 'resolution', label: '分辨率', type: 'enum', group: 'basic', section: '训练', default: '1024', options: [{ value: '768', label: '768' }, { value: '1024', label: '1024' }, { value: '1280', label: '1280' }] },
      { key: 'train_batch_size', label: 'Batch size', type: 'int', group: 'basic', section: '训练', default: 1, min: 1, max: 16 },
      { key: 'optimizer_type', label: '优化器', type: 'enum', group: 'advanced', section: '优化', default: 'AdamW8bit', options: [{ value: 'AdamW8bit', label: 'AdamW8bit' }, { value: 'AdamW', label: 'AdamW' }, { value: 'Prodigy', label: 'Prodigy' }, { value: 'Adafactor', label: 'Adafactor' }] },
      { key: 'lr_scheduler', label: '学习率调度', type: 'enum', group: 'advanced', section: '优化', default: 'cosine', options: [{ value: 'constant', label: 'constant' }, { value: 'cosine', label: 'cosine' }, { value: 'cosine_with_restarts', label: 'cosine_with_restarts' }] },
      { key: 'mixed_precision', label: '混合精度', type: 'enum', group: 'advanced', section: '精度与显存', default: 'bf16', options: [{ value: 'fp16', label: 'fp16' }, { value: 'bf16', label: 'bf16' }] },
      { key: 'gradient_checkpointing', label: '梯度检查点', type: 'bool', group: 'advanced', section: '精度与显存', default: true, help: '降低显存占用，训练速度略降' },
      { key: 'cache_latents', label: '缓存 latents', type: 'bool', group: 'advanced', section: '精度与显存', default: true },
      { key: 'fp8_base', label: 'FP8 底模', type: 'bool', group: 'advanced', section: '精度与显存', default: false, unsupported_reason: '该安装实例的版本档案未声明支持' },
      { key: 'save_every_n_steps', label: '每 N 步保存', type: 'int', group: 'advanced', section: '保存与采样', default: 500, min: 0 },
      { key: 'sample_every_n_steps', label: '每 N 步采样', type: 'int', group: 'advanced', section: '保存与采样', default: 500, min: 0 },
      { key: 'sample_prompts', label: '采样提示词', type: 'string', group: 'advanced', section: '保存与采样', default: '1girl, aki, portrait --w 1024 --h 1024 --s 28' },
      { key: 'noise_offset', label: 'noise_offset', type: 'float', group: 'native', section: 'sd-scripts', default: 0.0357, step: 0.001 },
      { key: 'min_snr_gamma', label: 'min_snr_gamma', type: 'float', group: 'native', section: 'sd-scripts', default: 5, step: 0.5 },
      { key: 'max_data_loader_n_workers', label: 'max_data_loader_n_workers', type: 'int', group: 'native', section: 'sd-scripts', default: 2, min: 0 },
    ],
  }
}

function validate(draft: TrainingDraft): ValidationResult {
  const issues: ValidationResult['issues'] = []
  if (!draft.name.trim()) issues.push({ field: 'name', level: 'error', message: '任务名称不能为空' })
  if (!draft.base_model_path.trim()) issues.push({ field: 'base_model_path', level: 'error', message: '请选择底模文件' })
  if (!draft.dataset_id) issues.push({ field: 'dataset_id', level: 'error', message: '请选择数据集' })
  if (!draft.output_dir.trim()) issues.push({ field: 'output_dir', level: 'error', message: '请选择输出目录' })
  if (Number(draft.params.network_alpha) > Number(draft.params.network_dim))
    issues.push({ field: 'network_alpha', level: 'warning', message: 'Alpha 大于 Rank，等效学习率会被放大' })

  const p = draft.params
  const toml = [
    '# 演示数据：由前端示例生成，非适配器输出',
    '[model]',
    `pretrained_model_name_or_path = "${draft.base_model_path.replace(/\\/g, '/')}"`,
    '',
    '[network]',
    'network_module = "networks.lora"',
    `network_dim = ${p.network_dim}`,
    `network_alpha = ${p.network_alpha}`,
    '',
    '[training]',
    `output_name = "${draft.name}"`,
    `output_dir = "${draft.output_dir.replace(/\\/g, '/')}"`,
    `max_train_steps = ${p.max_train_steps}`,
    `learning_rate = ${p.learning_rate}`,
    `train_batch_size = ${p.train_batch_size}`,
    `resolution = "${p.resolution},${p.resolution}"`,
    `optimizer_type = "${p.optimizer_type}"`,
    `lr_scheduler = "${p.lr_scheduler}"`,
    `mixed_precision = "${p.mixed_precision}"`,
    `gradient_checkpointing = ${p.gradient_checkpointing}`,
    `cache_latents = ${p.cache_latents}`,
    `save_every_n_steps = ${p.save_every_n_steps}`,
    `noise_offset = ${p.noise_offset}`,
    `min_snr_gamma = ${p.min_snr_gamma}`,
  ].join('\n')

  return {
    ok: !issues.some((i) => i.level === 'error'),
    issues,
    native_config: toml,
    native_format: 'toml',
    argv: [
      '<installation python>',
      'sdxl_train_network.py',
      '--config_file',
      '<job_dir>/config.native.toml',
    ],
  }
}

const system: SystemStatus = {
  backend_version: '0.1.0-demo',
  supervisor: 'running',
  gpu: { name: 'NVIDIA GeForce RTX 4090', memory_used_mb: 17_820, memory_total_mb: 24_564, utilization: 97 },
}

const settings: AppSettings = {
  engine_root: 'D:\\SDaki\\LP-Lora-Trainer\\engine',
  data_root: 'D:\\SDaki\\LP-Lora-Trainer\\data',
  comfyui_lora_dir: null,
  gpu_monitor: true,
  log_tail_lines: 500,
}

export const demo = {
  system: () => system,
  engines: () => engines,
  tasks: () => tasks,
  datasets: () => datasets,
  settings: () => settings,
  images,
  metrics,
  log,
  artifacts,
  capabilities,
  validate,
}
