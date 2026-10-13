/**
 * Sample data for UI development only. Enabled with `?demo` in a dev build;
 * production builds never enter this mode. Every screen shows a banner while
 * active and all write operations are rejected.
 */
import type {
  AdapterCatalog,
  AppSettings,
  Artifact,
  BaseModel,
  Dataset,
  DatasetImage,
  EngineCapabilities,
  EngineInstallation,
  HardwareInfo,
  InstallationSession,
  InstallPlan,
  MetricPoint,
  Page,
  SessionDiagnosis,
  SystemStatus,
  TaskSummary,
  TerminalLogChunk,
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
    source_repair: '补齐 sd-scripts 训练源码',
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

const baseModels: BaseModel[] = [
  { id: 'sd1', label: 'SD 1.5', engines: ['kohya'] },
  { id: 'sd2', label: 'SD 2.x', engines: ['kohya'] },
  { id: 'sdxl', label: 'SDXL', engines: ['kohya'] },
  // No musubi instance exists in the demo data: picking this shows the
  // "no engine supports it" state (red picker, no toast).
  { id: 'wan21', label: 'Wan 2.1', engines: ['musubi_tuner'] },
]

function capabilities(id: string): EngineCapabilities {
  return {
    installation_id: id,
    architectures: ['sd1', 'sd2', 'sdxl'],
    submittable: true,
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
      { key: 'sample_every_n_steps', label: '每 N 步采样', type: 'int', group: 'preview', section: '保存与采样', default: 500, min: 0 },
      { key: 'sample_prompts', label: '采样提示词', type: 'string', multiline: true, group: 'preview', section: '保存与采样', default: '1girl, aki, portrait --w 1024 --h 1024 --s 28' },
      { key: 'noise_offset', label: 'noise_offset', type: 'float', group: 'native', section: 'sd-scripts', default: 0.0357, step: 0.001 },
      { key: 'min_snr_gamma', label: 'min_snr_gamma', type: 'float', group: 'native', section: 'sd-scripts', default: 5, step: 0.5 },
      { key: 'sdxl_no_half_vae', label: 'sdxl_no_half_vae', type: 'bool', group: 'native', section: 'SDXL', default: false, architectures: ['sdxl'] },
      { key: 'clip_skip', label: 'clip_skip', type: 'int', group: 'native', section: 'SD 1.5 / 2.x', default: 1, min: 1, max: 12, architectures: ['sd1', 'sd2'] },
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
    `output_name = "${draft.output_name || draft.name}"`,
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
    submittable: true,
    pipeline: null,
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
  installation_workflow_version: 3,
  supervisor: 'running',
  gpu: { name: 'NVIDIA GeForce RTX 4090', memory_used_mb: 17_820, memory_total_mb: 24_564, utilization: 97, temperature_c: 67, memory_utilization: 72, power_draw_w: 286.4, power_limit_w: 450, fan_speed_percent: 54 },
}

const settings: AppSettings = {
  engine_root: 'D:\\SDaki\\LP-Lora-Trainer\\engine',
  data_root: 'D:\\SDaki\\LP-Lora-Trainer\\data',
  comfyui_lora_dir: null,
  gpu_monitor: true,
  log_tail_lines: 500,
  proxy_url: 'http://127.0.0.1:7890',
  proxy_enabled: true,
}

const adapterCatalog: AdapterCatalog = {
  // Static fixtures exist only in explicit development demo mode.
  adapters: [
    ['kohya', 'Kohya'], ['ai_toolkit', 'AI Toolkit'], ['musubi_tuner', 'Musubi Tuner'],
  ].map(([engine_id, display_name]) => ({
    engine_id, display_name, adapter_version: 'demo', submittable: engine_id !== 'ai_toolkit',
    environment_strategy: 'demo', installation_help: '演示模式不执行安装。',
    installation_source_help: '演示数据', default_installation_source: 'existing',
  })),
  errors: [],
}

const hardware: HardwareInfo = {
  os: 'Windows', arch: 'AMD64', gpu_name: 'NVIDIA GeForce RTX 4090',
  driver_version: '566.03', cuda_version: '12.6', compute_cap: '8.9', cuda_usable: true,
}

function installPlan(id: string): InstallPlan {
  return {
    installation_id: id, engine_id: 'kohya', supported: true, blocked_reason: null,
    engine_type_confirmed: true, engine_decides_cuda: true, torch_source: 'existing',
    torch_source_options: ['existing'],
    torch_source_reason: '由 Kohya 官方安装器自选 CUDA；本工具不指定 CUDA 配置',
    cuda_version: '12.6', cuda_tag: 'cu126', os: 'Windows',
    gpu_name: hardware.gpu_name, driver_version: hardware.driver_version, compute_cap: hardware.compute_cap,
    mirror_preset: 'official',
    mirror_presets: [
      { value: 'official', label: '官方 PyPI 源', url: null },
      { value: 'tuna', label: '清华大学镜像', url: 'https://pypi.tuna.tsinghua.edu.cn/simple' },
      { value: 'custom', label: '自定义镜像 URL', url: null },
    ],
    torch_mirror_preset: 'project_default',
    torch_mirror_presets: [
      { value: 'project_default', label: '跟随引擎项目默认', url: null },
      { value: 'official', label: '官方 PyTorch wheel 源', url: 'https://download.pytorch.org/whl/cu126' },
    ],
    python_mirror_preset: 'github',
    python_mirror_presets: [
      { value: 'github', label: '默认（GitHub）', url: null },
      { value: 'tuna', label: '清华大学镜像', url: 'https://mirrors.tuna.tsinghua.edu.cn/github-release/astral-sh/python-build-standalone' },
      { value: 'custom', label: '自定义镜像 URL', url: null },
    ],
  }
}

const sessionDiagnosis = (sessionId: string): SessionDiagnosis => ({
  session_id: sessionId,
  state: 'failed',
  hints: [
    {
      id: 'python_download',
      title: '无法下载受管 Python',
      detail: 'uv 需要下载受管解释器（python-build-standalone），但下载失败。',
      suggestion: '在安装对话框把「Python 下载源」改为可用镜像（南京大学 / 清华大学 / 自定义）后重试。',
      evidence: ['cause: Failed to download `https://github.com/astral-sh/python-build-standalone/releases/download/...`'],
    },
  ],
})

const installationSessions: InstallationSession[] = [
  {
    session_id: 'demo-install-failed',
    installation_id: 'demo-kohya',
    label: 'kohya_ss',
    engine_id: 'kohya',
    title: 'Kohya 官方 Windows headless 安装器（CUDA 由引擎决定）',
    cwd: 'D:\\SDaki\\LP-Lora-Trainer\\engine\\kohya_ss',
    python_executable: 'D:\\SDaki\\LP-Lora-Trainer\\engine\\kohya_ss\\venv\\Scripts\\python.exe',
    commands: [['venv\\Scripts\\python.exe', 'setup_windows.py']],
    command_index: 0,
    state: 'failed',
    started_at: '2026-10-12T08:40:12',
    finished_at: '2026-10-12T08:41:03',
    error: 'Installation failed: Official installer failed (step 1, exit code 1)',
    diagnostic_error: null,
    mirror_url: 'https://pypi.tuna.tsinghua.edu.cn/simple',
    warnings: [],
  },
  {
    session_id: 'demo-install-fallback',
    installation_id: 'demo-musubi',
    label: 'musubi-tuner',
    engine_id: 'musubi_tuner',
    title: 'Musubi Tuner 官方 uv 项目安装流程',
    cwd: 'D:\\SDaki\\LP-Lora-Trainer\\engine\\musubi-tuner',
    python_executable: 'D:\\SDaki\\LP-Lora-Trainer\\engine\\musubi-tuner\\.venv\\Scripts\\python.exe',
    commands: [['.venv\\Scripts\\python.exe', '-m', 'pip', 'install', '-r', 'requirements.txt']],
    command_index: 0,
    state: 'succeeded',
    started_at: '2026-10-12T09:02:41',
    finished_at: '2026-10-12T09:09:18',
    error: null,
    diagnostic_error: null,
    mirror_url: null,
    warnings: [
      {
        id: 'mirror_fallback',
        title: '镜像源不可用，已回退到官方源',
        detail: '镜像源下载失败（依赖源缺少所需包或版本），已改用官方源完成安装。',
        suggestion: '如需继续使用镜像，请确认该镜像地址可访问；也可以改用其它镜像或保留官方源。',
        evidence: [],
      },
    ],
  },
]

const sessionLogs: Record<string, string[]> = {
  'demo-install-failed': [
    '[LP] Kohya 官方 Windows headless 安装器（CUDA 由引擎决定）',
    '[LP] Engine directory: D:\\SDaki\\LP-Lora-Trainer\\engine\\kohya_ss',
    '[LP] Package index: https://pypi.tuna.tsinghua.edu.cn/simple',
    '',
    '[LP] Creating engine-local Python environment:',
    '[LP] runtime\\uv\\uv.exe venv engine\\kohya_ss\\venv --python 3.11 --managed-python --seed',
    'error: Failed to download `https://github.com/astral-sh/python-build-standalone/releases/download/20261003/cpython-3.11.17.tar.gz`',
    '  cause: tcp connect error',
    '',
    '[LP] Installation failed: Official installer failed (step 1, exit code 1)',
    '',
  ],
  'demo-install-fallback': [
    '[LP] Musubi Tuner 官方 uv 项目安装流程',
    '[LP] Engine directory: D:\\SDaki\\LP-Lora-Trainer\\engine\\musubi-tuner',
    '[LP] Package index: https://pypi.tuna.tsinghua.edu.cn/simple',
    '',
    '> uv.exe sync --extra cu130',
    'error: No matching distribution found for torch==2.14.1+cu130',
    '',
    '[LP] 镜像源下载失败（依赖源缺少所需包或版本），改用官方源重试',
    '[LP] Package index: official default',
    '',
    '> uv.exe sync --extra cu130',
    'Resolved 184 packages in 7.25s',
    '',
    '[LP] Official installation finished; engine Python found',
    '[LP] Environment diagnosis passed; installation complete',
    '',
  ],
}

function installationLog(id: string, offset: number): TerminalLogChunk {
  const text = (sessionLogs[id] ?? sessionLogs['demo-install-failed']).join('\n')
  return offset === 0 ? { text, offset: text.length } : { text: '', offset }
}

export const demo = {
  adapterCatalog,
  system: () => system,
  hardware,
  engines: () => engines,
  baseModels,
  tasks: () => tasks,
  datasets: () => datasets,
  settings: () => settings,
  images,
  metrics,
  log,
  artifacts,
  capabilities,
  installPlan,
  sessionDiagnosis,
  installationSessions,
  installationLog,
  validate,
}
