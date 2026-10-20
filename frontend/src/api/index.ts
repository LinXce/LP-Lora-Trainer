/**
 * Typed endpoint functions. In demo mode (dev only, `?demo`) reads are served
 * from clearly-labelled sample data and writes are rejected — the UI never
 * pretends an operation succeeded.
 */
import { ApiError, request } from './http'
import { loadAdapterCatalog } from './adapterCatalog'
import { demo, isDemoMode } from './demo'
import type {
  EngineCloneOptions,
  PromptTemplate,
  PromptTemplateInput,
  AppSettings,
  Artifact,
  BaseModel,
  Dataset,
  DatasetImage,
  EngineCapabilities,
  EngineInstallation,
  InstallationSession,
  TerminalLogChunk,
  SessionDiagnosis,
  HardwareInfo,
  InstallEnvironmentOptions,
  InstallPlan,
  MetricPoint,
  Page,
  SystemStatus,
  StoredTrainingDraft,
  TaskSummary,
  TaskConfiguration,
  TrainingDraft,
  NativeParameterExport,
  TrainingExportSource,
  ValidationResult,
} from '@/types/api'

function read<T>(live: () => Promise<T>, sample: () => T): Promise<T> {
  if (isDemoMode()) return new Promise((r) => setTimeout(() => r(structuredClone(sample())), 120))
  return live()
}

function write<T>(live: () => Promise<T>): Promise<T> {
  if (isDemoMode()) return Promise.reject(new ApiError('演示模式不执行任何操作', 409, 'demo'))
  return live()
}

export const api = {
  promptTemplates: {
    list: () => read(() => request<PromptTemplate[]>('/prompt-templates'), () => []),
    get: (id: string) => read(() => request<PromptTemplate>(`/prompt-templates/${encodeURIComponent(id)}`),
      () => { throw new Error('演示模式没有已保存的提示词模板') }),
    create: (body: PromptTemplateInput) => write(() => request<PromptTemplate>('/prompt-templates', { method: 'POST', body })),
    update: (id: string, body: PromptTemplateInput) => write(() => request<PromptTemplate>(`/prompt-templates/${encodeURIComponent(id)}`, { method: 'PUT', body })),
    delete: (id: string) => write(() => request<void>(`/prompt-templates/${encodeURIComponent(id)}`, { method: 'DELETE' })),
  },
  adapters: {
    list: () => read(loadAdapterCatalog, () => demo.adapterCatalog),
  },
  system: {
    status: () => read(() => request<SystemStatus>('/system/status'), demo.system),
    hardware: () => read(() => request<HardwareInfo>('/system/hardware'), () => demo.hardware),
    uvCache: () => read(() =>
      request<{ path: string; exists: boolean; count: number; bytes: number }>('/system/uv-cache'),
      () => ({ path: '', exists: false, count: 0, bytes: 0 })),
    clearUvCache: () => write(() =>
      request<{ removed: number; bytes: number }>('/system/uv-cache', { method: 'DELETE' })),
    // Refused by the backend (409) while training is active or an orphan is unverified.
    shutdown: () => write(() => request<{ status: string }>('/system/shutdown', { method: 'POST' })),
  },

  baseModels: {
    list: () => read(() => request<BaseModel[]>('/base-models'), () => demo.baseModels),
  },

  engines: {
    list: () => read(() => request<EngineInstallation[]>('/engines'), demo.engines),
    rescan: () => write(() => request<EngineInstallation[]>('/engines/rescan', { method: 'POST' })),
    setDefault: (id: string) =>
      write(() => request<void>(`/engines/${encodeURIComponent(id)}/default`, { method: 'POST' })),
    diagnose: (id: string) =>
      write(() => request<void>(`/engines/${encodeURIComponent(id)}/diagnose`, { method: 'POST' })),
    update: (id: string) =>
      write(() => request<InstallationSession>(`/engines/${encodeURIComponent(id)}/update`, { method: 'POST' })),
    remove: (id: string) =>
      write(() => request<void>(`/engines/${encodeURIComponent(id)}`, { method: 'DELETE' })),
    clone: (options: EngineCloneOptions) =>
      write(() => request<InstallationSession>('/engines/clone', { method: 'POST', body: options })),
    repairSource: (id: string, options?: { use_proxy?: boolean }) =>
      write(() => request<InstallationSession>(`/engines/${encodeURIComponent(id)}/repair`, { method: 'POST', body: options ?? {} })),
    installEnvironment: (id: string, options: InstallEnvironmentOptions) =>
      write(() => request<InstallationSession>(`/engines/${encodeURIComponent(id)}/install`, { method: 'POST', body: options })),
    bindPython: (id: string, python: string) =>
      write(() =>
        request<void>(`/engines/${encodeURIComponent(id)}/python`, {
          method: 'PUT',
          body: { python_executable: python },
        }),
      ),
    confirmType: (id: string, engineId: string) =>
      write(() =>
        request<void>(`/engines/${encodeURIComponent(id)}/engine-type`, {
          method: 'PUT',
          body: { engine_id: engineId },
        }),
      ),
    capabilities: (id: string) =>
      read(
        () => request<EngineCapabilities>(`/engines/${encodeURIComponent(id)}/capabilities`),
        () => demo.capabilities(id),
      ),
    // Machine-aware install recommendation (no side effects); optional source mirrors a dialog choice.
    installPlan: (id: string, torchSource?: string) =>
      read(
        () => request<InstallPlan>(`/engines/${encodeURIComponent(id)}/install-plan`, {
          query: torchSource ? { torch_source: torchSource } : undefined,
        }),
        () => demo.installPlan(id),
      ),
  },

  terminal: {
    delete: (id: string) => write(() => request<void>(`/terminal/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' })),
    sessions: () => read(() => request<InstallationSession[]>('/terminal/sessions'), () => demo.installationSessions),
    log: (id: string, offset: number) =>
      read(() => request<TerminalLogChunk>(`/terminal/sessions/${encodeURIComponent(id)}/log`, { query: { offset } }), () => demo.installationLog(id, offset)),
    stop: (id: string) => write(() => request<void>(`/terminal/sessions/${encodeURIComponent(id)}/stop`, { method: 'POST' })),
    diagnosis: (id: string) =>
      read(() => request<SessionDiagnosis>(`/terminal/sessions/${encodeURIComponent(id)}/diagnosis`), () => demo.sessionDiagnosis(id)),
  },

  datasets: {
    list: (sort: string) => read(() => request<Dataset[]>('/datasets', { query: { sort } }), demo.datasets),
    add: (path: string, name: string) =>
      write(() => request<Dataset>('/datasets', { method: 'POST', body: { path, name } })),
    scan: (id: string) =>
      write(() => request<void>(`/datasets/${encodeURIComponent(id)}/scan`, { method: 'POST' })),
    remove: (id: string) =>
      write(() => request<void>(`/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' })),
    images: (id: string, offset: number, limit: number, issue?: string) =>
      read(
        () =>
          request<Page<DatasetImage>>(`/datasets/${encodeURIComponent(id)}/images`, {
            query: { offset, limit, issue },
          }),
        () => demo.images(id, offset, limit, issue),
      ),
    saveCaption: (id: string, imageId: string, caption: string, captionNl?: string) =>
      write(() =>
        request<void>(
          `/datasets/${encodeURIComponent(id)}/images/${encodeURIComponent(imageId)}/caption`,
          { method: 'PUT', body: { caption, caption_nl: captionNl ?? null } },
        ),
      ),
  },

  training: {
    exportParameters: (source: TrainingExportSource) => {
      // Pick wire fields explicitly: history also carries format/version metadata.
      const { name, output_name, installation_id, architecture, base_model_path, dataset_id, output_dir, params, engine_id, dataset_path } = source
      return write(() => request<NativeParameterExport>('/training/export-parameters', {
        method: 'POST', body: { name, output_name, installation_id, architecture, base_model_path, dataset_id, output_dir, params, engine_id, dataset_path },
      }))
    },
    validate: (draft: TrainingDraft) =>
      read(
        () => request<ValidationResult>('/training/validate', { method: 'POST', body: draft }),
        () => demo.validate(draft),
      ),
    submit: (draft: TrainingDraft) =>
      write(() => request<TaskSummary>('/training/submit', { method: 'POST', body: draft })),
    // The staged form lives in the data root, so every client shares it.
    draft: () =>
      read(
        () => request<StoredTrainingDraft>('/training/draft'),
        () => ({ draft: null, params_by_architecture: {}, updated_at: null }),
      ),
    saveDraft: (draft: TrainingDraft & { params_by_architecture: Record<string, Record<string, string | number | boolean | null>> }) =>
      write(() => request<StoredTrainingDraft>('/training/draft', { method: 'PUT', body: draft })),
    clearDraft: () => write(() => request<void>('/training/draft', { method: 'DELETE' })),
  },

  tasks: {
    list: () => read(() => request<TaskSummary[]>('/tasks'), demo.tasks),
    configuration: (id: string) => {
      if (isDemoMode()) return Promise.reject(new Error('演示任务没有可复用的历史配置'))
      return request<TaskConfiguration>(`/tasks/${encodeURIComponent(id)}/configuration`)
    },
    delete: (id: string) => write(() => request<void>(`/tasks/${encodeURIComponent(id)}`, { method: 'DELETE' })),
    metrics: (id: string) =>
      read(
        () => request<MetricPoint[]>(`/tasks/${encodeURIComponent(id)}/metrics`),
        () => demo.metrics(id),
      ),
    logTail: (id: string, lines: number) =>
      read(
        () => request<string[]>(`/tasks/${encodeURIComponent(id)}/log`, { query: { tail: lines } }),
        () => demo.log(id),
      ),
    acknowledgeExit: (id: string) =>
      write(() => request<void>(`/tasks/${encodeURIComponent(id)}/acknowledge-exit`, { method: 'POST', body: { confirmed_exited: true } })),
    stop: (id: string, force: boolean) =>
      write(() =>
        request<void>(`/tasks/${encodeURIComponent(id)}/stop`, { method: 'POST', body: { force } }),
      ),
  },

  artifacts: {
    history: () => read(() => request<TaskSummary[]>('/artifacts/tasks'), () => []),
    list: (taskId?: string) =>
      read(
        () => request<Artifact[]>('/artifacts', { query: { task_id: taskId } }),
        () => demo.artifacts(taskId),
      ),
    publish: (id: string, targetDir: string, fileName: string) =>
      write(() =>
        request<void>(`/artifacts/${encodeURIComponent(id)}/publish`, {
          method: 'POST',
          body: { target_dir: targetDir, file_name: fileName },
        }),
      ),
  },

  settings: {
    get: () => read(() => request<AppSettings>('/settings'), demo.settings),
    save: (s: AppSettings) => write(() => request<AppSettings>('/settings', { method: 'PUT', body: s })),
  },
}
