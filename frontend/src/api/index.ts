/**
 * Typed endpoint functions. In demo mode (dev only, `?demo`) reads are served
 * from clearly-labelled sample data and writes are rejected — the UI never
 * pretends an operation succeeded.
 */
import { ApiError, request } from './http'
import { demo, isDemoMode } from './demo'
import type {
  AppSettings,
  Artifact,
  BaseModel,
  Dataset,
  DatasetImage,
  EngineCapabilities,
  EngineInstallation,
  InstallationSession,
  TerminalLogChunk,
  InstallEnvironmentOptions,
  MetricPoint,
  Page,
  SystemStatus,
  StoredTrainingDraft,
  TaskSummary,
  TrainingDraft,
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
  system: {
    status: () => read(() => request<SystemStatus>('/system/status'), demo.system),
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
  },

  terminal: {
    sessions: () => read(() => request<InstallationSession[]>('/terminal/sessions'), () => []),
    log: (id: string, offset: number) =>
      read(() => request<TerminalLogChunk>(`/terminal/sessions/${encodeURIComponent(id)}/log`, { query: { offset } }), () => ({ text: '', offset: 0 })),
    stop: (id: string) => write(() => request<void>(`/terminal/sessions/${encodeURIComponent(id)}/stop`, { method: 'POST' })),
  },

  datasets: {
    list: () => read(() => request<Dataset[]>('/datasets'), demo.datasets),
    add: (path: string, name: string) =>
      write(() => request<Dataset>('/datasets', { method: 'POST', body: { path, name } })),
    scan: (id: string) =>
      write(() => request<void>(`/datasets/${encodeURIComponent(id)}/scan`, { method: 'POST' })),
    images: (id: string, offset: number, limit: number, issue?: string) =>
      read(
        () =>
          request<Page<DatasetImage>>(`/datasets/${encodeURIComponent(id)}/images`, {
            query: { offset, limit, issue },
          }),
        () => demo.images(id, offset, limit, issue),
      ),
    saveCaption: (id: string, imageId: string, caption: string) =>
      write(() =>
        request<void>(
          `/datasets/${encodeURIComponent(id)}/images/${encodeURIComponent(imageId)}/caption`,
          { method: 'PUT', body: { caption } },
        ),
      ),
  },

  training: {
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
