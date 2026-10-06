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
  },

  engines: {
    list: () => read(() => request<EngineInstallation[]>('/engines'), demo.engines),
    rescan: () => write(() => request<EngineInstallation[]>('/engines/rescan', { method: 'POST' })),
    setDefault: (id: string) =>
      write(() => request<void>(`/engines/${encodeURIComponent(id)}/default`, { method: 'POST' })),
    diagnose: (id: string) =>
      write(() => request<void>(`/engines/${encodeURIComponent(id)}/diagnose`, { method: 'POST' })),
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
