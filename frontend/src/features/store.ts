/**
 * App-wide live state: system status, tasks and engine installations.
 *
 * Snapshot first, then increments: every (re)connect of the event stream
 * refetches snapshots before applying events. Events are batched and flushed
 * at ~2 Hz while visible, and slowed down when the window is hidden/minimised.
 */
import { reactive } from 'vue'
import { api } from '@/api'
import { isDemoMode } from '@/api/demo'
import { ApiError } from '@/api/http'
import { openEventStream, type StreamStatus } from '@/api/events'
import type { EngineInstallation, MetricPoint, ServerEvent, SystemStatus, TaskSummary } from '@/types/api'

export type Connection = StreamStatus | 'unreachable' | 'demo'

export const store = reactive({
  connection: 'connecting' as Connection,
  system: null as SystemStatus | null,
  tasks: [] as TaskSummary[],
  engines: [] as EngineInstallation[],
  tasksLoaded: false,
  enginesLoaded: false,
  loadError: null as string | null,
})

type LogListener = (taskId: string, lines: string[]) => void
type MetricListener = (taskId: string, points: MetricPoint[]) => void
const logListeners = new Set<LogListener>()
const metricListeners = new Set<MetricListener>()

export function onTaskLog(fn: LogListener): () => void {
  logListeners.add(fn)
  return () => logListeners.delete(fn)
}
export function onTaskMetrics(fn: MetricListener): () => void {
  metricListeners.add(fn)
  return () => metricListeners.delete(fn)
}

function noteError(err: unknown) {
  if (err instanceof ApiError && err.isUnreachable) store.connection = 'unreachable'
  store.loadError = err instanceof Error ? err.message : String(err)
}

export async function refreshTasks(): Promise<void> {
  try {
    store.tasks = await api.tasks.list()
    store.loadError = null
  } catch (err) {
    noteError(err)
  } finally {
    store.tasksLoaded = true
  }
}

export async function refreshEngines(): Promise<void> {
  try {
    store.engines = await api.engines.list()
    store.loadError = null
  } catch (err) {
    noteError(err)
  } finally {
    store.enginesLoaded = true
  }
}

export async function refreshSystem(): Promise<void> {
  try {
    store.system = await api.system.status()
  } catch (err) {
    noteError(err)
  }
}

export function refreshAll(): Promise<unknown> {
  return Promise.all([refreshSystem(), refreshTasks(), refreshEngines()])
}

function upsert<T>(list: T[], item: T, key: (x: T) => string) {
  const i = list.findIndex((x) => key(x) === key(item))
  if (i >= 0) list[i] = item
  else list.unshift(item)
}

/* ---------- Batched event application ---------- */

let pending: ServerEvent[] = []
let flushTimer: number | undefined

function flush() {
  flushTimer = undefined
  const batch = pending
  pending = []
  const logs = new Map<string, string[]>()
  const metrics = new Map<string, MetricPoint[]>()

  for (const ev of batch) {
    switch (ev.kind) {
      case 'task.updated':
        upsert(store.tasks, ev.task, (t) => t.task_id)
        break
      case 'engine.updated':
        upsert(store.engines, ev.installation, (e) => e.installation_id)
        break
      case 'system.status':
        store.system = ev.status
        break
      case 'task.log':
        logs.set(ev.task_id, [...(logs.get(ev.task_id) ?? []), ...ev.lines])
        break
      case 'task.metrics':
        metrics.set(ev.task_id, [...(metrics.get(ev.task_id) ?? []), ...ev.points])
        break
    }
  }
  logs.forEach((lines, id) => logListeners.forEach((fn) => fn(id, lines)))
  metrics.forEach((pts, id) => metricListeners.forEach((fn) => fn(id, pts)))
}

function enqueue(ev: ServerEvent) {
  pending.push(ev)
  if (flushTimer === undefined) {
    flushTimer = window.setTimeout(flush, document.hidden ? 3000 : 500)
  }
}

let stopStream: (() => void) | null = null

export function startLiveUpdates(): void {
  if (stopStream) return
  if (isDemoMode()) {
    store.connection = 'demo'
    void refreshAll()
    return
  }
  void refreshAll()
  stopStream = openEventStream({
    onEvent: enqueue,
    onOpen: () => void refreshAll(),
    onStatus: (s) => {
      // Keep 'unreachable' sticky until the stream actually opens.
      if (s === 'retrying' && store.connection === 'unreachable') return
      store.connection = s
    },
  })
}
