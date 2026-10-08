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
import { isEngineId, knownEngineIds } from './format'
import type { EngineInstallation, MetricPoint, ServerEvent, SystemStatus, TaskSummary } from '@/types/api'

// Keep the UI resilient to records produced by an older backend.  The backend
// remains authoritative, but a single adapter candidate is safe to display as
// the type while the record is being repaired by the next snapshot.
function normalizeEngine(item: EngineInstallation, fallback: EngineInstallation | null = null): EngineInstallation {
  const candidates = knownEngineIds(item.candidate_engines)
  const explicit = isEngineId(item.engine_id) ? item.engine_id : null
  // SSE can deliver an event that was queued just before a manual rescan.
  // Never let that older, incomplete record erase the type already displayed
  // from the fresh snapshot. The backend also treats engine_id as durable
  // identity when a temporary static scan returns no candidates.
  const fallbackId = fallback && isEngineId(fallback.engine_id) ? fallback.engine_id : null
  const engine_id = explicit ?? fallbackId ?? (candidates.length === 1 ? candidates[0] : null)
  return { ...item, engine_id, candidate_engines: candidates }
}

function normalizeEngines(items: EngineInstallation[]): EngineInstallation[] {
  return items.map((item) => normalizeEngine(item))
}

export type Connection = StreamStatus | 'unreachable' | 'demo'

export const store = reactive({
  connection: 'connecting' as Connection,
  system: null as SystemStatus | null,
  tasks: [] as TaskSummary[],
  engines: [] as EngineInstallation[],
  tasksLoaded: false,
  enginesLoaded: false,
  loadError: null as string | null,
  /** Events discarded because the pending queue hit its cap; a UI may surface this. */
  droppedEvents: 0,
})

type LogListener = (taskId: string, lines: string[]) => void
type MetricListener = (taskId: string, points: MetricPoint[]) => void
const logListeners = new Set<LogListener>()
const metricListeners = new Set<MetricListener>()
const snapshotListeners = new Set<() => void>()

/** Reload selected-task histories after reconnect, including updates missed offline. */
export function onSnapshotsRefreshed(fn: () => void): () => void {
  snapshotListeners.add(fn)
  return () => snapshotListeners.delete(fn)
}

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
    store.engines = normalizeEngines(await api.engines.list())
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

/** Bound on the pending queue: a stalled flush must not grow without limit. */
const MAX_PENDING_EVENTS = 5000
/** Deferred flush attempts allowed while a snapshot is still loading. */
const MAX_SNAPSHOT_DEFERRALS = 50

let pending: ServerEvent[] = []
let flushTimer: number | undefined
let snapshotDeferrals = 0

function flush() {
  flushTimer = undefined
  if (snapshotLoading) {
    if (snapshotDeferrals < MAX_SNAPSHOT_DEFERRALS) {
      snapshotDeferrals += 1
      flushTimer = window.setTimeout(flush, 100)
      return
    }
    // The snapshot never arrived: stop spinning, tell the user why, and apply
    // whatever is already queued instead of dropping it silently.
    snapshotLoading = false
    store.loadError = '加载快照超时，后端可能未响应'
  }
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
        upsert(
          store.engines,
          normalizeEngine(
            ev.installation,
            store.engines.find((e) => e.installation_id === ev.installation.installation_id) ?? null,
          ),
          (e) => e.installation_id,
        )
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
  if (pending.length > MAX_PENDING_EVENTS) {
    // Keep the newest events: the freshest state wins for the UI.
    const overflow = pending.length - MAX_PENDING_EVENTS
    pending.splice(0, overflow)
    store.droppedEvents += overflow
  }
  if (flushTimer === undefined) {
    flushTimer = window.setTimeout(flush, document.hidden ? 3000 : 500)
  }
}

let snapshotLoading = false

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
    onOpen: () => {
      pending = []
      store.droppedEvents = 0
      snapshotDeferrals = 0
      snapshotLoading = true
      void refreshAll().finally(() => {
        snapshotLoading = false
        for (const fn of snapshotListeners) fn()
      })
    },
    onStatus: (s) => {
      // Keep 'unreachable' sticky until the stream actually opens.
      if (s === 'retrying' && store.connection === 'unreachable') return
      store.connection = s
    },
  })
}

export function stopLiveUpdates(): void {
  stopStream?.()
  stopStream = null
  window.clearTimeout(flushTimer)
  flushTimer = undefined
  pending = []
  snapshotDeferrals = 0
  store.droppedEvents = 0
}
