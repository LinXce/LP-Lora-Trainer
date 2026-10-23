import { onBeforeUnmount, ref, watch, type Ref } from 'vue'
import { api } from '@/api'
import { onSnapshotsRefreshed, onTaskLog, onTaskMetrics } from './store'
import type { MetricPoint } from '@/types/api'

const MAX_METRIC_POINTS = 20_000

/** Snapshot of a task's metrics, then live increments from the event stream. */
export function useTaskMetrics(taskId: Ref<string | null>) {
  const points = ref<MetricPoint[]>([])
  const loading = ref(false)
  let generation = 0
  let pending: MetricPoint[] = []

  function normalize(incoming: MetricPoint[]): MetricPoint[] {
    // Engines may write loss and learning rate on different lines of the same step;
    // merge by step so switching between series never drops a value.
    const byStep = new Map<number, MetricPoint>()
    for (const point of incoming) {
      byStep.set(point.step, { ...byStep.get(point.step), ...point })
    }
    return [...byStep.values()].sort((a, b) => a.step - b.step)
  }

  function append(incoming: MetricPoint[]) {
    const byStep = new Map<number, MetricPoint>(points.value.map((point) => [point.step, point]))
    for (const point of incoming) {
      byStep.set(point.step, { ...byStep.get(point.step), ...point })
    }
    points.value = [...byStep.values()]
      .sort((a, b) => a.step - b.step)
      .slice(-MAX_METRIC_POINTS)
  }

  async function reload(reset = false) {
    const id = taskId.value
    const current = ++generation
    pending = []
    if (reset) points.value = []
    if (!id) { loading.value = false; return }
    loading.value = true
    try {
      const snap = await api.tasks.metrics(id)
      if (current === generation && taskId.value === id) points.value = normalize(snap).slice(-MAX_METRIC_POINTS)
    } catch {
      // Keep existing data when a reconnect refresh fails; never invent metrics.
    } finally {
      if (current === generation) {
        append(pending)
        pending = []
        loading.value = false
      }
    }
  }

  watch(taskId, () => void reload(true), { immediate: true })
  const offSnapshots = onSnapshotsRefreshed(() => void reload())
  const off = onTaskMetrics((id, incoming) => {
    if (id !== taskId.value || !incoming.length) return
    if (loading.value) pending = pending.concat(incoming).slice(-MAX_METRIC_POINTS)
    else append(incoming)
  })
  onBeforeUnmount(() => { generation += 1; off(); offSnapshots() })

  return { points, loading }
}

/** Tail of the raw log, capped to `limit` lines in memory. */
export function useTaskLog(taskId: Ref<string | null>, limit: Ref<number>) {
  const lines = ref<string[]>([])
  const loading = ref(false)
  let generation = 0
  let pending: string[] = []

  async function reload(reset = false) {
    const id = taskId.value
    const current = ++generation
    pending = []
    if (reset) lines.value = []
    if (!id) { loading.value = false; return }
    loading.value = true
    let loaded = false
    try {
      const tail = await api.tasks.logTail(id, limit.value)
      if (current === generation && taskId.value === id) { lines.value = tail; loaded = true }
    } catch {
      // Leave the last known tail visible when the backend is temporarily unreachable.
    } finally {
      if (current === generation) {
        // A tail snapshot may already contain events buffered while it was loading.
        let overlap = 0
        if (loaded) {
          for (let size = Math.min(lines.value.length, pending.length); size > 0; size--) {
            if (pending.slice(0, size).every((line, i) => line === lines.value[lines.value.length - size + i])) {
              overlap = size
              break
            }
          }
        }
        lines.value = lines.value.concat(pending.slice(overlap)).slice(-limit.value)
        pending = []
        loading.value = false
      }
    }
  }
  watch([taskId, limit], () => void reload(true), { immediate: true })
  const offSnapshots = onSnapshotsRefreshed(() => void reload())
  const off = onTaskLog((id, incoming) => {
    if (id !== taskId.value) return
    if (loading.value) pending = pending.concat(incoming).slice(-limit.value)
    else lines.value = lines.value.concat(incoming).slice(-limit.value)
  })
  onBeforeUnmount(() => { generation += 1; off(); offSnapshots() })

  return { lines, loading, reload }
}
