import { onBeforeUnmount, ref, watch, type Ref } from 'vue'
import { api } from '@/api'
import { onTaskLog, onTaskMetrics } from './store'
import type { MetricPoint } from '@/types/api'

const MAX_METRIC_POINTS = 20_000

/** Snapshot of a task's metrics, then live increments from the event stream. */
export function useTaskMetrics(taskId: Ref<string | null>) {
  const points = ref<MetricPoint[]>([])
  const loading = ref(false)

  watch(
    taskId,
    async (id) => {
      points.value = []
      if (!id) return
      loading.value = true
      try {
        const snap = await api.tasks.metrics(id)
        if (taskId.value === id) points.value = snap
      } catch {
        // Leave empty — the chart shows "no parsable data" rather than invented values.
      } finally {
        loading.value = false
      }
    },
    { immediate: true },
  )

  const off = onTaskMetrics((id, pts) => {
    if (id !== taskId.value || !pts.length) return
    const lastStep = points.value[points.value.length - 1]?.step ?? -Infinity
    const fresh = pts.filter((p) => p.step > lastStep)
    if (!fresh.length) return
    const next = points.value.concat(fresh)
    points.value = next.length > MAX_METRIC_POINTS ? next.slice(-MAX_METRIC_POINTS) : next
  })
  onBeforeUnmount(off)

  return { points, loading }
}

/** Tail of the raw log, capped to `limit` lines in memory. */
export function useTaskLog(taskId: Ref<string | null>, limit: Ref<number>) {
  const lines = ref<string[]>([])
  const loading = ref(false)

  async function reload() {
    const id = taskId.value
    lines.value = []
    if (!id) return
    loading.value = true
    try {
      const tail = await api.tasks.logTail(id, limit.value)
      if (taskId.value === id) lines.value = tail
    } catch {
      lines.value = []
    } finally {
      loading.value = false
    }
  }
  watch(taskId, reload, { immediate: true })

  const off = onTaskLog((id, incoming) => {
    if (id !== taskId.value) return
    const next = lines.value.concat(incoming)
    lines.value = next.length > limit.value ? next.slice(-limit.value) : next
  })
  onBeforeUnmount(off)

  return { lines, loading, reload }
}
