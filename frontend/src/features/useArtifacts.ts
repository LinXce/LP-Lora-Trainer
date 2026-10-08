/**
 * Live artifact list for one task. Artifacts are snapshotted per task and
 * refreshed whenever the task changes state, plus polled while the task can
 * still produce new files (a running job writes checkpoints and samples).
 */
import { computed, onScopeDispose, ref, toValue, watch, type MaybeRefOrGetter, type Ref } from 'vue'
import { api } from '@/api'
import { store } from './store'
import type { Artifact, TaskState } from '@/types/api'

/** States in which an engine may still produce new artifacts. */
export const LIVE_ARTIFACT_STATES: TaskState[] = ['preparing', 'running', 'stopping']

const POLL_INTERVAL_MS = 5000

export interface UseArtifacts {
  artifacts: Ref<Artifact[]>
  loading: Ref<boolean>
  error: Ref<string | null>
  refresh: () => Promise<void>
}

export function useArtifacts(taskId: MaybeRefOrGetter<string | null>): UseArtifacts {
  const artifacts = ref<Artifact[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  /** Guards against a slow response overwriting a newer task's list. */
  let generation = 0
  let timer: number | undefined

  const stateKey = computed(() => {
    const id = toValue(taskId)
    if (!id) return ''
    return `${id}:${store.tasks.find((t) => t.task_id === id)?.state ?? ''}`
  })

  const live = computed(() => {
    const id = toValue(taskId)
    return !!id && store.tasks.some((t) => t.task_id === id && LIVE_ARTIFACT_STATES.includes(t.state))
  })

  /** Polling runs only while the watched task is in an active state. */
  function syncPolling() {
    if (live.value) {
      if (timer === undefined) timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS)
      return
    }
    if (timer !== undefined) {
      window.clearInterval(timer)
      timer = undefined
    }
  }

  async function refresh(): Promise<void> {
    const id = toValue(taskId)
    const current = ++generation
    if (!id) {
      artifacts.value = []
      error.value = null
      loading.value = false
      return
    }
    loading.value = true
    try {
      const list = await api.artifacts.list(id)
      if (current !== generation || toValue(taskId) !== id) return
      artifacts.value = list
      error.value = null
    } catch (err) {
      // Reading artifacts must never break the page; show the reason instead.
      if (current === generation) error.value = err instanceof Error ? err.message : String(err)
    } finally {
      if (current === generation) loading.value = false
    }
  }

  watch(
    stateKey,
    (key) => {
      if (!key) {
        generation += 1
        artifacts.value = []
        error.value = null
        loading.value = false
      } else {
        void refresh()
      }
      syncPolling()
    },
    { immediate: true },
  )
  watch(live, syncPolling)

  onScopeDispose(() => {
    generation += 1
    if (timer !== undefined) window.clearInterval(timer)
    timer = undefined
  })

  return { artifacts, loading, error, refresh }
}
