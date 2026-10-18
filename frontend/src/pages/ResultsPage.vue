<script setup lang="ts">
import SelectInput from '@/components/SelectInput.vue'
/**
 * Results: per-task checkpoints & samples. Publishing is explicit, per file,
 * never overwrites (backend enforces), and carries architecture/base-model
 * constraints alongside the artifact.
 */
import { computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import EmptyState from '@/components/EmptyState.vue'
import ImageViewer, { type ViewerItem } from '@/components/ImageViewer.vue'
import ModalDialog from '@/components/ModalDialog.vue'
import PathInput from '@/components/PathInput.vue'
import StatusTag from '@/components/StatusTag.vue'
import { api } from '@/api'
import { revealPath } from '@/api/bridge'
import { store } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import { useArtifacts } from '@/features/useArtifacts'
import { groupSamples } from '@/features/samples'
import { engineName, fmtBytes, fmtTime, taskStateMeta } from '@/features/format'
import type { AppSettings, TaskSummary } from '@/types/api'

const route = useRoute()
const router = useRouter()

const outputHistory = ref<TaskSummary[]>([])
let historyGeneration = 0

/* Deleting a record reuses the task page's endpoint: it removes the record, the
   config snapshot, the logs and the metrics, and leaves every artifact on disk. */
const deleteOpen = ref(false)
const deleting = ref(false)
const canDeleteRecord = computed(() => !!task.value && !task.value.history_deleted)
async function deleteRecord() {
  const target = task.value
  if (!target || deleting.value) return
  deleting.value = true
  try {
    await api.tasks.delete(target.task_id)
    const remaining = await api.artifacts.history()
    outputHistory.value = remaining
    deleteOpen.value = false
  } finally {
    deleting.value = false
  }
}
watch(() => store.tasks.map(t => `${t.task_id}:${t.state}`).join(','), async () => {
  const current = ++historyGeneration
  try {
    const history = await api.artifacts.history()
    if (current === historyGeneration) outputHistory.value = history
  } catch (err) { toastError(err, '读取保留的训练产物') }
}, { immediate: true })
const finished = computed(() => {
  const all = [...store.tasks, ...outputHistory.value.filter(h => !store.tasks.some(t => t.task_id === h.task_id))]
    .filter(t => ['succeeded', 'stopped', 'running', 'failed'].includes(t.state))
  // Several deleted records can share a name (one per past run); the picker shows one
  // entry per name — the live task first, otherwise the newest history entry.
  const seen = new Set<string>()
  return all.filter((t) => {
    if (seen.has(t.name)) return false
    seen.add(t.name)
    return true
  })
})
const taskId = computed<string | null>(() => {
  const p = route.params.taskId
  return (typeof p === 'string' && p) || finished.value.find((t) => t.state === 'succeeded')?.task_id || finished.value[0]?.task_id || null
})
const task = computed(() => finished.value.find((t) => t.task_id === taskId.value) ?? null)

/* Artifacts follow the selected task and refresh live while it is still writing. */
const { artifacts, loading, error: artifactError } = useArtifacts(taskId)
const selectedCkpt = ref<string | null>(null)
watch(artifacts, (a) => {
  if (!a.some((item) => item.artifact_id === selectedCkpt.value)) selectedCkpt.value = null
})

const checkpoints = computed(() => artifacts.value.filter((a) => a.kind === 'checkpoint').sort((a, b) => (b.step ?? 0) - (a.step ?? 0)))
// The backend leaves `step` empty for samples; groups come from the file name.
const sampleGroups = computed(() => groupSamples(artifacts.value))

const ckpt = computed(() => checkpoints.value.find((c) => c.artifact_id === selectedCkpt.value) ?? checkpoints.value[0] ?? null)

/* Compare: pick up to 4 steps */
const compareKeys = ref<string[]>([])
const compareGroups = computed(() => compareKeys.value
  .map((key) => sampleGroups.value.find((g) => g.key === key))
  .filter((g): g is NonNullable<typeof g> => !!g))
// Follow new groups while the task runs, but keep a manual selection that is still valid.
watch(sampleGroups, (groups) => {
  const valid = compareKeys.value.filter((key) => groups.some((g) => g.key === key))
  compareKeys.value = valid.length ? valid : groups.slice(-4).map((g) => g.key)
})
function toggleCompare(key: string) {
  const i = compareKeys.value.indexOf(key)
  if (i >= 0) compareKeys.value.splice(i, 1)
  else {
    compareKeys.value.push(key)
    if (compareKeys.value.length > 4) compareKeys.value.shift()
    const order = new Map(sampleGroups.value.map((g, n) => [g.key, n]))
    compareKeys.value.sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0))
  }
}

/* Viewer: every compared image in display order, so arrows walk row by row. */
const viewerImages = computed(() => compareGroups.value.flatMap((g) => g.items.map((a) => ({ a, group: g }))))
const viewerItems = computed<ViewerItem[]>(() => viewerImages.value.map(({ a, group }) => ({
  src: a.preview_url ? `${a.preview_url}?full=true` : null,
  title: a.file_name,
  subtitle: group.label,
})))
const viewerIndex = ref<number | null>(null)
function openViewer(id: string) {
  const i = viewerImages.value.findIndex(({ a }) => a.artifact_id === id)
  viewerIndex.value = i >= 0 ? i : null
}

/* Publish */
const settings = ref<AppSettings | null>(null)
api.settings.get().then((s) => (settings.value = s)).catch(() => {})
const pubOpen = ref(false)
const pubDir = ref('')
const pubName = ref('')
function openPublish() {
  if (!ckpt.value) return
  pubDir.value = settings.value?.comfyui_lora_dir ?? ''
  pubName.value = ckpt.value.file_name
  pubOpen.value = true
}
const publishing = ref(false)
async function publish() {
  if (!ckpt.value) return
  publishing.value = true
  try {
    await api.artifacts.publish(ckpt.value.artifact_id, pubDir.value.trim(), pubName.value.trim())
    toast(`已发布 ${pubName.value}`, 'ok')
    pubOpen.value = false
  } catch (err) {
    toastError(err, '发布')
  } finally {
    publishing.value = false
  }
}

async function reveal(path: string) {
  if (!(await revealPath(path))) toast('仅在桌面窗口中可定位文件', 'info')
}
</script>

<template>
  <div class="page">
    <section class="page__main panel">
      <header class="page__header">
        <div class="stack" style="gap: 4px; min-width: 0">
          <h1 class="title-lg">训练结果</h1>
          <p v-if="task" class="muted truncate">{{ engineName(task.engine_id) }} / {{ task.installation_label }} · {{ task.architecture }} · {{ task.dataset_name ?? '—' }}</p>
        </div>
        <span class="spacer" />
        <SelectInput
          class="input task-select"
          :value="taskId ?? ''"
          @change="router.replace({ name: 'results', params: { taskId: ($event.target as HTMLSelectElement).value } })"
        >
          <option v-for="t in finished" :key="t.task_id" :value="t.task_id">{{ t.name }} — {{ t.history_deleted ? '任务记录已删除，产物保留' : taskStateMeta[t.state].label }}</option>
        </SelectInput>
        <button
          class="btn"
          :disabled="!canDeleteRecord || deleting"
          :title="canDeleteRecord ? '删除该任务记录与日志，已生成的产物保留' : '该任务记录已删除'"
          @click="deleteOpen = true"
        >
          删除记录
        </button>
      </header>

      <ModalDialog v-if="deleteOpen" title="删除训练任务记录？" @close="!deleting && (deleteOpen = false)">
        <p>删除 <strong>{{ task?.name }}</strong> 的任务记录、配置快照、运行日志和指标。</p>
        <p class="muted">不会删除已生成的模型、预览图、数据集或引擎环境。此操作不可撤销。</p>
        <template #footer>
          <button class="btn btn--ghost" :disabled="deleting" @click="deleteOpen = false">取消</button>
          <button class="btn btn--danger" :disabled="deleting" @click="deleteRecord">{{ deleting ? '正在删除…' : '确认删除' }}</button>
        </template>
      </ModalDialog>

      <div class="page__body" :class="{ 'is-loading': loading }">
        <EmptyState v-if="!task" icon="layers" title="暂无训练结果" text="任务产生 checkpoint 或采样图后会显示在这里。" />
        <template v-else>
          <p v-if="artifactError" class="art-err">加载失败：{{ artifactError }}</p>
          <div class="row" style="margin-bottom: 12px">
            <span class="label-pill">采样对比</span>
            <span class="muted">选择最多 4 个 step / epoch 并排比较</span>
          </div>
          <div class="steps">
            <button
              v-for="g in sampleGroups"
              :key="g.key"
              class="chip num"
              :class="{ 'is-on': compareKeys.includes(g.key) }"
              @click="toggleCompare(g.key)"
            >
              {{ g.label.replace(/^(step|epoch) /, '') }}
            </button>
            <span v-if="!sampleGroups.length && !loading" class="muted">
              {{ artifactError ? `加载失败：${artifactError}` : '该任务没有引擎生成的采样图' }}
            </span>
          </div>

          <!-- One row per step/epoch; prompts line up in columns so steps compare top to bottom. -->
          <div v-if="compareGroups.length" class="compare">
            <div v-for="g in compareGroups" :key="g.key" class="cmp">
              <span class="cmp__label num">{{ g.label }}</span>
              <div class="cmp__row">
                <button v-for="a in g.items" :key="a.artifact_id" class="cmp__img" :title="`${a.file_name}（点击查看大图）`" @click="openViewer(a.artifact_id)">
                  <img v-if="a.preview_url" :src="a.preview_url" alt="" loading="lazy" decoding="async" />
                  <AppIcon v-else name="image" :size="24" />
                </button>
              </div>
            </div>
          </div>
          <ImageViewer v-model:index="viewerIndex" :items="viewerItems" />
        </template>
      </div>
    </section>

    <aside class="page__side">
      <section class="panel ckpts">
        <div class="ckpts__head">
          <span class="label-pill">Checkpoints</span>
          <span class="muted num">{{ checkpoints.length }}</span>
        </div>
        <div class="ckpts__list scroll">
          <button
            v-for="c in checkpoints"
            :key="c.artifact_id"
            class="ck"
            :class="{ 'is-on': ckpt?.artifact_id === c.artifact_id }"
            @click="selectedCkpt = c.artifact_id"
          >
            <span class="truncate mono ck__name">{{ c.file_name }}</span>
            <span class="ck__meta num">
              {{ c.final ? '最终模型' : `step ${c.step ?? '?'}` }} · {{ fmtBytes(c.size_bytes) }}
              <StatusTag v-if="!c.complete" tone="warn" label="写入中" />
            </span>
          </button>
          <p v-if="!checkpoints.length" class="muted" style="padding: 8px 14px">
            {{ artifactError ? `加载失败：${artifactError}` : '暂无 checkpoint' }}
          </p>
        </div>
      </section>

      <section v-if="ckpt" class="panel panel--light pub">
        <strong class="truncate mono pub__name">{{ ckpt.file_name }}</strong>
        <dl class="kv pub__kv selectable">
          <dt>架构</dt>
          <dd>{{ ckpt.architecture }}</dd>
          <dt>基础模型</dt>
          <dd class="truncate">{{ ckpt.base_model ?? '未知' }}</dd>
          <dt>生成于</dt>
          <dd>{{ fmtTime(ckpt.created_at) }}</dd>
        </dl>
        <p class="pub__note">仅在与相同架构的基础模型搭配时使用。</p>
        <div class="row">
          <button class="btn btn--sm pub__ghost" @click="reveal(ckpt.path)"><AppIcon name="folder" :size="13" />定位</button>
          <span class="spacer" />
          <button class="btn btn--sm pub__go" :disabled="!ckpt.complete" @click="openPublish">
            <AppIcon name="upload" :size="13" />
            发布…
          </button>
        </div>
      </section>
    </aside>

    <ModalDialog v-if="pubOpen && ckpt" title="发布 LoRA" @close="pubOpen = false">
      <p class="muted">
        只复制选中的这一个文件。先写入临时文件再完成落盘；目标目录存在同名文件时会拒绝发布，不会覆盖。
      </p>
      <label class="field">
        <span class="field__label">目标目录 <span class="field__hint">例如 ComfyUI/models/loras</span></span>
        <PathInput v-model="pubDir" kind="directory" dialog-title="选择发布目录" />
      </label>
      <label class="field">
        <span class="field__label">文件名</span>
        <input v-model="pubName" class="input mono" spellcheck="false" />
      </label>
      <template #footer>
        <button class="btn btn--ghost" @click="pubOpen = false">取消</button>
        <button class="btn btn--primary" :disabled="!pubDir.trim() || !pubName.trim() || publishing" @click="publish">发布</button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
.page__header,
.page__body {
  padding-left: 44px;
}
.task-select {
  width: 280px;
}
.page__body.is-loading {
  opacity: 0.55;
}
.art-err {
  margin-bottom: 12px;
  padding: 9px 14px;
  border-radius: var(--radius-sm);
  background: var(--danger-bg);
  color: var(--danger);
  font-size: 12px;
}
.steps {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-bottom: 18px;
}
.compare {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.cmp {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.cmp__label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-2);
}
.cmp__row {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(132px, 1fr));
  gap: 10px;
}
.cmp__img {
  aspect-ratio: 1;
  display: grid;
  place-items: center;
  border-radius: var(--radius-md);
  background: linear-gradient(160deg, #5a5a5a, #474747);
  color: var(--light);
  overflow: hidden;
  cursor: zoom-in;
  transition: transform 0.15s var(--ease), box-shadow 0.15s;
}
.cmp__img:hover {
  transform: translateY(-2px);
  box-shadow: 0 0 0 2px var(--light);
}
.cmp__img img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.ckpts {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.ckpts__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 16px 10px;
}
.ckpts__list {
  flex: 1;
  padding: 0 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.ck {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  text-align: left;
}
.ck:hover {
  background: rgba(255, 255, 255, 0.05);
}
.ck.is-on {
  background: var(--panel-strong);
}
.ck__name {
  font-size: 11.5px;
}
.ck__meta {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-3);
}
.pub {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 18px;
}
.pub__name {
  font-size: 12px;
}
.pub__kv dt {
  color: var(--ink-soft);
}
.pub__note {
  font-size: 11px;
  color: var(--ink-soft);
}
.pub__ghost {
  background: rgba(0, 0, 0, 0.08);
  color: var(--ink);
}
.pub__ghost:hover:not(:disabled) {
  background: rgba(0, 0, 0, 0.14);
}
.pub__go {
  background: var(--frame);
  color: var(--light-strong);
}
.pub__go:hover:not(:disabled) {
  background: #111;
}
</style>
