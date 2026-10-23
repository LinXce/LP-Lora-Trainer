<script setup lang="ts">
/**
 * Training tasks: queue + detail (metrics, loss curve, log tail, control).
 * Unparseable metrics show as 未知; stop vs. force-kill are distinct actions,
 * and only the recovery operations the adapter declares are offered.
 */
import { computed, nextTick, onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import StatusTag from '@/components/StatusTag.vue'
import ProgressBar from '@/components/ProgressBar.vue'
import MetricChart from '@/components/MetricChart.vue'
import EmptyState from '@/components/EmptyState.vue'
import ModalDialog from '@/components/ModalDialog.vue'
import SegTabs from '@/components/SegTabs.vue'
import { api } from '@/api'
import { revealPath } from '@/api/bridge'
import TrainingExportDialog from '@/components/TrainingExportDialog.vue'
import { snapshotTrainingExport } from '@/features/trainingExport'
import { refreshTasks, removeTask, store } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import { useTaskLog, useTaskMetrics } from '@/features/useTask'
import {
  ACTIVE_TASK_STATES,
  engineName,
  fmtDuration,
  fmtLoss,
  fmtNum,
  fmtTime,
  progressRatio,
  taskStateMeta,
} from '@/features/format'
import type { ResumeInfo, TaskSummary, TrainingExportSource } from '@/types/api'
import type { Tone } from '@/features/format'

const route = useRoute()
const router = useRouter()

/** The backend also persists `interrupted` for recovered orphaned runs; label it safely. */
function stateMeta(state: string) {
  const meta = (taskStateMeta as Record<string, { label: string; tone: Tone } | undefined>)[state]
  return meta ?? { label: '已中断', tone: 'warn' as const }
}

type Filter = 'all' | 'active' | 'done' | 'failed'
const filter = ref<Filter>('all')
const list = computed(() => {
  const t = store.tasks
  if (filter.value === 'active') return t.filter((x) => ACTIVE_TASK_STATES.includes(x.state))
  if (filter.value === 'done') return t.filter((x) => ['succeeded', 'stopped', 'interrupted'].includes(x.state))
  if (filter.value === 'failed') return t.filter((x) => ['failed', 'connection_lost'].includes(x.state))
  return t
})

const selectedId = computed<string | null>(() => {
  const p = route.params.taskId
  const id = typeof p === 'string' && p ? p : null
  return (id && store.tasks.some(t => t.task_id === id) ? id : null) ?? store.tasks[0]?.task_id ?? null
})
const task = computed(() => store.tasks.find((t) => t.task_id === selectedId.value) ?? null)

function select(t: TaskSummary) {
  void router.replace({ name: 'tasks', params: { taskId: t.task_id } })
}

/* A menu owns its target; changing selection must never redirect an action. */
const menu = ref<{ target: TaskSummary; x: number; y: number } | null>(null)
const menuEl = ref<HTMLElement | null>(null)
let menuAnchor: HTMLElement | null = null
const deleteTarget = ref<TaskSummary | null>(null)
const historyBusy = ref(false)
const exportSource = ref<TrainingExportSource | null>(null)
const deletable = (t: TaskSummary) => ['succeeded', 'failed', 'stopped'].includes(t.state)
function closeMenu() { menu.value = null }
async function openMenu(t: TaskSummary, event: MouseEvent | KeyboardEvent) {
  event.preventDefault()
  menuAnchor = event.currentTarget as HTMLElement
  const rect = menuAnchor.getBoundingClientRect()
  const x = event instanceof MouseEvent ? event.clientX : rect.left
  const y = event instanceof MouseEvent ? event.clientY : rect.bottom
  menu.value = { target: t, x, y }
  await nextTick()
  const bounds = menuEl.value?.getBoundingClientRect()
  if (menu.value && bounds) {
    menu.value.x = Math.max(8, Math.min(x, window.innerWidth - bounds.width - 8))
    menu.value.y = Math.max(8, Math.min(y, window.innerHeight - bounds.height - 8))
    menuEl.value?.querySelector<HTMLButtonElement>('button:not(:disabled)')?.focus()
  }
}
function outsideMenu(event: PointerEvent) {
  if (!menuEl.value?.contains(event.target as Node)) closeMenu()
}
function menuScroll(event: Event) {
  // Live log autoscroll is unrelated to the menu's anchor.
  const target = event.target
  if (target instanceof Node && menuAnchor && target.contains(menuAnchor)) closeMenu()
}
function menuKey(event: KeyboardEvent) {
  if (!menu.value) return
  if (event.key === 'Escape') { event.preventDefault(); closeMenu(); menuAnchor?.focus(); return }
  if (event.key === 'Tab') { closeMenu(); return }
  if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
  event.preventDefault()
  const buttons = Array.from(menuEl.value?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])
  const index = buttons.indexOf(document.activeElement as HTMLButtonElement)
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 :
    (index + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length
  buttons[next]?.focus()
}
async function historyAction(kind: 'export' | 'use' | 'delete', target: TaskSummary) {
  closeMenu()
  if (historyBusy.value) return
  if (kind === 'delete') { if (deletable(target)) deleteTarget.value = target; return }
  historyBusy.value = true
  try {
    if (kind === 'use') {
      await router.push({ name: 'train', query: { fromTask: target.task_id } })
    } else {
      const config = await api.tasks.configuration(target.task_id)
      exportSource.value = snapshotTrainingExport(config, config.engine_id, config.dataset_path)
    }
  } catch (err) { toastError(err, kind === 'use' ? '使用配置' : '导出配置') }
  finally { historyBusy.value = false }
}
async function deleteTask() {
  const target = deleteTarget.value
  if (!target || historyBusy.value) return
  historyBusy.value = true
  try {
    await api.tasks.delete(target.task_id)
    removeTask(target.task_id)
    deleteTarget.value = null
    if (route.params.taskId === target.task_id) await router.replace({ name: 'tasks' })
    toast('已删除任务记录及日志，训练产物保留', 'ok')
  } catch (err) { toastError(err, '删除任务') }
  finally { historyBusy.value = false }
}
onMounted(() => {
  window.addEventListener('pointerdown', outsideMenu)
  window.addEventListener('keydown', menuKey)
  window.addEventListener('resize', closeMenu)
  document.addEventListener('scroll', menuScroll, true)
})
onBeforeUnmount(() => {
  window.removeEventListener('pointerdown', outsideMenu)
  window.removeEventListener('keydown', menuKey)
  window.removeEventListener('resize', closeMenu)
  document.removeEventListener('scroll', menuScroll, true)
})

const { points } = useTaskMetrics(selectedId)
const logLimit = ref(500)
api.settings
  .get()
  .then((s) => (logLimit.value = s.log_tail_lines))
  .catch(() => {})
const { lines } = useTaskLog(selectedId, logLimit)

/* Log view */
const logQuery = ref('')
const follow = ref(true)
const logEl = ref<HTMLElement>()
const shownLines = computed(() => {
  const q = logQuery.value.trim().toLowerCase()
  return q ? lines.value.filter((l) => l.toLowerCase().includes(q)) : lines.value
})
watch(
  () => shownLines.value.length,
  async () => {
    if (!follow.value) return
    await nextTick()
    logEl.value?.scrollTo({ top: logEl.value.scrollHeight })
  },
)
function onLogScroll() {
  const el = logEl.value
  if (!el) return
  follow.value = el.scrollHeight - el.scrollTop - el.clientHeight < 24
}
function lineClass(l: string) {
  if (/error|traceback|exception/i.test(l)) return 'is-err'
  if (/warn/i.test(l)) return 'is-warn'
  if (l.startsWith('[supervisor]')) return 'is-sys'
  return ''
}

const chartTable = ref(false)
const chartMetric = ref<'loss' | 'lr'>('loss')
const chartPointCount = computed(() =>
  points.value.filter((point) => typeof point[chartMetric.value] === 'number').length,
)

/* Control */
const confirmKind = ref<'stop' | 'kill' | null>(null)
const acting = ref(false)
const canStop = computed(() => !!task.value && ['queued', 'preparing', 'running', 'connection_lost'].includes(task.value.state))
const canKill = computed(() => !!task.value && ['preparing', 'running', 'stopping'].includes(task.value.state))

/** A graceful stop is shown as a pause only when the adapter also offers a resume. */
const gracefulPause = computed(() => {
  const t = task.value
  return !!t && t.recovery.graceful_stop
    && (t.recovery.resume_full_state || t.recovery.resume_from_weights)
    && ['preparing', 'running'].includes(t.state)
})
const stopAction = computed(() => {
  const t = task.value
  if (!t) return { label: '中断', graceful: false }
  if (t.recovery.graceful_stop
    && (t.recovery.resume_full_state || t.recovery.resume_from_weights)
    && ['preparing', 'running'].includes(t.state)) return { label: '暂停', graceful: true }
  const label = t.state === 'connection_lost' ? '确认已退出' : t.state === 'queued' ? '取消排队' : '中断'
  return { label, graceful: false }
})
const confirmTitle = computed(() => {
  const t = task.value
  if (!t) return ''
  if (t.state === 'connection_lost') return '确认原训练进程已退出？'
  if (confirmKind.value === 'kill') return '强制结束训练进程？'
  return gracefulPause.value ? '请求暂停训练？' : '请求中断训练？'
})

/* Resume: the backend lists saved points and creates a new task from the chosen one. */
/** `interrupted` is written by the backend when an orphaned run is recovered. */
const RESUMABLE_STATES: readonly string[] = ['stopped', 'failed', 'interrupted']
/* Offered for every finished task, also for older records whose recovery flags were
   stored before engines declared them: the dialog asks the backend and explains when
   nothing can be resumed. */
const canResume = computed(() => {
  const t = task.value
  return !!t && RESUMABLE_STATES.includes(t.state)
})
const resumeInfo = ref<ResumeInfo | null>(null)
/* The dialog owns its target: changing the selection must not redirect the action. */
const resumeTarget = ref<TaskSummary | null>(null)
const resumeSource = ref<string | null>(null)
const resumeLoading = ref(false)
const resumeSubmitting = ref(false)
const resumeBusy = computed(() => resumeLoading.value || resumeSubmitting.value)

async function openResume() {
  const t = task.value
  if (!t || resumeBusy.value) return
  resumeLoading.value = true
  try {
    const info = await api.tasks.resumeInfo(t.task_id)
    resumeTarget.value = t
    resumeInfo.value = info
    resumeSource.value = info.sources[0]?.path ?? null
  } catch (err) { toastError(err, '读取续训信息') }
  finally { resumeLoading.value = false }
}
function closeResume() {
  if (resumeSubmitting.value) return
  resumeInfo.value = null
  resumeTarget.value = null
  resumeSource.value = null
}
async function doResume() {
  const t = resumeTarget.value
  const info = resumeInfo.value
  const source = info?.sources.find((s) => s.path === resumeSource.value) ?? null
  if (!t || !source || resumeSubmitting.value) return
  resumeSubmitting.value = true
  try {
    const created = await api.tasks.resume(t.task_id, source.path)
    resumeInfo.value = null
    resumeTarget.value = null
    resumeSource.value = null
    await refreshTasks()
    toast(`已从${source.label}继续训练，新任务已创建`, 'ok')
    await router.push({ name: 'tasks', params: { taskId: created.task_id } })
  } catch (err) { toastError(err, '继续训练') }
  finally { resumeSubmitting.value = false }
}

/** Show the save point in Explorer so the path the dialog lists can be inspected. */
async function revealResumeSource() {
  if (!resumeSource.value) return
  const ok = await revealPath(resumeSource.value)
  if (!ok) toast('仅在桌面窗口中可打开文件夹', 'info')
}

async function doStop() {
  const t = task.value
  const force = confirmKind.value === 'kill'
  const pause = !force && gracefulPause.value
  if (!t) return
  acting.value = true
  try {
    if (t.state === 'connection_lost') {
      await api.tasks.acknowledgeExit(t.task_id)
      toast('已记录人工确认，队列可继续执行', 'ok')
    } else {
      await api.tasks.stop(t.task_id, force)
      toast(force ? '已请求强制结束进程树' : pause ? '已请求暂停，是否可续训取决于训练状态是否保存成功' : '已请求中断，不保证保存 checkpoint', 'ok')
    }
    confirmKind.value = null
  } catch (err) {
    toastError(err, force ? '强制结束' : pause ? '暂停' : '停止')
  } finally {
    acting.value = false
  }
}

async function reveal() {
  if (!task.value) return
  const ok = await revealPath(task.value.output_dir)
  if (!ok) toast('仅在桌面窗口中可打开文件夹', 'info')
}

const ratio = computed(() => (task.value ? progressRatio(task.value.progress.step, task.value.progress.total_steps) : null))
const duration = computed(() => {
  const t = task.value
  if (!t?.started_at) return null
  const end = t.finished_at ? new Date(t.finished_at).getTime() : Date.now()
  return (end - new Date(t.started_at).getTime()) / 1000
})
</script>

<template>
  <div class="tasks">
    <!-- Queue -->
    <aside class="tasks__list panel">
      <div class="list__head">
        <h1 class="title-md">训练任务</h1>
        <span class="spacer" />
        <RouterLink :to="{ name: 'train' }" class="btn btn--sm btn--primary">
          <AppIcon name="plus" :size="14" />
          新建
        </RouterLink>
      </div>
      <div class="list__filter">
        <SegTabs
          v-model="filter"
          :options="[
            { value: 'all', label: '全部' },
            { value: 'active', label: '进行中' },
            { value: 'done', label: '已结束' },
            { value: 'failed', label: '失败' },
          ]"
        />
      </div>
      <div class="list__body scroll">
        <button
          v-for="t in list"
          :key="t.task_id"
          class="titem"
          :class="{ 'is-on': t.task_id === selectedId }"
          @click="select(t)"
          @contextmenu="openMenu(t, $event)"
          @keydown.shift.f10="openMenu(t, $event)"
          @keydown="($event.key === 'ContextMenu') && openMenu(t, $event)"
        >
          <span class="row">
            <strong class="truncate">{{ t.name }}</strong>
            <span class="spacer" />
            <StatusTag v-bind="stateMeta(t.state)" />
          </span>
          <span class="titem__meta">{{ engineName(t.engine_id) }} · {{ t.architecture }} · {{ fmtTime(t.created_at) }}</span>
          <ProgressBar v-if="t.state === 'running'" :ratio="progressRatio(t.progress.step, t.progress.total_steps)" :light="t.task_id === selectedId" />
        </button>
        <p v-if="store.tasksLoaded && !list.length" class="muted list__empty">
          {{ store.loadError ? `加载失败：${store.loadError}` : '没有任务' }}
        </p>
      </div>
    </aside>

    <!-- Detail -->
    <section v-if="task" :key="task.task_id" class="tasks__detail">
      <div class="detail__top panel enter">
        <header class="detail__head">
          <div class="stack" style="gap: 4px; min-width: 0">
            <span class="eyebrow">{{ task.task_id }} · {{ engineName(task.engine_id) }} / {{ task.installation_label }} · {{ task.architecture }}</span>
            <h2 class="title-lg truncate selectable">{{ task.name }}</h2>
          </div>
          <span class="spacer" />
          <StatusTag v-bind="stateMeta(task.state)" />
          <button class="btn btn--ghost btn--sm" :disabled="historyBusy" aria-haspopup="menu" @click="openMenu(task, $event)">更多操作</button>
          <button v-if="canResume" class="btn btn--continue btn--sm" :disabled="resumeBusy" @click="openResume">
            <AppIcon name="play" :size="13" />
            {{ resumeLoading ? '正在读取续训信息…' : resumeSubmitting ? '正在继续…' : '继续' }}
          </button>
          <button
            class="btn btn--sm"
            :class="stopAction.graceful ? 'btn--pause' : 'btn--ghost'"
            :disabled="!canStop || acting"
            @click="confirmKind = 'stop'"
          >
            <AppIcon :name="stopAction.graceful ? 'pause' : 'stop'" :size="13" />
            {{ stopAction.label }}
          </button>
          <button class="btn btn--danger btn--sm" :disabled="!canKill" @click="confirmKind = 'kill'">强制结束</button>
        </header>

        <div v-if="task.error_summary" class="err selectable">
          <AppIcon name="alert" :size="15" />
          <span class="mono">{{ task.error_summary }}</span>
        </div>

        <div class="metrics">
          <div class="m">
            <span class="m__l">Step</span>
            <span class="m__v">{{ fmtNum(task.progress.step) }}<small> / {{ fmtNum(task.progress.total_steps) }}</small></span>
          </div>
          <div class="m">
            <span class="m__l">Epoch</span>
            <span class="m__v">{{ fmtNum(task.progress.epoch) }}</span>
          </div>
          <div class="m">
            <span class="m__l">Loss</span>
            <span class="m__v">{{ fmtLoss(task.progress.loss) }}</span>
          </div>
          <div class="m">
            <span class="m__l">速度</span>
            <span class="m__v">{{ fmtNum(task.progress.it_per_sec, 2) }}<small v-if="task.progress.it_per_sec !== null"> it/s</small></span>
          </div>
          <div class="m">
            <span class="m__l">已用时</span>
            <span class="m__v">{{ fmtDuration(duration) }}</span>
          </div>
          <div class="m">
            <span class="m__l">预计剩余</span>
            <span class="m__v">{{ ACTIVE_TASK_STATES.includes(task.state) ? fmtDuration(task.progress.eta_seconds) : '—' }}</span>
          </div>
        </div>
        <ProgressBar :ratio="ratio" :segments="24" />

        <div class="chart-wrap">
          <div class="row chart-head">
            <span class="label-pill">指标</span>
            <div class="chart-switch" role="tablist" aria-label="选择指标">
              <button
                type="button"
                class="chip"
                :class="{ 'is-on': chartMetric === 'loss' }"
                role="tab"
                :aria-selected="chartMetric === 'loss'"
                @click="chartMetric = 'loss'"
              >
                Loss
              </button>
              <button
                type="button"
                class="chip"
                :class="{ 'is-on': chartMetric === 'lr' }"
                role="tab"
                :aria-selected="chartMetric === 'lr'"
                @click="chartMetric = 'lr'"
              >
                学习率
              </button>
            </div>
            <span class="muted num">{{ chartPointCount }} 个采样点</span>
            <span class="spacer" />
            <button class="chip" :class="{ 'is-on': chartTable }" @click="chartTable = !chartTable">表格</button>
          </div>
          <MetricChart :points="points" :metric="chartMetric" :height="190" :show-table="chartTable" />
        </div>
      </div>

      <div class="detail__bottom">
        <!-- Log -->
        <section class="log panel">
          <div class="log__head">
            <span class="label-pill"><AppIcon name="terminal" :size="12" />stdout.log</span>
            <span class="muted num">尾部 {{ lines.length }} 行</span>
            <span class="spacer" />
            <div class="log__search">
              <AppIcon name="search" :size="13" />
              <input v-model="logQuery" placeholder="搜索日志" spellcheck="false" />
            </div>
            <button class="chip" :class="{ 'is-on': follow }" @click="follow = !follow">跟随</button>
          </div>
          <div ref="logEl" class="log__body scroll selectable" @scroll="onLogScroll">
            <div v-for="(l, i) in shownLines" :key="i" class="log__line" :class="lineClass(l)">{{ l }}</div>
            <div v-if="!shownLines.length" class="muted log__empty">{{ logQuery ? '无匹配行' : '暂无日志' }}</div>
          </div>
        </section>

        <!-- Info -->
        <section class="info panel panel--pad stack">
          <span class="label-pill" style="align-self: flex-start">任务信息</span>
          <dl class="kv selectable">
            <dt>数据集</dt>
            <dd>{{ task.dataset_name ?? '—' }}</dd>
            <dt>创建</dt>
            <dd>{{ fmtTime(task.created_at) }}</dd>
            <dt>开始</dt>
            <dd>{{ fmtTime(task.started_at) }}</dd>
            <dt>结束</dt>
            <dd>{{ fmtTime(task.finished_at) }}</dd>
            <dt>输出</dt>
            <dd class="mono" style="font-size: 11px">{{ task.output_dir }}</dd>
            <template v-if="task.resumed_from">
              <dt>续训自</dt>
              <dd class="mono" style="font-size: 11px">{{ task.resumed_from }}</dd>
            </template>
          </dl>
          <div class="caps">
            <span :class="{ on: task.recovery.graceful_stop }">正常停止</span>
            <span :class="{ on: task.recovery.resume_from_weights }">权重续训</span>
            <span :class="{ on: task.recovery.resume_full_state }">完整状态恢复</span>
          </div>
          <div class="row" style="margin-top: auto; flex-wrap: wrap">
            <button class="btn btn--sm" @click="reveal"><AppIcon name="folder" :size="13" />打开输出目录</button>
            <RouterLink v-if="['succeeded', 'stopped', 'running'].includes(task.state)" :to="{ name: 'results', params: { taskId: task.task_id } }" class="btn btn--sm">
              查看产物
            </RouterLink>
          </div>
        </section>
      </div>
    </section>

    <section v-else class="tasks__detail panel">
      <EmptyState
        icon="activity"
        :title="store.loadError ? `加载失败：${store.loadError}` : store.tasksLoaded ? '还没有训练任务' : '正在读取任务…'"
        :text="store.loadError ? '任务列表未能从本地后端读取；恢复连接后会自动重试。' : '新建训练并提交后，任务会在这里排队和运行。'"
      >
        <RouterLink :to="{ name: 'train' }" class="btn btn--primary">新建训练</RouterLink>
      </EmptyState>
    </section>

    <Teleport to="body">
      <div v-if="menu" ref="menuEl" class="task-menu" role="menu" aria-label="任务操作"
        :style="{ left: `${menu.x}px`, top: `${menu.y}px` }" @contextmenu.prevent>
        <button role="menuitem" :disabled="historyBusy" @click="historyAction('export', menu.target)">导出配置</button>
        <button role="menuitem" :disabled="historyBusy" @click="historyAction('use', menu.target)">使用配置</button>
        <button role="menuitem" class="task-menu__delete" :disabled="historyBusy || !deletable(menu.target)"
          :title="deletable(menu.target) ? '删除历史任务' : '请先结束任务；连接丢失时需确认进程已退出'"
          @click="historyAction('delete', menu.target)">删除</button>
      </div>
    </Teleport>
    <TrainingExportDialog v-if="exportSource" :source="exportSource" @close="exportSource = null" />
    <ModalDialog v-if="deleteTarget" title="删除历史训练任务？" @close="!historyBusy && (deleteTarget = null)">
      <p>删除 <strong>{{ deleteTarget.name }}</strong> 的任务记录、配置快照、运行日志和指标。</p>
      <p class="muted">不会删除数据集、底模、引擎环境、已生成的模型或预览图。此操作不可撤销，建议先导出配置。</p>
      <template #footer>
        <button class="btn btn--ghost" :disabled="historyBusy" @click="deleteTarget = null">取消</button>
        <button class="btn btn--danger" :disabled="historyBusy" @click="deleteTask">{{ historyBusy ? '正在删除…' : '确认删除' }}</button>
      </template>
    </ModalDialog>
    <ModalDialog v-if="resumeInfo" title="继续训练" :width="560" @close="closeResume">
      <template v-if="!resumeInfo.sources.length">
        <p>没有找到可用于 <strong>{{ resumeTarget?.name ?? '该任务' }}</strong> 的续训保存点。</p>
        <p class="muted">{{ resumeInfo.reason || '没有找到可续训的保存点，请确认训练时启用了保存训练状态。' }}</p>
      </template>
      <template v-else>
        <p>从保存点继续训练会创建一个新任务；原任务的记录与产物保持不变。</p>
        <fieldset class="resume-options" :disabled="resumeSubmitting">
          <legend class="resume-legend">选择保存点</legend>
          <label v-for="s in resumeInfo.sources" :key="s.path" class="resume-option" :class="{ 'is-selected': resumeSource === s.path }">
            <input v-model="resumeSource" type="radio" name="task-resume-source" :value="s.path" />
            <span class="resume-option__text">
              <strong>{{ s.label }}</strong>
              <small>{{ s.kind === 'state' ? '完整训练状态' : '模型权重' }}<template v-if="s.step !== null"> · 第 {{ fmtNum(s.step) }} 步</template></small>
              <code>{{ s.path }}</code>
            </span>
          </label>
        </fieldset>
        <p v-if="resumeInfo.reason" class="muted">{{ resumeInfo.reason }}</p>
      </template>
      <template #footer>
        <button class="btn btn--ghost" :disabled="resumeSubmitting" @click="closeResume">{{ resumeInfo.sources.length ? '取消' : '关闭' }}</button>
        <button v-if="resumeSource" class="btn btn--ghost" @click="revealResumeSource">
          <AppIcon name="folder" :size="13" />打开保存点目录
        </button>
        <button v-if="resumeInfo.sources.length" class="btn btn--continue" :disabled="!resumeSource || resumeSubmitting" @click="doResume">
          {{ resumeSubmitting ? '正在继续…' : '继续训练' }}
        </button>
      </template>
    </ModalDialog>
    <ModalDialog v-if="confirmKind && task" :title="confirmTitle" @close="confirmKind = null">
      <template v-if="confirmKind === 'kill'">
        <p>将结束 <strong>{{ task.name }}</strong> 的整个训练进程树。</p>
        <p class="warnbox">
          <AppIcon name="alert" :size="14" />
          如果进程正在写入 checkpoint，文件可能损坏或不完整；未完成的文件不会被登记为可用产物。
        </p>
      </template>
      <template v-else-if="task.state === 'connection_lost'">
        <p>请先在系统任务管理器中核实 <strong>{{ task.name }}</strong> 的原训练进程和子进程均已退出。</p>
        <p class="warnbox">此操作只记录你的人工确认，不会结束未知进程；确认后将解除队列阻塞。原进程仍在运行时不要确认。</p>
      </template>
      <template v-else-if="gracefulPause">
        <p>向 <strong>{{ task.name }}</strong> 发送暂停请求；是否可续训取决于任务是否保存了训练状态。</p>
        <p class="muted">引擎会先尝试保存当前进度；任务结束且保存成功时，可在任务页点击「继续」从保存点创建新的续训任务。</p>
      </template>
      <template v-else>
        <p>向 <strong>{{ task.name }}</strong> 发送中断请求；排队任务直接取消。进程未及时退出时会强制结束。</p>
        <p class="muted">不保证停止时一定生成完整 checkpoint；是否可继续取决于引擎声明的恢复能力。</p>
      </template>
      <template #footer>
        <button class="btn btn--ghost" @click="confirmKind = null">取消</button>
        <button class="btn" :class="confirmKind === 'kill' ? 'btn--danger' : gracefulPause ? 'btn--pause' : 'btn--primary'" :disabled="acting" @click="doStop">
          {{ task.state === 'connection_lost' ? '我已确认进程退出' : confirmKind === 'kill' ? '强制结束' : gracefulPause ? '确认暂停' : '确认中断' }}
        </button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
.task-menu { position: fixed; z-index: 10000; min-width: 168px; padding: 6px; border: 1px solid var(--line); border-radius: var(--radius-md); background: var(--panel-raised); box-shadow: 0 12px 36px rgba(0,0,0,.35); animation: fade-up .16s var(--ease); }
.task-menu button { display: block; width: 100%; padding: 10px 12px; border-radius: var(--radius-sm); text-align: left; color: var(--text); background: transparent; }
.task-menu button:hover:not(:disabled), .task-menu button:focus-visible { background: rgba(255,255,255,.06); outline: 1px solid var(--line); }
.task-menu button:disabled { opacity: .4; cursor: not-allowed; }
.task-menu .task-menu__delete { color: var(--danger); }

/* Transport actions keep the header's translucent pill language instead of solid
   fills: pause = amber tint, continue = blue tint (继续=蓝色). They follow the
   danger variant's shape (tinted surface, no border) so the header stays quiet. */
.btn--pause,
.btn--continue {
  font-weight: 600;
  transition: background 0.15s var(--ease);
}
.btn--pause { background: rgba(217, 165, 58, 0.16); color: var(--chip-warn); }
.btn--pause:hover:not(:disabled) { background: rgba(217, 165, 58, 0.26); }
.btn--continue { background: rgba(63, 127, 208, 0.20); color: #9cc7f5; }
.btn--continue:hover:not(:disabled) { background: rgba(63, 127, 208, 0.32); }
.btn--pause:focus-visible,
.btn--continue:focus-visible { outline: 2px solid var(--accent-info); outline-offset: 2px; }

/* Resume source picker */
.resume-options { display: flex; flex-direction: column; gap: 8px; border: 0; padding: 0; margin: 0; min-width: 0; }
.resume-legend { margin-bottom: 2px; font-size: 11px; color: var(--text-3); }
.resume-option { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border: 1px solid var(--line); border-radius: var(--radius-md); background: var(--field); cursor: pointer; transition: border-color 0.2s var(--ease); }
.resume-option.is-selected { border-color: var(--focus); }
.resume-option input { margin-top: 3px; accent-color: var(--focus); }
.resume-option__text { display: flex; flex-direction: column; gap: 3px; min-width: 0; }
.resume-option__text strong { font-size: 13px; }
.resume-option__text small { color: var(--text-3); font-size: 11px; }
.resume-option__text code { overflow-wrap: anywhere; font-size: 11px; color: var(--text-3); }
.resume-option:focus-within { outline: 2px solid var(--focus); outline-offset: 2px; }

.tasks {
  display: grid;
  grid-template-columns: 290px minmax(0, 1fr);
  gap: var(--gap);
  height: 100%;
}
.tasks__list {
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.list__head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 20px 16px 10px 40px;
}
.list__filter {
  padding: 0 16px 12px 40px;
}
.list__filter :deep(.chip) {
  height: 24px;
  padding: 0 10px;
  font-size: 11px;
}
.list__body {
  flex: 1;
  padding: 0 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.list__empty {
  padding: 12px 30px;
}
.titem {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px 14px;
  border-radius: var(--radius-md);
  text-align: left;
  transition: background 0.15s;
}
.titem:hover {
  background: rgba(255, 255, 255, 0.05);
}
.titem.is-on {
  background: var(--light);
  color: var(--ink);
}
.titem.is-on :deep(.tag) {
  background: rgba(0, 0, 0, 0.1);
  color: var(--ink);
}
.titem__meta {
  font-size: 11px;
  opacity: 0.6;
}

.tasks__detail {
  display: flex;
  flex-direction: column;
  gap: var(--gap);
  min-height: 0;
  min-width: 0;
}
.detail__top {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 22px 26px 18px;
  border-radius: var(--radius-xl);
}
.detail__head {
  display: flex;
  align-items: center;
  gap: 10px;
}
.err {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  background: var(--danger-bg);
  color: var(--danger);
}
.metrics {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
}
.m {
  display: flex;
  flex-direction: column;
}
.m__l {
  font-size: 11px;
  color: var(--text-3);
}
.m__v {
  font-size: 19px;
  font-weight: 600;
  white-space: nowrap;
}
.m__v small {
  font-size: 11px;
  font-weight: 400;
  color: var(--text-3);
}
.chart-wrap {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.chart-head .chip {
  height: 24px;
  font-size: 11px;
}
.chart-switch {
  display: flex;
  gap: 6px;
}

.detail__bottom {
  flex: 1;
  min-height: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 280px;
  gap: var(--gap);
}
.log {
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.log__head {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 14px 16px 10px;
}
.log__head .chip {
  height: 26px;
}
.log__search {
  display: flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  padding: 0 12px;
  border-radius: var(--radius-pill);
  background: var(--field);
  color: var(--text-3);
}
.log__search input {
  width: 120px;
  border: 0;
  background: none;
  font-size: 12px;
}
.log__search:focus-within {
  outline: 2px solid var(--focus);
  outline-offset: 2px;
}
.log__body {
  flex: 1;
  min-height: 120px;
  margin: 0 12px 12px;
  padding: 12px 14px;
  border-radius: var(--radius-md);
  background: var(--code-bg);
  font-family: var(--font-mono);
  font-size: 11.5px;
  line-height: 1.65;
  color: var(--text-2);
}
.log__line {
  white-space: pre-wrap;
  word-break: break-all;
}
.log__line.is-err {
  color: var(--danger);
}
.log__line.is-warn {
  color: var(--warn);
}
.log__line.is-sys {
  color: var(--info);
}
.log__empty {
  font-family: var(--font-sans);
}
.info {
  min-height: 0;
  overflow: auto;
}
.caps {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.caps span {
  padding: 3px 10px;
  border-radius: var(--radius-pill);
  border: 1px dashed var(--line-strong);
  font-size: 11px;
  color: var(--text-3);
  text-decoration: line-through;
}
.caps span.on {
  border-style: solid;
  color: var(--text-2);
  text-decoration: none;
}
.warnbox {
  display: flex;
  gap: 8px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  background: var(--warn-bg);
  color: var(--warn);
  font-size: 12px;
}
.warnbox .icon {
  margin-top: 2px;
}

@media (max-width: 1180px) {
  .tasks {
    grid-template-columns: 240px minmax(0, 1fr);
  }
  .metrics {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
  .detail__bottom {
    grid-template-columns: minmax(0, 1fr);
  }
  .info {
    display: none;
  }
}
</style>
