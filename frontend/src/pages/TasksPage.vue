<script setup lang="ts">
/**
 * Training tasks: queue + detail (metrics, loss curve, log tail, control).
 * Unparseable metrics show as 未知; stop vs. force-kill are distinct actions,
 * and only the recovery operations the adapter declares are offered.
 */
import { computed, nextTick, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import StatusTag from '@/components/StatusTag.vue'
import ProgressBar from '@/components/ProgressBar.vue'
import LossChart from '@/components/LossChart.vue'
import EmptyState from '@/components/EmptyState.vue'
import ModalDialog from '@/components/ModalDialog.vue'
import SegTabs from '@/components/SegTabs.vue'
import { api } from '@/api'
import { revealPath } from '@/api/bridge'
import { store } from '@/features/store'
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
import type { TaskSummary } from '@/types/api'

const route = useRoute()
const router = useRouter()

type Filter = 'all' | 'active' | 'done' | 'failed'
const filter = ref<Filter>('all')
const list = computed(() => {
  const t = store.tasks
  if (filter.value === 'active') return t.filter((x) => ACTIVE_TASK_STATES.includes(x.state))
  if (filter.value === 'done') return t.filter((x) => ['succeeded', 'stopped'].includes(x.state))
  if (filter.value === 'failed') return t.filter((x) => ['failed', 'connection_lost'].includes(x.state))
  return t
})

const selectedId = computed<string | null>(() => {
  const p = route.params.taskId
  const id = typeof p === 'string' && p ? p : null
  return id ?? store.tasks[0]?.task_id ?? null
})
const task = computed(() => store.tasks.find((t) => t.task_id === selectedId.value) ?? null)

function select(t: TaskSummary) {
  void router.replace({ name: 'tasks', params: { taskId: t.task_id } })
}

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

/* Control */
const confirmKind = ref<'stop' | 'kill' | null>(null)
const acting = ref(false)
const canStop = computed(() => !!task.value && ['queued', 'preparing', 'running', 'connection_lost'].includes(task.value.state))
const canKill = computed(() => !!task.value && ['preparing', 'running', 'stopping'].includes(task.value.state))

async function doStop() {
  const t = task.value
  const force = confirmKind.value === 'kill'
  if (!t) return
  acting.value = true
  try {
    if (t.state === 'connection_lost') {
      await api.tasks.acknowledgeExit(t.task_id)
      toast('已记录人工确认，队列可继续执行', 'ok')
    } else {
      await api.tasks.stop(t.task_id, force)
      toast(force ? '已请求强制结束进程树' : '已请求中断，不保证保存 checkpoint', 'ok')
    }
    confirmKind.value = null
  } catch (err) {
    toastError(err, force ? '强制结束' : '停止')
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
        >
          <span class="row">
            <strong class="truncate">{{ t.name }}</strong>
            <span class="spacer" />
            <StatusTag v-bind="taskStateMeta[t.state]" />
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
          <StatusTag v-bind="taskStateMeta[task.state]" />
          <button class="btn btn--ghost btn--sm" :disabled="!canStop || acting" @click="confirmKind = 'stop'">
            <AppIcon name="stop" :size="13" />
            {{ task.state === 'connection_lost' ? '确认已退出' : task.state === 'queued' ? '取消排队' : '中断' }}
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
            <span class="label-pill">Loss</span>
            <span class="muted num">{{ points.length }} 个采样点</span>
            <span class="spacer" />
            <button class="chip" :class="{ 'is-on': chartTable }" @click="chartTable = !chartTable">表格</button>
          </div>
          <LossChart :points="points" :height="190" :show-table="chartTable" />
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

    <ModalDialog v-if="confirmKind && task" :title="task.state === 'connection_lost' ? '确认原训练进程已退出？' : confirmKind === 'kill' ? '强制结束训练进程？' : '请求中断训练？'" @close="confirmKind = null">
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
      <template v-else>
        <p>向 <strong>{{ task.name }}</strong> 发送中断请求；排队任务直接取消。进程未及时退出时会强制结束。</p>
        <p class="muted">不保证停止时一定生成完整 checkpoint；是否可继续取决于引擎声明的恢复能力。</p>
      </template>
      <template #footer>
        <button class="btn btn--ghost" @click="confirmKind = null">取消</button>
        <button class="btn" :class="confirmKind === 'kill' ? 'btn--danger' : 'btn--primary'" :disabled="acting" @click="doStop">
          {{ task.state === 'connection_lost' ? '我已确认进程退出' : confirmKind === 'kill' ? '强制结束' : '确认中断' }}
        </button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
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
  outline: 0;
  background: none;
  font-size: 12px;
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
