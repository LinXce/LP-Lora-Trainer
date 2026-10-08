<script setup lang="ts">
/**
 * Overview — the composition follows the reference most literally:
 * an L-shaped hero (current task) wrapping a nested "engine" card, a right
 * column with a status pill, a stacked task queue and a fanned sample deck.
 */
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import StatusTag from '@/components/StatusTag.vue'
import ProgressBar from '@/components/ProgressBar.vue'
import LossChart from '@/components/LossChart.vue'
import EmptyState from '@/components/EmptyState.vue'
import { store } from '@/features/store'
import { useArtifacts } from '@/features/useArtifacts'
import { useTaskMetrics } from '@/features/useTask'
import {
  ACTIVE_TASK_STATES,
  engineName,
  fmtDuration,
  fmtLoss,
  fmtNum,
  fmtRelative,
  installStateMeta,
  progressRatio,
  shortCommit,
  taskStateMeta,
  verificationMeta,
} from '@/features/format'
import type { TaskState } from '@/types/api'

const router = useRouter()

const focusTask = computed(
  () =>
    store.tasks.find((t) => t.state === 'running') ??
    store.tasks.find((t) => ['preparing', 'stopping', 'validating'].includes(t.state)) ??
    null,
)
const focusId = computed(() => focusTask.value?.task_id ?? null)
const { points } = useTaskMetrics(focusId)

const ratio = computed(() =>
  focusTask.value ? progressRatio(focusTask.value.progress.step, focusTask.value.progress.total_steps) : null,
)

/* Queue stack: active first, then most recent; frontmost (light) card is the most relevant. */
const queue = computed(() => {
  const active = store.tasks.filter((t) => ACTIVE_TASK_STATES.includes(t.state) && t.task_id !== focusId.value)
  const rest = store.tasks.filter((t) => !ACTIVE_TASK_STATES.includes(t.state))
  return [...active, ...rest].slice(0, 3).reverse()
})

const defaultEngine = computed(() => store.engines.find((e) => e.is_default) ?? null)
const enginesNeedingAttention = computed(() => store.engines.filter((e) => e.state !== 'ready').length)

/* Lifecycle strip for the nested card — the dashed segments in the reference. */
const stages: { key: TaskState; label: string }[] = [
  { key: 'validating', label: '校验' },
  { key: 'queued', label: '排队' },
  { key: 'preparing', label: '准备' },
  { key: 'running', label: '运行' },
  { key: 'succeeded', label: '完成' },
]
const stageIndex = computed(() => {
  const s = focusTask.value?.state
  if (!s) return -1
  if (s === 'stopping') return 3
  return stages.findIndex((x) => x.key === s)
})

/* Sample deck: follows the focused task so images appear while it runs. */
const sampleTaskId = computed(() => focusId.value ?? store.tasks[0]?.task_id ?? null)
const { artifacts } = useArtifacts(sampleTaskId)
const samples = computed(() =>
  artifacts.value.filter((a) => a.kind === 'sample' && a.complete).slice(-4).reverse(),
)

const gpu = computed(() => store.system?.gpu ?? null)
const vramRatio = computed(() => (gpu.value ? gpu.value.memory_used_mb / gpu.value.memory_total_mb : null))
</script>

<template>
  <div class="ov">
    <!-- Hero (L-shape) -->
    <section class="ov__hero panel enter">
      <div class="hero__top">
        <template v-if="focusTask">
          <header class="hero__head">
            <div class="stack" style="gap: 6px">
              <span class="eyebrow">当前任务 · {{ engineName(focusTask.engine_id) }} · {{ focusTask.architecture }}</span>
              <h1 class="title-lg truncate selectable">{{ focusTask.name }}</h1>
            </div>
            <span class="spacer" />
            <StatusTag v-bind="taskStateMeta[focusTask.state]" />
            <button class="btn btn--ghost btn--sm" @click="router.push({ name: 'tasks', params: { taskId: focusTask.task_id } })">
              查看详情
              <AppIcon name="chevron" :size="14" />
            </button>
          </header>

          <div class="hero__stats">
            <div class="stat">
              <span class="stat__label">Step</span>
              <span class="stat__value">
                {{ fmtNum(focusTask.progress.step) }}<small> / {{ fmtNum(focusTask.progress.total_steps) }}</small>
              </span>
            </div>
            <div class="stat">
              <span class="stat__label">Loss</span>
              <span class="stat__value">{{ fmtLoss(focusTask.progress.loss) }}</span>
            </div>
            <div class="stat">
              <span class="stat__label">速度</span>
              <span class="stat__value">
                {{ fmtNum(focusTask.progress.it_per_sec, 2) }}<small v-if="focusTask.progress.it_per_sec !== null"> it/s</small>
              </span>
            </div>
            <div class="stat">
              <span class="stat__label">预计剩余</span>
              <span class="stat__value">{{ fmtDuration(focusTask.progress.eta_seconds) }}</span>
            </div>
          </div>
          <ProgressBar :ratio="ratio" />
          <div class="hero__chart">
            <LossChart :points="points" :height="200" />
          </div>
        </template>

        <EmptyState
          v-else
          icon="image"
          :title="store.loadError ? `加载失败：${store.loadError}` : store.tasksLoaded ? '当前没有运行中的训练' : '正在读取任务…'"
          text="训练在独立的监管进程中运行，关闭窗口不会结束训练；再次打开应用时会自动重新连接。"
        />
      </div>

      <div class="hero__bottom">
        <RouterLink :to="{ name: 'train' }" class="launch">
          <span class="launch__badge"><AppIcon name="star" :size="13" /></span>
          <span class="launch__text">新建训练任务…</span>
          <AppIcon name="chevron" :size="14" />
        </RouterLink>
        <div class="hero__mini">
          <div>
            <span class="muted">任务总数</span>
            <strong class="num">{{ store.tasks.length }}</strong>
          </div>
          <div>
            <span class="muted">进行中</span>
            <strong class="num">{{ store.tasks.filter((t) => ACTIVE_TASK_STATES.includes(t.state)).length }}</strong>
          </div>
          <div>
            <span class="muted">已成功</span>
            <strong class="num">{{ store.tasks.filter((t) => t.state === 'succeeded').length }}</strong>
          </div>
        </div>
      </div>
    </section>

    <!-- Nested engine card -->
    <section class="ov__engine panel panel--pad enter" style="animation-delay: 60ms">
      <div class="row">
        <span class="label-pill">默认引擎</span>
        <span class="spacer" />
        <RouterLink :to="{ name: 'engines' }" class="chip">
          管理
          <span v-if="enginesNeedingAttention" class="num">· {{ enginesNeedingAttention }} 项待处理</span>
        </RouterLink>
      </div>

      <template v-if="defaultEngine">
        <p class="engine__name truncate">
          {{ engineName(defaultEngine.engine_id) }}
          <span class="muted"> / {{ defaultEngine.label }}</span>
        </p>
        <div class="engine__chips">
          <span class="chip">{{ shortCommit(defaultEngine.revision.commit) ?? '无 Git 信息' }}</span>
          <span class="chip">Python {{ defaultEngine.python_version ?? '?' }}</span>
          <span class="chip is-on" :class="{ 'is-ok': defaultEngine.state === 'ready' }">{{ installStateMeta[defaultEngine.state].label }}</span>
          <span class="chip">torch {{ defaultEngine.torch_version ?? '?' }}</span>
          <span class="chip">{{ defaultEngine.cuda_wheel ?? 'CUDA ?' }}</span>
          <span class="chip is-on">{{ verificationMeta[defaultEngine.verification].label }}</span>
        </div>
      </template>
      <p v-else class="muted engine__none">
        {{ store.enginesLoaded ? '尚未设置默认安装实例。将引擎复制到 engine/ 目录后在“引擎管理”中刷新。' : '读取中…' }}
      </p>

      <div class="stages" :title="focusTask ? `当前任务阶段：${taskStateMeta[focusTask.state].label}` : '无活动任务'">
        <div v-for="(s, i) in stages" :key="s.key" class="stage" :class="{ 'is-done': i < stageIndex, 'is-now': i === stageIndex }">
          <span class="stage__bar" />
          <span class="stage__label">{{ s.label }}</span>
        </div>
      </div>
    </section>

    <!-- Right column, upper -->
    <div class="ov__side">
      <div class="gpu-pill enter" style="animation-delay: 90ms" :title="gpu ? gpu.name : 'GPU 监控未启用或不可用'">
        <template v-if="gpu">
          <span class="truncate gpu-pill__name">{{ gpu.name.replace('NVIDIA GeForce ', '') }}</span>
          <span class="gpu-pill__meter"><ProgressBar :ratio="vramRatio" /></span>
          <span class="num gpu-pill__val">{{ (gpu.memory_used_mb / 1024).toFixed(1) }}/{{ (gpu.memory_total_mb / 1024).toFixed(0) }} GB</span>
        </template>
        <span v-else class="muted">GPU 状态未知</span>
      </div>

      <div class="deck enter" style="animation-delay: 120ms">
        <RouterLink
          v-for="(t, i) in queue"
          :key="t.task_id"
          :to="{ name: 'tasks', params: { taskId: t.task_id } }"
          class="deck__card"
          :class="{ 'is-front': i === queue.length - 1 }"
          :style="{ zIndex: i + 1 }"
        >
          <span class="deck__knob" :class="`is-${taskStateMeta[t.state].tone}`" />
          <div class="deck__body">
            <span class="deck__name truncate">{{ t.name }}</span>
            <span class="deck__meta">
              {{ taskStateMeta[t.state].label }} · {{ fmtRelative(t.finished_at ?? t.started_at ?? t.created_at) }}
            </span>
            <ProgressBar
              v-if="i === queue.length - 1"
              class="deck__bar"
              :ratio="progressRatio(t.progress.step, t.progress.total_steps)"
              light
            />
          </div>
        </RouterLink>
        <div v-if="!queue.length" class="deck__card is-front deck__empty">
          <span class="deck__knob" />
          <div class="deck__body">
            <span class="deck__name">任务队列为空</span>
            <span class="deck__meta">提交的训练会在这里排队</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Right column, lower: fanned samples -->
    <RouterLink :to="{ name: 'results' }" class="ov__samples panel enter" style="animation-delay: 150ms">
      <div class="samples__head">
        <span class="label-pill">最新采样</span>
        <span class="samples__ring"><AppIcon name="chevron" :size="14" /></span>
      </div>
      <div class="fan">
        <div v-for="(s, i) in samples" :key="s.artifact_id" class="fan__card" :style="{ '--i': samples.length - 1 - i }" :title="s.file_name">
          <img v-if="s.preview_url" :src="s.preview_url" alt="" loading="lazy" />
          <AppIcon v-else name="image" :size="22" />
          <span class="fan__step num">step {{ s.step ?? '?' }}</span>
        </div>
        <div v-if="!samples.length" class="fan__card fan__empty" style="--i: 0">
          <span class="muted">暂无采样图</span>
        </div>
      </div>
    </RouterLink>
  </div>
</template>

<style scoped>
.ov {
  --bottom-h: 214px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.05fr) 300px;
  grid-template-rows: minmax(0, 1fr) var(--bottom-h);
  gap: var(--gap);
  height: 100%;
}

/* ---------- Hero ---------- */

.ov__hero {
  grid-column: 1 / 3;
  grid-row: 1 / 3;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.05fr);
  grid-template-rows: minmax(0, 1fr) var(--bottom-h);
  column-gap: var(--gap);
  border-radius: var(--radius-xl);
  overflow: hidden;
}
.hero__top {
  grid-column: 1 / 3;
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 26px 30px 18px 44px;
  min-height: 0;
}
.hero__head {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}
.hero__stats {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.stat {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.stat__label {
  font-size: 11px;
  color: var(--text-3);
}
.stat__value {
  font-size: 22px;
  font-weight: 600;
  letter-spacing: -0.01em;
}
.stat__value small {
  font-size: 12px;
  font-weight: 400;
  color: var(--text-3);
}
.hero__chart {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: flex-end;
}
.hero__bottom {
  grid-column: 1;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
  gap: 16px;
  padding: 0 24px 22px 24px;
}
.hero__mini {
  display: flex;
  gap: 22px;
  padding: 0 8px;
}
.hero__mini div {
  display: flex;
  flex-direction: column;
  font-size: 11px;
}
.hero__mini strong {
  font-size: 18px;
  font-weight: 600;
  color: var(--text);
}

/* The bottom-left pill with a dark sparkle badge, as in the reference. */
.launch {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 46px;
  padding: 0 16px 0 6px;
  border-radius: var(--radius-pill);
  background: var(--light);
  color: var(--ink);
  text-decoration: none;
  font-weight: 600;
  transition: background 0.15s, transform 0.15s;
}
.launch:hover {
  background: var(--light-strong);
  transform: translateY(-1px);
}
.launch__badge {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  background: var(--frame);
  color: var(--light-strong);
  box-shadow: 0 0 0 2px var(--light-strong);
}
.launch__text {
  flex: 1;
}

/* ---------- Nested engine card ---------- */

.ov__engine {
  grid-column: 2;
  grid-row: 2;
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: var(--panel);
  border-radius: var(--radius-lg);
  /* The frame-coloured ring carves the inverted corner into the hero. */
  box-shadow: 0 0 0 var(--gap) var(--frame);
  filter: brightness(1.04);
}
.ov__engine a.chip {
  text-decoration: none;
}
.engine__name {
  font-size: 15px;
  font-weight: 600;
}
.engine__chips {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
}
.engine__chips .chip {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
}
.engine__chips .chip.is-ok {
  background: var(--ok);
  border-color: var(--ok);
}
.engine__none {
  flex: 1;
  font-size: 12px;
}
.stages {
  margin-top: auto;
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 6px;
}
.stage {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.stage__bar {
  height: 2px;
  border-radius: 2px;
  background: rgba(255, 255, 255, 0.16);
}
.stage__label {
  font-size: 10px;
  color: var(--text-3);
}
.stage.is-done .stage__bar {
  background: var(--text-2);
}
.stage.is-now .stage__bar {
  background: var(--light-strong);
  height: 3px;
}
.stage.is-now .stage__label {
  color: var(--text);
  font-weight: 600;
}

/* ---------- Right column ---------- */

.ov__side {
  grid-column: 3;
  grid-row: 1;
  display: flex;
  flex-direction: column;
  gap: var(--gap);
  min-height: 0;
}
.gpu-pill {
  flex: none;
  display: flex;
  align-items: center;
  gap: 10px;
  height: 46px;
  padding: 0 18px;
  border-radius: var(--radius-pill);
  background: var(--panel-raised);
  font-size: 12px;
}
.gpu-pill__name {
  flex: none;
  max-width: 90px;
  font-weight: 600;
}
.gpu-pill__meter {
  flex: 1;
  min-width: 30px;
}
.gpu-pill__val {
  color: var(--text-2);
}

/* Stacked task cards */
.deck {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}
.deck__card {
  position: relative;
  display: flex;
  align-items: flex-start;
  gap: 12px;
  min-height: 110px;
  margin-top: -44px;
  padding: 16px 16px 0 6px;
  border-radius: var(--radius-lg);
  background: var(--panel-raised);
  color: var(--text);
  text-decoration: none;
  box-shadow: 0 -6px 18px rgba(0, 0, 0, 0.18);
  transition: transform 0.25s var(--ease), background 0.2s;
}
.deck__card:first-child {
  margin-top: 0;
  background: var(--panel);
}
.deck__card:nth-child(2):not(.is-front) {
  background: #4f4f4f;
}
.deck__card:hover:not(.is-front) {
  transform: translateY(-6px);
}
.deck__card.is-front {
  min-height: 140px;
  padding-top: 20px;
  padding-bottom: 18px;
  background: var(--light);
  color: var(--ink);
}
.deck__knob {
  flex: none;
  width: 22px;
  height: 34px;
  border-radius: 11px;
  border: 1px solid rgba(255, 255, 255, 0.28);
  background: rgba(255, 255, 255, 0.12);
  position: relative;
}
.deck__knob::after {
  content: '';
  position: absolute;
  left: 50%;
  top: 7px;
  width: 6px;
  height: 6px;
  margin-left: -3px;
  border-radius: 50%;
  background: currentColor;
  opacity: 0.6;
}
.deck__knob.is-ok::after {
  background: var(--ok);
  opacity: 1;
}
.deck__knob.is-danger::after {
  background: var(--danger);
  opacity: 1;
}
.deck__knob.is-warn::after {
  background: var(--warn);
  opacity: 1;
}
.deck__knob.is-live::after {
  opacity: 1;
  animation: pulse 1.4s infinite;
}
.is-front .deck__knob {
  border-color: rgba(0, 0, 0, 0.3);
  background: rgba(0, 0, 0, 0.08);
}
.is-front .deck__knob.is-ok::after {
  background: #3f7a4d;
}
.is-front .deck__knob.is-danger::after {
  background: #a3423a;
}
.is-front .deck__knob.is-warn::after {
  background: #9a7426;
}
.deck__body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.deck__name {
  font-weight: 600;
}
.deck__meta {
  font-size: 11px;
  opacity: 0.65;
}
.deck__bar {
  margin-top: 14px;
}
.deck__empty {
  cursor: default;
}

/* Fanned sample deck */
.ov__samples {
  grid-column: 3;
  grid-row: 2;
  position: relative;
  overflow: hidden;
  display: block;
  color: var(--text);
  text-decoration: none;
  border-radius: var(--radius-lg);
}
.samples__head {
  position: relative;
  z-index: 5;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 14px 0 16px;
}
.samples__ring {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 1px solid var(--line-strong);
  color: var(--text-2);
  transition: background 0.15s;
}
.ov__samples:hover .samples__ring {
  background: rgba(255, 255, 255, 0.08);
}
.fan {
  position: absolute;
  inset: 0;
}
.fan__card {
  --i: 0;
  position: absolute;
  right: -18px;
  bottom: -28px;
  width: 150px;
  height: 160px;
  display: grid;
  place-items: center;
  border-radius: 18px;
  overflow: hidden;
  background: color-mix(in srgb, var(--light) calc(100% - var(--i) * 18%), var(--panel));
  color: var(--ink-soft);
  transform-origin: 100% 100%;
  transform: rotate(calc(var(--i) * -14deg)) translateX(calc(var(--i) * -10px));
  box-shadow: -6px 0 14px rgba(0, 0, 0, 0.18);
  transition: transform 0.35s var(--ease);
  z-index: calc(10 - var(--i));
}
.ov__samples:hover .fan__card {
  transform: rotate(calc(var(--i) * -18deg)) translateX(calc(var(--i) * -14px));
}
.fan__card img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}
.fan__step {
  position: absolute;
  left: 12px;
  top: 10px;
  font-size: 10px;
  font-weight: 600;
  color: var(--ink);
  opacity: 0.6;
}
.fan__empty {
  font-size: 12px;
}

@media (max-width: 1180px) {
  .ov {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.05fr) 260px;
  }
  .engine__chips {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
