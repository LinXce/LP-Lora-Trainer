<script setup lang="ts">
/**
 * Overview — the composition follows the reference most literally:
 * an L-shaped hero (current task) wrapping a nested "engine" card, a right
 * column with a status pill, a stacked task queue and a fanned sample deck.
 */
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import homeImage from '@/assets/home.png'
import loveImage from '@/assets/love.png'
import AppIcon from '@/components/AppIcon.vue'
import StatusTag from '@/components/StatusTag.vue'
import ProgressBar from '@/components/ProgressBar.vue'
import LossChart from '@/components/LossChart.vue'
import EmptyState from '@/components/EmptyState.vue'
import { store } from '@/features/store'
import { useArtifacts } from '@/features/useArtifacts'
import { newestSamples, sampleInfo } from '@/features/samples'
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

/* The card shows the engine the current task used (the newest submission), because that
   is what a follow-up run would pick; with no task it falls back to the first engine. */
const defaultEngine = computed(() => {
  const latest = [...store.tasks].sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))[0]
  return store.engines.find((e) => e.installation_id === latest?.installation_id) ?? store.engines[0] ?? null
})
const enginesNeedingAttention = computed(() => store.engines.filter((e) => e.state !== 'ready').length)
/* The pill names which task the card's engine came from. */
const enginePillLabel = computed(() => {
  if (store.tasks.some((t) => ACTIVE_TASK_STATES.includes(t.state))) return '当前任务'
  return store.tasks.length ? '前一任务' : '引擎'
})

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

/* The engine card orbits a comet only while a run is actually training; preparing
   and stopping already have their own stage feedback and should stay quiet. */
const engineTraining = computed(() => focusTask.value?.state === 'running')

/* The empty-state mascot is the character. Clicking shows the "love" reaction
   while EmptyState shakes the whole tile, then it settles back to home. */
const mascotLoves = ref(false)

/* Sample deck: follows the focused task so images appear while it runs. */
const sampleTaskId = computed(() => focusId.value ?? store.tasks[0]?.task_id ?? null)
const { artifacts } = useArtifacts(sampleTaskId)
// Newest first by creation time; the list itself is in registration (file name) order.
const samples = computed(() => newestSamples(artifacts.value).slice(0, 4))

const gpu = computed(() => store.system?.gpu ?? null)
/* Host RAM: the GPU panel already shows VRAM, and most "out of memory" reports in this
   app come from host/page-locked memory rather than from the card. */
const hostMemory = computed(() => store.system?.memory ?? null)
const memRatio = computed(() => (hostMemory.value ? hostMemory.value.used_mb / hostMemory.value.total_mb : null))
const vramRatio = computed(() => (gpu.value ? gpu.value.memory_used_mb / gpu.value.memory_total_mb : null))
const gpuHistory = ref<number[]>([])
watch(
  () => store.system?.gpu?.utilization ?? null,
  (value) => {
    if (typeof value !== 'number' || !Number.isFinite(value)) return
    gpuHistory.value = [...gpuHistory.value, Math.max(0, Math.min(100, value))].slice(-36)
  },
  { immediate: true },
)
const gpuChartPoints = computed(() => {
  const values = gpuHistory.value.length ? gpuHistory.value : [0]
  const width = 240
  const height = 56
  const xStep = values.length > 1 ? width / (values.length - 1) : width
  return values.map((value, index) => `${(index * xStep).toFixed(1)},${(height - (value / 100) * height).toFixed(1)}`).join(' ')
})
const gpuChartArea = computed(() => `0,56 ${gpuChartPoints.value} 240,56`)
const gpuUtilization = computed(() => gpu.value?.utilization ?? null)
const gpuTemperature = computed(() => gpu.value?.temperature_c ?? null)
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
          :image="mascotLoves ? loveImage : homeImage"
          :art-size="233"
          image-clickable
          :title="store.loadError ? `加载失败：${store.loadError}` : store.tasksLoaded ? '当前没有运行中的训练' : '正在读取任务…'"
          text="训练在独立的监管进程中运行，关闭窗口不会结束训练；再次打开应用时会自动重新连接。"
          @image-click="mascotLoves = true"
          @image-reaction-end="mascotLoves = false"
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
    <section class="ov__engine panel panel--pad enter" :class="{ 'is-training': engineTraining }" style="animation-delay: 60ms">
      <span v-if="engineTraining" class="engine-orbit" aria-hidden="true" />
      <div class="row">
        <span class="label-pill">{{ enginePillLabel }}</span>
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
      <section class="gpu-card enter" style="animation-delay: 90ms" :title="gpu ? gpu.name : 'GPU monitor unavailable'">
        <template v-if="gpu">
          <div class="gpu-card__head">
            <div class="stack" style="gap: 2px; min-width: 0">
              <strong class="truncate gpu-card__name">{{ gpu.name.replace('NVIDIA GeForce ', '') }}</strong>
              <span class="muted gpu-card__caption">GPU 使用率 · {{ gpuHistory.length }} 次采样</span>
            </div>
            <strong class="gpu-card__headline num">{{ gpuUtilization ?? '--' }}%</strong>
          </div>
          <svg class="gpu-card__chart" viewBox="0 0 240 56" preserveAspectRatio="none" role="img" aria-label="GPU usage chart">
            <polygon class="gpu-card__area" :points="gpuChartArea" />
            <polyline class="gpu-card__line" :points="gpuChartPoints" />
          </svg>
          <div class="gpu-card__stats">
            <span><b>显存</b> <em class="num">{{ (gpu.memory_used_mb / 1024).toFixed(1) }}/{{ (gpu.memory_total_mb / 1024).toFixed(1) }} GB</em></span>
            <span><b>温度</b> <em class="num">{{ gpuTemperature ?? '--' }}°C</em></span>
            <span><b>功耗</b> <em class="num">{{ gpu.power_draw_w?.toFixed(0) ?? '--' }}/{{ gpu.power_limit_w?.toFixed(0) ?? '--' }} W</em></span>
            <span><b>风扇</b> <em class="num">{{ gpu.fan_speed_percent ?? '--' }}%</em></span>
          </div>
          <div class="gpu-card__meter"><ProgressBar :ratio="vramRatio" /></div>
        </template>
        <div v-else class="gpu-card__empty">
          <AppIcon name="activity" :size="15" />
          <span class="muted">GPU 监控未启用或不可用</span>
        </div>
      </section>

      <!-- Host memory is a separate resource from VRAM: its own card, white accent. -->
      <section v-if="hostMemory" class="gpu-card gpu-card--memory enter" style="animation-delay: 120ms" title="整机物理内存">
        <div class="gpu-card__head">
          <div class="stack" style="gap: 2px; min-width: 0">
            <strong class="truncate gpu-card__name">系统内存</strong>
            <span class="muted gpu-card__caption">整机物理内存 · 含训练、引擎与系统进程</span>
          </div>
          <strong class="gpu-card__headline num">{{ hostMemory.percent ?? '--' }}%</strong>
        </div>
        <div class="gpu-card__stats">
          <span><b>已用</b> <em class="num">{{ (hostMemory.used_mb / 1024).toFixed(1) }} GB</em></span>
          <span><b>总量</b> <em class="num">{{ (hostMemory.total_mb / 1024).toFixed(1) }} GB</em></span>
        </div>
        <div class="gpu-card__meter"><ProgressBar :ratio="memRatio" /></div>
      </section>

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
          <span class="fan__step num">{{ sampleInfo(s).label }}</span>
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
/* While a run is training, a blue comet with a transparent tail orbits this card.
   The span is masked down to a 2px ring; its oversized child paints the conic
   gradient and rotates clockwise, so the bright head leaves a fading trail. */
.engine-orbit {
  position: absolute;
  inset: 0;
  z-index: 2;
  padding: 2px;
  border-radius: inherit;
  overflow: hidden;
  pointer-events: none;
  -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor;
  mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  mask-composite: exclude;
}
.engine-orbit::before {
  content: '';
  position: absolute;
  left: -50%;
  top: -50%;
  width: 200%;
  height: 200%;
  background: conic-gradient(
    from 0deg,
    rgba(121, 184, 255, 0) 0deg 210deg,
    rgba(121, 184, 255, 0.10) 245deg,
    rgba(121, 184, 255, 0.42) 305deg,
    rgba(121, 184, 255, 0.85) 348deg,
    #d6e8ff 357deg,
    rgba(121, 184, 255, 0) 360deg
  );
  animation: engine-orbit-spin 2.6s linear infinite;
}
.ov__engine.is-training {
  box-shadow: 0 0 0 var(--gap) var(--frame), 0 0 22px rgba(121, 184, 255, 0.16);
}
@keyframes engine-orbit-spin {
  to {
    transform: rotate(360deg);
  }
}
@media (prefers-reduced-motion: reduce) {
  .engine-orbit::before {
    animation: none;
  }
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
.gpu-card {
  flex: none;
  min-height: 166px;
  padding: 14px 16px 12px;
  border-radius: var(--radius-lg);
  background: var(--panel-raised);
  overflow: hidden;
}
.gpu-card__head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
}
.gpu-card__name {
  font-size: 12px;
  font-weight: 700;
}
.gpu-card__caption {
  font-size: 10px;
}
.gpu-card__headline {
  color: var(--light-strong);
  font-size: 18px;
  line-height: 1;
}
.gpu-card__chart {
  display: block;
  width: 100%;
  height: 56px;
  margin: 12px 0 8px;
  overflow: visible;
}
.gpu-card__area {
  fill: rgba(96, 165, 250, 0.15);
}
.gpu-card__line {
  fill: none;
  stroke: var(--accent-info);
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
  vector-effect: non-scaling-stroke;
}
.gpu-card__stats {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 5px 12px;
  color: var(--text-2);
  font-size: 10px;
}
.gpu-card__stats span {
  display: flex;
  justify-content: space-between;
  gap: 6px;
  min-width: 0;
}
.gpu-card__stats b {
  color: var(--text-3);
  font-weight: 500;
}
.gpu-card__stats em {
  color: var(--text-2);
  font-style: normal;
  white-space: nowrap;
}
.gpu-card__meter {
  margin-top: 9px;
}
.gpu-card__empty {
  display: flex;
  align-items: center;
  gap: 8px;
  min-height: 138px;
  font-size: 12px;
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
  background: var(--ok-solid);
}
.is-front .deck__knob.is-danger::after {
  background: var(--danger-solid);
}
.is-front .deck__knob.is-warn::after {
  background: var(--warn-solid);
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
/* Two resources, two colours: the GPU card reads blue, the host-memory card white. */
.gpu-card__headline {
  color: var(--accent-info);
}
.gpu-card--memory {
  border-color: rgba(255, 255, 255, 0.26);
  /* No usage chart here, so the GPU card's fixed height would leave a large gap. */
  min-height: 0;
  gap: 10px;
}
.gpu-card--memory .gpu-card__name,
.gpu-card--memory .gpu-card__headline {
  color: #f5f5f5;
}
</style>
