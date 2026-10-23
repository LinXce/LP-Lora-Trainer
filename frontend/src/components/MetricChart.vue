<script setup lang="ts">
/**
 * Single-series metric-vs-step line chart (SVG, no chart library).
 * 2px line, ~10% area wash, hairline solid grid, crosshair snapping to the
 * nearest step, keyboard focus with ←/→, and a table view toggle.
 *
 * `metric` selects loss or learning rate; points produced by older runs may
 * carry only one of them, so absent values are skipped rather than plotted as 0.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { MetricPoint } from '@/types/api'

type MetricName = 'loss' | 'lr'

const props = withDefaults(defineProps<{
  points: MetricPoint[]
  metric?: MetricName
  height?: number
  showTable?: boolean
}>(), {
  metric: 'loss',
  height: 220,
  showTable: false,
})

const MAX_POINTS = 600
const pad = { top: 14, right: 84, bottom: 26, left: 52 }

const isLr = computed(() => props.metric === 'lr')
const metricLabel = computed(() => (isLr.value ? '学习率' : 'Loss'))

function valueOf(point: MetricPoint): number | null {
  const value = point[props.metric]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

const series = computed(() => props.points.filter((point) => valueOf(point) !== null))

const host = ref<HTMLDivElement>()
const width = ref(600)
let ro: ResizeObserver | undefined
onMounted(() => {
  ro = new ResizeObserver(([e]) => (width.value = Math.max(240, e.contentRect.width)))
  if (host.value) ro.observe(host.value)
})
onBeforeUnmount(() => ro?.disconnect())

/** Bucketed min/max downsampling keeps spikes visible without rendering every step. */
const data = computed<MetricPoint[]>(() => {
  const pts = series.value
  if (pts.length <= MAX_POINTS) return pts
  const bucket = Math.ceil(pts.length / (MAX_POINTS / 2))
  const out: MetricPoint[] = []
  for (let i = 0; i < pts.length; i += bucket) {
    const slice = pts.slice(i, i + bucket)
    let lo = slice[0]
    let hi = slice[0]
    for (const p of slice) {
      const value = valueOf(p)!
      if (value < valueOf(lo)!) lo = p
      if (value > valueOf(hi)!) hi = p
    }
    if (lo.step < hi.step) out.push(lo, hi)
    else if (lo === hi) out.push(lo)
    else out.push(hi, lo)
  }
  const last = pts[pts.length - 1]
  if (out[out.length - 1] !== last) out.push(last)
  return out
})

function niceStep(range: number, target: number): number {
  const raw = range / Math.max(1, target)
  const mag = 10 ** Math.floor(Math.log10(raw))
  const n = raw / mag
  return (n >= 5 ? 10 : n >= 2 ? 5 : n >= 1 ? 2 : 1) * mag
}

const scales = computed(() => {
  const d = data.value
  const innerW = width.value - pad.left - pad.right
  const innerH = props.height - pad.top - pad.bottom
  const x0 = d.length ? d[0].step : 0
  const x1 = d.length ? Math.max(d[d.length - 1].step, x0 + 1) : 1
  let y0 = Infinity
  let y1 = -Infinity
  for (const p of d) {
    const value = valueOf(p)!
    y0 = Math.min(y0, value)
    y1 = Math.max(y1, value)
  }
  if (!Number.isFinite(y0)) {
    y0 = 0
    y1 = 1
  }

  let yStep: number
  if (isLr.value) {
    // A scheduler can move over orders of magnitude; forcing a zero baseline would
    // flatten the whole curve, so pad the actual range instead.
    const span = Math.max(y1 - y0, Math.abs(y1) * 1e-6, 1e-12)
    y0 = Math.max(0, y0 - span * 0.08)
    y1 += span * 0.08
    yStep = (y1 - y0) / 4
  } else {
    yStep = niceStep(Math.max(y1 - y0, 1e-6), 4)
    y0 = Math.floor(y0 / yStep) * yStep
    y1 = Math.ceil(y1 / yStep) * yStep
    if (y1 === y0) y1 = y0 + yStep
  }

  const xs = (s: number) => pad.left + ((s - x0) / (x1 - x0)) * innerW
  const ys = (v: number) => pad.top + (1 - (v - y0) / (y1 - y0)) * innerH

  const yTicks: number[] = []
  for (let i = 0; i <= 4; i++) {
    yTicks.push(isLr.value ? y0 + i * yStep : +(y0 + i * yStep).toFixed(8))
  }
  const xStep = niceStep(x1 - x0, Math.max(2, Math.floor(innerW / 110)))
  const xTicks: number[] = []
  for (let v = Math.ceil(x0 / xStep) * xStep; v <= x1; v += xStep) xTicks.push(v)

  const decimals = Math.max(0, -Math.floor(Math.log10(yStep)))
  return { xs, ys, yTicks, xTicks, innerW, innerH, decimals, baseY: pad.top + innerH }
})

const linePath = computed(() => {
  const { xs, ys } = scales.value
  return data.value
    .map((p, i) => `${i ? 'L' : 'M'}${xs(p.step).toFixed(1)},${ys(valueOf(p)!).toFixed(1)}`)
    .join('')
})

const areaPath = computed(() => {
  const d = data.value
  if (d.length < 2) return ''
  const { xs, baseY } = scales.value
  return `${linePath.value}L${xs(d[d.length - 1].step).toFixed(1)},${baseY}L${xs(d[0].step).toFixed(1)},${baseY}Z`
})

const last = computed(() => data.value[data.value.length - 1] ?? null)
const lastValue = computed(() => (last.value ? valueOf(last.value) : null))

/**
 * Moving average over the visible loss series. The raw loss is noisy per step; the
 * average is what tells an operator whether the run is converging. A scheduler
 * value is already a smooth curve, so no average is drawn for learning rate.
 */
const averagePath = computed(() => {
  if (isLr.value) return ''
  const d = data.value
  if (d.length < 4) return ''
  const window = Math.max(3, Math.round(d.length * 0.04))
  const { xs, ys } = scales.value
  const parts: string[] = []
  let sum = 0
  for (let i = 0; i < d.length; i++) {
    sum += valueOf(d[i])!
    if (i >= window) sum -= valueOf(d[i - window])!
    if (i < window - 1) continue
    const value = sum / window
    parts.push(`${parts.length ? 'L' : 'M'}${xs(d[i].step).toFixed(1)},${ys(value).toFixed(1)}`)
  }
  return parts.join('')
})

/* ---------- Hover / focus ---------- */

const hoverIndex = ref<number | null>(null)
const hovered = computed(() => (hoverIndex.value === null ? null : data.value[hoverIndex.value] ?? null))

function nearestIndex(px: number): number {
  const d = data.value
  const { xs } = scales.value
  let lo = 0
  let hi = d.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (xs(d[mid].step) < px) lo = mid
    else hi = mid
  }
  return Math.abs(xs(d[lo].step) - px) <= Math.abs(xs(d[hi].step) - px) ? lo : hi
}

function onMove(e: PointerEvent) {
  if (!data.value.length) return
  const rect = (e.currentTarget as SVGElement).getBoundingClientRect()
  hoverIndex.value = nearestIndex(e.clientX - rect.left)
}

function onKey(e: KeyboardEvent) {
  const n = data.value.length
  if (!n) return
  const cur = hoverIndex.value ?? n - 1
  const jump = e.shiftKey ? 10 : 1
  if (e.key === 'ArrowLeft') hoverIndex.value = Math.max(0, cur - jump)
  else if (e.key === 'ArrowRight') hoverIndex.value = Math.min(n - 1, cur + jump)
  else if (e.key === 'Home') hoverIndex.value = 0
  else if (e.key === 'End') hoverIndex.value = n - 1
  else return
  e.preventDefault()
}

const tooltipStyle = computed(() => {
  if (!hovered.value) return {}
  const { xs, ys } = scales.value
  const x = xs(hovered.value.step)
  const flip = x > width.value - 150
  return {
    left: `${x}px`,
    top: `${ys(valueOf(hovered.value)!)}px`,
    transform: `translate(${flip ? 'calc(-100% - 14px)' : '14px'}, -50%)`,
  }
})

function fmtSci(value: number): string {
  if (!value) return '0'
  return value.toExponential(1).replace(/\.0e/, 'e')
}

function fmtValue(value: number): string {
  return isLr.value ? value.toExponential(2) : value.toFixed(4)
}

const fmtY = (v: number) => (isLr.value ? fmtSci(v) : v.toFixed(scales.value.decimals))
const fmtStep = (v: number) => v.toLocaleString('en-US')

/* Table view: at most ~50 evenly spaced rows plus the last point. */
const tableRows = computed(() => {
  const pts = series.value
  if (pts.length <= 50) return pts
  const stride = Math.ceil(pts.length / 50)
  const rows = pts.filter((_, i) => i % stride === 0)
  if (rows[rows.length - 1] !== pts[pts.length - 1]) rows.push(pts[pts.length - 1])
  return rows
})
</script>

<template>
  <div ref="host" class="chart">
    <template v-if="!showTable">
      <svg
        :width="width"
        :height="height"
        class="chart__svg"
        role="img"
        tabindex="0"
        :aria-label="last && lastValue !== null ? `${metricLabel} 曲线，最新 step ${last.step}，${metricLabel} ${fmtValue(lastValue)}` : `${metricLabel} 曲线，暂无数据`"
        @pointermove="onMove"
        @pointerleave="hoverIndex = null"
        @keydown="onKey"
        @blur="hoverIndex = null"
      >
        <!-- Grid + axes -->
        <g class="chart__grid">
          <line
            v-for="t in scales.yTicks"
            :key="`y${t}`"
            :x1="pad.left"
            :x2="pad.left + scales.innerW"
            :y1="scales.ys(t)"
            :y2="scales.ys(t)"
          />
        </g>
        <g class="chart__axis">
          <text v-for="t in scales.yTicks" :key="`yl${t}`" :x="pad.left - 10" :y="scales.ys(t)" text-anchor="end" dominant-baseline="middle">
            {{ fmtY(t) }}
          </text>
          <text
            v-for="t in scales.xTicks"
            :key="`xl${t}`"
            :x="scales.xs(t)"
            :y="height - 6"
            text-anchor="middle"
          >
            {{ fmtStep(t) }}
          </text>
        </g>

        <template v-if="data.length">
          <path class="chart__area" :class="{ 'chart__area--lr': isLr }" :d="areaPath" />
          <path class="chart__line" :class="{ 'chart__line--lr': isLr }" :d="linePath" />
          <path v-if="averagePath" class="chart__avg" :d="averagePath" />

          <!-- End marker + direct end label -->
          <g v-if="last && lastValue !== null">
            <circle class="chart__dot" :class="{ 'chart__dot--lr': isLr }" :cx="scales.xs(last.step)" :cy="scales.ys(lastValue)" r="4" />
            <text class="chart__end" :x="scales.xs(last.step) + 10" :y="scales.ys(lastValue)" dominant-baseline="middle">
              {{ fmtValue(lastValue) }}
            </text>
          </g>

          <!-- Crosshair -->
          <g v-if="hovered">
            <line
              class="chart__cross"
              :x1="scales.xs(hovered.step)"
              :x2="scales.xs(hovered.step)"
              :y1="pad.top"
              :y2="scales.baseY"
            />
            <circle class="chart__dot" :class="{ 'chart__dot--lr': isLr }" :cx="scales.xs(hovered.step)" :cy="scales.ys(valueOf(hovered)!)" r="4.5" />
          </g>
        </template>

        <rect class="chart__hit" :x="pad.left" :y="0" :width="scales.innerW" :height="height" />
      </svg>

      <div v-if="hovered" class="chart__tip" :style="tooltipStyle">
        <strong class="num">{{ fmtValue(valueOf(hovered)!) }}</strong>
        <span class="chart__tip-row"><i class="chart__key" />{{ metricLabel }} · step {{ fmtStep(hovered.step) }}</span>
      </div>
      <div v-if="!data.length" class="chart__empty">暂无可解析的 {{ metricLabel }} 数据</div>
    </template>

    <div v-else class="chart__table scroll" :style="{ height: `${height}px` }">
      <table class="table num">
        <thead>
          <tr><th>Step</th><th>{{ metricLabel }}</th></tr>
        </thead>
        <tbody>
          <tr v-for="(p, index) in tableRows" :key="`${p.step}-${index}`">
            <td>{{ fmtStep(p.step) }}</td>
            <td>{{ fmtValue(valueOf(p)!) }}</td>
          </tr>
          <tr v-if="!tableRows.length"><td colspan="2" class="muted">暂无数据</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.chart {
  position: relative;
  width: 100%;
  min-width: 0;
}
.chart__svg {
  display: block;
  outline: none;
  touch-action: none;
}
.chart__svg:focus-visible {
  outline: 2px solid var(--focus);
  outline-offset: 2px;
  border-radius: 8px;
}
.chart__grid line {
  stroke: rgba(255, 255, 255, 0.08);
  stroke-width: 1;
  shape-rendering: crispEdges;
}
.chart__axis text {
  fill: var(--text-3);
  font-size: 10.5px;
  font-variant-numeric: tabular-nums;
}
.chart__line {
  fill: none;
  stroke: var(--light-strong);
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
}
.chart__line--lr {
  stroke: var(--accent-info);
}
/* Moving average: the run's direction, drawn over the raw per-step noise. */
.chart__avg {
  fill: none;
  stroke: var(--accent-info);
  stroke-width: 2;
  stroke-linejoin: round;
  stroke-linecap: round;
  opacity: 0.95;
}
.chart__area {
  fill: var(--light-strong);
  opacity: 0.1;
}
.chart__area--lr {
  fill: var(--accent-info);
  opacity: 0.12;
}
.chart__dot {
  fill: var(--light-strong);
  stroke: var(--panel);
  stroke-width: 2;
}
.chart__dot--lr {
  fill: var(--accent-info);
}
.chart__end {
  fill: var(--text);
  font-size: 11px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.chart__cross {
  stroke: rgba(255, 255, 255, 0.35);
  stroke-width: 1;
  shape-rendering: crispEdges;
}
.chart__hit {
  fill: transparent;
  cursor: crosshair;
}
.chart__tip {
  position: absolute;
  pointer-events: none;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  background: var(--light-strong);
  color: var(--ink);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
  white-space: nowrap;
  z-index: 2;
}
.chart__tip strong {
  font-size: 14px;
}
.chart__tip-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--ink-soft);
}
.chart__key {
  width: 12px;
  height: 2px;
  border-radius: 2px;
  background: var(--ink);
}
.chart__empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: var(--text-3);
  font-size: 12px;
  pointer-events: none;
}
.chart__table {
  border-radius: var(--radius-sm);
  background: var(--field);
}
.chart__table .table th {
  background: #2c2c2c;
}
</style>
