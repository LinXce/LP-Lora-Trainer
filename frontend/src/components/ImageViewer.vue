<script lang="ts">
export interface ViewerItem {
  /** Large image URL; null shows a placeholder. */
  src: string | null
  title: string
  subtitle?: string
}
</script>

<script setup lang="ts">
/**
 * Full-screen image viewer. `items` is the navigable set (one page of a gallery,
 * or one comparison set); `index` is the open image, or null when closed.
 *
 * The image always opens fitted inside the window (never larger). The wheel
 * zooms around the cursor, dragging pans a zoomed image, double-click toggles
 * fit / 100 %. Arrow keys move between images, Esc or a click outside closes.
 */
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import AppIcon from './AppIcon.vue'

const props = defineProps<{ items: ViewerItem[]; index: number | null }>()
const emit = defineEmits<{ 'update:index': [value: number | null] }>()

const MAX_ZOOM = 8 // times the loaded image's own pixels
const WHEEL_STEP = 1.0015 // per wheel delta unit; smooth for trackpads and notched wheels

const open = computed(() => props.index !== null && !!props.items[props.index])
const item = computed(() => (props.index === null ? null : props.items[props.index] ?? null))
const loaded = ref(false)
const failed = ref(false)
const root = ref<HTMLElement>()
const stage = ref<HTMLElement>()

const stageW = ref(0)
const stageH = ref(0)
const natW = ref(0)
const natH = ref(0)
// Offset of the image centre from the stage centre, in screen pixels.
const tx = ref(0)
const ty = ref(0)
const scale = ref(1)
/** True while the image follows the window size (no manual zoom yet). */
const fitted = ref(true)
const dragging = ref(false)

/** Largest scale that keeps the whole image inside the stage, never above 100 %. */
const fitScale = computed(() => {
  if (!natW.value || !natH.value || !stageW.value || !stageH.value) return 1
  return Math.min(1, stageW.value / natW.value, stageH.value / natH.value)
})
const minScale = computed(() => Math.min(fitScale.value, 0.1))
const percent = computed(() => Math.round(scale.value * 100))

function clampPan() {
  const w = natW.value * scale.value
  const h = natH.value * scale.value
  // Smaller than the stage on an axis: stay centred. Larger: the edge may not pass the stage edge.
  const maxX = Math.max(0, (w - stageW.value) / 2)
  const maxY = Math.max(0, (h - stageH.value) / 2)
  tx.value = Math.min(maxX, Math.max(-maxX, w <= stageW.value ? 0 : tx.value))
  ty.value = Math.min(maxY, Math.max(-maxY, h <= stageH.value ? 0 : ty.value))
}
function fit() {
  scale.value = fitScale.value
  tx.value = 0
  ty.value = 0
  fitted.value = true
}
function zoomTo(next: number, cx = 0, cy = 0) {
  const target = Math.min(MAX_ZOOM, Math.max(minScale.value, next))
  const ratio = target / scale.value
  // Keep the point under the cursor fixed: t' = c - (c - t) * ratio.
  tx.value = cx - (cx - tx.value) * ratio
  ty.value = cy - (cy - ty.value) * ratio
  scale.value = target
  fitted.value = Math.abs(target - fitScale.value) < 0.001
  clampPan()
}
function pointer(e: { clientX: number; clientY: number }) {
  const r = stage.value!.getBoundingClientRect()
  return { x: e.clientX - r.left - r.width / 2, y: e.clientY - r.top - r.height / 2 }
}
function onWheel(e: WheelEvent) {
  if (!loaded.value) return
  const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? stageH.value : 1
  const { x, y } = pointer(e)
  zoomTo(scale.value * Math.pow(WHEEL_STEP, -e.deltaY * unit), x, y)
}
function toggleZoom(e: MouseEvent) {
  if (!loaded.value) return
  if (fitted.value && fitScale.value < 1) {
    const { x, y } = pointer(e)
    zoomTo(1, x, y)
  } else fit()
}

let drag: { x: number; y: number; tx: number; ty: number } | null = null
let moved = false
/** A click on the empty stage closes, but not the click that ends a pan. */
function onStageClick() {
  if (moved) { moved = false; return }
  close()
}
function onPointerDown(e: PointerEvent) {
  if (e.button !== 0 || fitted.value) return
  drag = { x: e.clientX, y: e.clientY, tx: tx.value, ty: ty.value }
  moved = false
  dragging.value = true
  ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
}
function onPointerMove(e: PointerEvent) {
  if (!drag) return
  if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) > 3) moved = true
  tx.value = drag.tx + (e.clientX - drag.x)
  ty.value = drag.ty + (e.clientY - drag.y)
  clampPan()
}
function onPointerUp() {
  drag = null
  dragging.value = false
}

function onLoad(e: Event) {
  const img = e.target as HTMLImageElement
  natW.value = img.naturalWidth
  natH.value = img.naturalHeight
  loaded.value = true
  fit()
}

let observer: ResizeObserver | undefined
function measure() {
  if (!stage.value) return
  stageW.value = stage.value.clientWidth
  stageH.value = stage.value.clientHeight
  if (!loaded.value) return
  if (fitted.value) fit()
  else {
    scale.value = Math.min(MAX_ZOOM, Math.max(minScale.value, scale.value))
    clampPan()
  }
}

watch(() => item.value?.src, () => {
  loaded.value = false
  failed.value = false
  natW.value = natH.value = 0
  fitted.value = true
  tx.value = ty.value = 0
})

function close() { emit('update:index', null) }
function go(delta: number) {
  if (props.index === null || props.items.length < 2) return
  emit('update:index', (props.index + delta + props.items.length) % props.items.length)
}
function onKey(e: KeyboardEvent) {
  if (!open.value) return
  if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close() }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1) }
  else if (e.key === 'ArrowRight') { e.preventDefault(); go(1) }
  else if (e.key === '0') { e.preventDefault(); fit() }
  else if (e.key === '+' || e.key === '=') { e.preventDefault(); zoomTo(scale.value * 1.25) }
  else if (e.key === '-') { e.preventDefault(); zoomTo(scale.value / 1.25) }
}
watch(open, (isOpen) => {
  if (isOpen) {
    window.addEventListener('keydown', onKey, true)
    void nextTick(() => {
      root.value?.focus()
      if (stage.value) {
        observer = new ResizeObserver(measure)
        observer.observe(stage.value)
        measure()
      }
    })
  } else {
    window.removeEventListener('keydown', onKey, true)
    observer?.disconnect()
    observer = undefined
  }
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKey, true)
  observer?.disconnect()
})
</script>

<template>
  <Teleport to="body">
    <Transition name="viewer">
      <div v-if="open && item" ref="root" class="viewer" role="dialog" aria-modal="true" :aria-label="item.title" tabindex="-1">
        <header class="viewer__bar">
          <div class="viewer__title">
            <strong class="truncate selectable">{{ item.title }}</strong>
            <span v-if="item.subtitle" class="muted num">{{ item.subtitle }}</span>
          </div>
          <div v-if="loaded" class="viewer__zoom">
            <button class="viewer__zbtn" aria-label="缩小" title="缩小（-）" @click="zoomTo(scale / 1.25)">−</button>
            <button class="viewer__zval num" title="适应窗口（0）" @click="fit">{{ percent }}%</button>
            <button class="viewer__zbtn" aria-label="放大" title="放大（+）" @click="zoomTo(scale * 1.25)">+</button>
          </div>
          <span class="viewer__count num">{{ (index ?? 0) + 1 }} / {{ items.length }}</span>
          <button class="viewer__btn" aria-label="关闭" @click="close"><AppIcon name="x" :size="16" /></button>
        </header>

        <div
          ref="stage"
          class="viewer__stage"
          :class="{ 'is-pannable': loaded && !fitted, 'is-dragging': dragging }"
          @wheel.prevent="onWheel"
          @pointerdown="onPointerDown"
          @pointermove="onPointerMove"
          @pointerup="onPointerUp"
          @pointercancel="onPointerUp"
          @click.self="onStageClick"
        >
          <img
            v-if="item.src && !failed"
            :key="item.src"
            class="viewer__img"
            :class="{ 'is-loaded': loaded }"
            :style="loaded ? {
              width: `${natW}px`,
              height: `${natH}px`,
              transform: `translate(-50%, -50%) translate(${tx}px, ${ty}px) scale(${scale})`,
            } : undefined"
            :src="item.src"
            :alt="item.title"
            draggable="false"
            decoding="async"
            @load="onLoad"
            @error="failed = true"
            @dblclick="toggleZoom"
          />
          <div v-else class="viewer__ph"><AppIcon name="image" :size="40" /><span>{{ failed ? '图片加载失败' : '没有可显示的图片' }}</span></div>
          <span v-if="item.src && !loaded && !failed" class="viewer__spin" aria-hidden="true" />
        </div>

        <button v-if="items.length > 1" class="viewer__nav is-prev" aria-label="上一张" @click="go(-1)">
          <AppIcon name="chevron" :size="20" class="flip" />
        </button>
        <button v-if="items.length > 1" class="viewer__nav is-next" aria-label="下一张" @click="go(1)">
          <AppIcon name="chevron" :size="20" />
        </button>
        <p v-if="loaded" class="viewer__hint">滚轮缩放 · 拖动平移 · 双击切换 100% / 适应窗口</p>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.viewer {
  position: fixed;
  inset: 0;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  background: rgba(10, 10, 10, 0.9);
  backdrop-filter: blur(6px);
  outline: none;
}
.viewer__bar {
  position: relative;
  z-index: 2;
  flex: none;
  display: flex;
  align-items: center;
  gap: 14px;
  height: 52px;
  padding: 0 14px 0 20px;
  min-width: 0;
}
.viewer__title {
  display: flex;
  align-items: baseline;
  gap: 10px;
  min-width: 0;
  flex: 1;
  font-size: 13px;
}
.viewer__count {
  font-size: 12px;
  color: var(--text-2);
}
.viewer__zoom {
  display: flex;
  align-items: center;
  border-radius: var(--radius-pill);
  background: rgba(255, 255, 255, 0.1);
}
.viewer__zbtn,
.viewer__zval {
  height: 30px;
  color: var(--text);
  font-size: 13px;
}
.viewer__zbtn {
  width: 30px;
  font-size: 16px;
  border-radius: 50%;
}
.viewer__zval {
  min-width: 52px;
  padding: 0 6px;
  font-size: 12px;
}
.viewer__zbtn:hover,
.viewer__zval:hover {
  background: rgba(255, 255, 255, 0.12);
}
.viewer__zval:hover {
  border-radius: var(--radius-pill);
}
.viewer__btn,
.viewer__nav {
  display: grid;
  place-items: center;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  color: var(--text);
  transition: background 0.15s, transform 0.15s;
}
.viewer__btn:hover,
.viewer__nav:hover {
  background: rgba(255, 255, 255, 0.22);
}
.viewer__nav {
  position: absolute;
  top: 50%;
  z-index: 2;
  margin-top: -18px;
}
.viewer__nav.is-prev {
  left: 14px;
}
.viewer__nav.is-next {
  right: 14px;
}
.viewer__nav:active {
  transform: scale(0.92);
}
.flip {
  transform: rotate(180deg);
}
/* The stage is the remaining window area; the image is positioned and sized by script. */
.viewer__stage {
  position: relative;
  flex: 1;
  min-height: 0;
  overflow: hidden;
  touch-action: none;
}
.viewer__stage.is-pannable {
  cursor: grab;
}
.viewer__stage.is-dragging {
  cursor: grabbing;
}
.viewer__img {
  position: absolute;
  left: 50%;
  top: 50%;
  max-width: none;
  max-height: none;
  border-radius: 4px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
  user-select: none;
  opacity: 0;
  transition: opacity 0.2s;
  will-change: transform;
}
.viewer__img.is-loaded {
  opacity: 1;
}
.viewer__ph {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  color: var(--text-3);
  font-size: 13px;
}
.viewer__spin {
  position: absolute;
  left: 50%;
  top: 50%;
  width: 26px;
  height: 26px;
  margin: -13px 0 0 -13px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.2);
  border-top-color: var(--light-strong);
  animation: viewer-spin 0.8s linear infinite;
}
.viewer__hint {
  position: absolute;
  left: 50%;
  bottom: 14px;
  z-index: 2;
  margin: 0;
  padding: 5px 12px;
  border-radius: var(--radius-pill);
  background: rgba(0, 0, 0, 0.5);
  color: var(--text-3);
  font-size: 11px;
  transform: translateX(-50%);
  pointer-events: none;
  animation: viewer-hint 4s forwards;
}
@keyframes viewer-spin {
  to { transform: rotate(360deg); }
}
@keyframes viewer-hint {
  0%, 70% { opacity: 1; }
  100% { opacity: 0; }
}
.viewer-enter-active,
.viewer-leave-active {
  transition: opacity 0.18s;
}
.viewer-enter-from,
.viewer-leave-to {
  opacity: 0;
}
</style>
