<script setup lang="ts">
// Overlay scrollbars. Native bars are hidden globally (base.css) because WebView2
// still draws arrowed system bars and pseudo-element scrollbars cannot fade.
// A thumb appears while its scroll area is hovered, scrolled or dragged.
import { onBeforeUnmount, onMounted, ref, shallowRef } from 'vue'

type Axis = 'x' | 'y'
interface Box {
  l: number
  t: number
  r: number
  b: number
}
interface Bar {
  key: string
  el: HTMLElement
  axis: Axis
  frame: Box
  thumb: Box
  ratio: number
}

const TRACK_INSET = 12 // keeps the pill clear of rounded panel corners
const EDGE = 2
const HIT = 12
const MIN_THUMB = 32
const SCROLL_LINGER = 800

const root = ref<HTMLElement | null>(null)
const bars = shallowRef<Bar[]>([])
const dragKey = shallowRef<string | null>(null)

const ids = new WeakMap<HTMLElement, number>()
let nextId = 0
let hovered: HTMLElement[] = []
let lastTarget: EventTarget | null = null
const scrolling = new Map<HTMLElement, number>()
let drag: { el: HTMLElement; axis: Axis; start: number; startScroll: number; ratio: number } | null = null
let frame = 0
let poll = 0

const isScrollable = (v: string) => v === 'auto' || v === 'scroll' || v === 'overlay'

function axesOf(el: HTMLElement): Axis[] {
  const s = getComputedStyle(el)
  const axes: Axis[] = []
  if (isScrollable(s.overflowY) && el.scrollHeight > el.clientHeight + 1) axes.push('y')
  if (isScrollable(s.overflowX) && el.scrollWidth > el.clientWidth + 1) axes.push('x')
  return axes
}

function scrollableAncestors(node: Element | null): HTMLElement[] {
  const list: HTMLElement[] = []
  for (let el = node; el && el !== document.body; el = el.parentElement) {
    if (el instanceof HTMLElement && axesOf(el).length) list.push(el)
  }
  return list
}

function clientBox(el: HTMLElement): Box {
  const rect = el.getBoundingClientRect()
  const l = rect.left + el.clientLeft
  const t = rect.top + el.clientTop
  return { l, t, r: l + el.clientWidth, b: t + el.clientHeight }
}

// Visible part of the scroll area after clipping by overflowing ancestors.
function clipOf(el: HTMLElement, box: Box): Box | null {
  const clip = { l: Math.max(box.l, 0), t: Math.max(box.t, 0), r: Math.min(box.r, innerWidth), b: Math.min(box.b, innerHeight) }
  for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
    const s = getComputedStyle(p)
    if (s.overflowX === 'visible' && s.overflowY === 'visible') continue
    const pb = clientBox(p)
    clip.l = Math.max(clip.l, pb.l)
    clip.t = Math.max(clip.t, pb.t)
    clip.r = Math.min(clip.r, pb.r)
    clip.b = Math.min(clip.b, pb.b)
  }
  return clip.r > clip.l && clip.b > clip.t ? clip : null
}

function measure(el: HTMLElement, axis: Axis, both: boolean): Bar | null {
  const box = clientBox(el)
  const clip = clipOf(el, box)
  if (!clip) return null

  const vertical = axis === 'y'
  const viewport = vertical ? el.clientHeight : el.clientWidth
  const content = vertical ? el.scrollHeight : el.scrollWidth
  const scroll = vertical ? el.scrollTop : el.scrollLeft
  const track = viewport - TRACK_INSET * 2 - (both ? HIT : 0)
  if (track <= MIN_THUMB) return null

  const size = Math.max(MIN_THUMB, (track * viewport) / content)
  const max = content - viewport
  const offset = max > 0 ? (Math.min(Math.max(scroll, 0), max) / max) * (track - size) : 0
  const l = vertical ? box.r - EDGE - HIT : box.l + TRACK_INSET + offset
  const t = vertical ? box.t + TRACK_INSET + offset : box.b - EDGE - HIT
  const thumb = { l: l - clip.l, t: t - clip.t, r: vertical ? HIT : size, b: vertical ? size : HIT }

  let id = ids.get(el)
  if (id === undefined) ids.set(el, (id = ++nextId))
  return { key: `${id}-${axis}`, el, axis, frame: clip, thumb, ratio: max / (track - size) }
}

function render() {
  frame = 0
  const active = new Set<HTMLElement>([...hovered, ...scrolling.keys()])
  if (drag) active.add(drag.el)

  const next: Bar[] = []
  for (const el of active) {
    if (!el.isConnected) continue
    const axes = axesOf(el)
    for (const axis of axes) {
      const bar = measure(el, axis, axes.length === 2)
      if (bar) next.push(bar)
    }
  }
  bars.value = next

  // Content can grow without any event (e.g. streaming logs); refresh slowly while shown.
  if (next.length && !poll) poll = window.setInterval(schedule, 400)
  else if (!next.length && poll) {
    window.clearInterval(poll)
    poll = 0
  }
}

function schedule() {
  if (!frame) frame = requestAnimationFrame(render)
}

function onPointerMove(event: PointerEvent) {
  const target = event.target
  if (drag || target === lastTarget) return
  if (target instanceof Node && root.value?.contains(target)) return
  lastTarget = target
  hovered = scrollableAncestors(target instanceof Element ? target : null)
  schedule()
}

function onPointerLeave() {
  hovered = []
  lastTarget = null
  schedule()
}

function onScroll(event: Event) {
  const target = event.target
  if (!(target instanceof HTMLElement)) return
  window.clearTimeout(scrolling.get(target))
  scrolling.set(
    target,
    window.setTimeout(() => {
      scrolling.delete(target)
      schedule()
    }, SCROLL_LINGER),
  )
  schedule()
}

function startDrag(bar: Bar, event: PointerEvent) {
  if (event.button !== 0) return
  event.preventDefault()
  ;(event.currentTarget as HTMLElement).setPointerCapture(event.pointerId)
  const vertical = bar.axis === 'y'
  drag = {
    el: bar.el,
    axis: bar.axis,
    start: vertical ? event.clientY : event.clientX,
    startScroll: vertical ? bar.el.scrollTop : bar.el.scrollLeft,
    ratio: bar.ratio,
  }
  dragKey.value = bar.key
}

function onDragMove(event: PointerEvent) {
  if (!drag) return
  if (drag.axis === 'y') drag.el.scrollTop = drag.startScroll + (event.clientY - drag.start) * drag.ratio
  else drag.el.scrollLeft = drag.startScroll + (event.clientX - drag.start) * drag.ratio
}

function endDrag() {
  if (!drag) return
  drag = null
  dragKey.value = null
  lastTarget = null
  schedule()
}

// The thumb sits above the content, so forward wheel input to its scroll area.
function onWheel(bar: Bar, event: WheelEvent) {
  const scale = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? bar.el.clientHeight : 1
  bar.el.scrollBy({ left: event.deltaX * scale, top: event.deltaY * scale })
}

onMounted(() => {
  document.addEventListener('pointermove', onPointerMove, { passive: true })
  document.documentElement.addEventListener('pointerleave', onPointerLeave)
  document.addEventListener('scroll', onScroll, true)
  window.addEventListener('resize', schedule)
})
onBeforeUnmount(() => {
  document.removeEventListener('pointermove', onPointerMove)
  document.documentElement.removeEventListener('pointerleave', onPointerLeave)
  document.removeEventListener('scroll', onScroll, true)
  window.removeEventListener('resize', schedule)
  if (frame) cancelAnimationFrame(frame)
  window.clearInterval(poll)
  for (const timer of scrolling.values()) window.clearTimeout(timer)
})
</script>

<template>
  <div ref="root" class="sb-layer" aria-hidden="true">
    <TransitionGroup name="sb">
      <div
        v-for="bar in bars"
        :key="bar.key"
        class="sb-frame"
        :style="{
          left: `${bar.frame.l}px`,
          top: `${bar.frame.t}px`,
          width: `${bar.frame.r - bar.frame.l}px`,
          height: `${bar.frame.b - bar.frame.t}px`,
        }"
      >
        <div
          class="sb-thumb"
          :class="[`sb-thumb--${bar.axis}`, { 'is-dragging': dragKey === bar.key }]"
          :style="{
            left: `${bar.thumb.l}px`,
            top: `${bar.thumb.t}px`,
            width: `${bar.thumb.r}px`,
            height: `${bar.thumb.b}px`,
          }"
          @pointerdown="startDrag(bar, $event)"
          @pointermove="onDragMove"
          @pointerup="endDrag"
          @pointercancel="endDrag"
          @wheel.passive="onWheel(bar, $event)"
        />
      </div>
    </TransitionGroup>
  </div>
</template>

<style>
.sb-layer {
  position: fixed;
  inset: 0;
  z-index: 90;
  pointer-events: none;
}
.sb-frame {
  position: fixed;
  overflow: hidden;
  pointer-events: none;
}
.sb-thumb {
  position: absolute;
  pointer-events: auto;
  touch-action: none;
}
.sb-thumb::before {
  content: '';
  position: absolute;
  border-radius: 999px;
  background: rgba(214, 214, 214, 0.28);
  transition: background 0.18s ease, width 0.18s var(--ease), height 0.18s var(--ease);
}
.sb-thumb--y::before {
  top: 0;
  bottom: 0;
  right: 3px;
  width: 4px;
}
.sb-thumb--x::before {
  left: 0;
  right: 0;
  bottom: 3px;
  height: 4px;
}
.sb-thumb--y:hover::before,
.sb-thumb--y.is-dragging::before {
  width: 6px;
}
.sb-thumb--x:hover::before,
.sb-thumb--x.is-dragging::before {
  height: 6px;
}
.sb-thumb:hover::before {
  background: rgba(236, 236, 236, 0.55);
}
.sb-thumb.is-dragging::before {
  background: rgba(236, 236, 236, 0.82);
}

.sb-enter-active {
  transition: opacity 0.2s ease;
}
.sb-leave-active {
  transition: opacity 0.4s ease;
}
.sb-enter-from,
.sb-leave-to {
  opacity: 0;
}
</style>
