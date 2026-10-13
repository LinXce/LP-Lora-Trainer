<script setup lang="ts">
/**
 * Device-style frame from the reference: LP badge + pill nav rail on the
 * left, and a circular "notch" that cuts into the workspace's left edge at the
 * active nav item. The workspace is masked so whichever panel sits on the left
 * edge shows the cut-out.
 */
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from './AppIcon.vue'
import { routes } from '@/router'
import { store } from '@/features/store'
import { ACTIVE_TASK_STATES } from '@/features/format'

const route = useRoute()

const navItems = routes
  .filter((r) => r.meta?.icon && r.name)
  .map((r) => ({ name: r.name as string, title: r.meta!.title!, icon: r.meta!.icon!, bottom: !!r.meta!.bottom }))
const topItems = navItems.filter((i) => !i.bottom)
const bottomItems = navItems.filter((i) => i.bottom)

const shell = ref<HTMLElement>()
const workspace = ref<HTMLElement>()
const itemEls = new Map<string, HTMLElement>()
const notchY = ref<number | null>(null)
let notchFrame = 0
function moveNotch(target: number) {
  cancelAnimationFrame(notchFrame)
  if (notchY.value === null || window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    notchY.value = target
    return
  }
  const from = notchY.value
  const start = performance.now()
  const step = (now: number) => {
    const progress = Math.min(1, (now - start) / 440)
    const eased = 1 - Math.pow(1 - progress, 3)
    // Both the cut-out mask and marker use this interpolated coordinate.
    notchY.value = from + (target - from) * eased
    if (progress < 1) notchFrame = requestAnimationFrame(step)
  }
  notchFrame = requestAnimationFrame(step)
}

const active = computed(() => navItems.find((i) => i.name === route.name) ?? null)

/* Horizontal counterpart of moveNotch: these pages nudge the marker (and its cut-out)
   towards the rail, because their content sits closest to it. Same frame loop and
   easing as the vertical move, so both axes look alike. */
const NOTCH_SHIFT_PAGES: Record<string, number> = { datasets: 12 }
const notchX = ref(0)
let notchXFrame = 0
function moveNotchX(target: number) {
  cancelAnimationFrame(notchXFrame)
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    notchX.value = target
    return
  }
  const from = notchX.value
  const start = performance.now()
  const step = (now: number) => {
    const progress = Math.min(1, (now - start) / 440)
    const eased = 1 - Math.pow(1 - progress, 3)
    notchX.value = from + (target - from) * eased
    if (progress < 1) notchXFrame = requestAnimationFrame(step)
  }
  notchXFrame = requestAnimationFrame(step)
}
watch(active, (item) => moveNotchX(item ? NOTCH_SHIFT_PAGES[item.name] ?? 0 : 0), { immediate: true })

/** RouterLink refs are component instances; keep their root element. */
function setItemRef(name: string, ref: unknown) {
  const el = (ref as { $el?: unknown } | null)?.$el
  if (el instanceof HTMLElement) itemEls.set(name, el)
  else itemEls.delete(name)
}

function measure() {
  const el = active.value ? itemEls.get(active.value.name) : null
  if (!el || !workspace.value) {
    notchY.value = null
    return
  }
  const a = el.getBoundingClientRect()
  const w = workspace.value.getBoundingClientRect()
  moveNotch(a.top + a.height / 2 - w.top)
}

let ro: ResizeObserver | undefined
onMounted(() => {
  ro = new ResizeObserver(measure)
  if (shell.value) ro.observe(shell.value)
  void nextTick(measure)
})
onBeforeUnmount(() => { ro?.disconnect(); cancelAnimationFrame(notchFrame) })
watch(() => route.name, () => void nextTick(measure))

const runningCount = computed(() => store.tasks.filter((t) => ACTIVE_TASK_STATES.includes(t.state)).length)

/* The batch that just finished: while tasks run the badge counts them, and once the
   queue drains it turns into a colour — green when every task of that batch succeeded,
   red when any of them failed or was stopped. */
const activeIds = computed(() => store.tasks.filter((t) => ACTIVE_TASK_STATES.includes(t.state)).map((t) => t.task_id))
const batch = ref<string[]>([])
const batchOutcome = ref<'ok' | 'failed' | null>(null)
watch(activeIds, (ids, previous) => {
  if (ids.length) {
    batch.value = ids
    return
  }
  if (!previous?.length || !batch.value.length) return
  const states = new Map(store.tasks.map((t) => [t.task_id, t.state]))
  const outcome = batch.value.every((id) => states.get(id) === 'succeeded') ? 'ok' : 'failed'
  batch.value = []
  // Sitting on that page already means the result is being read, so no dot for it.
  batchOutcome.value = route.name === 'tasks' ? null : outcome
})
/* Entering the page dismisses whatever the dot reported. */
watch(() => route.name, (name) => {
  if (name === 'tasks') batchOutcome.value = null
})

const connection = computed(() => {
  switch (store.connection) {
    case 'open':
      return { tone: 'ok', label: '已连接本地后端' }
    case 'demo':
      return { tone: 'info', label: '演示模式：示例数据，不执行任何操作' }
    case 'unreachable':
      return { tone: 'danger', label: '无法连接本地后端' }
    case 'retrying':
      return { tone: 'warn', label: '与后端的事件连接中断，正在重连…' }
    default:
      return { tone: 'warn', label: '正在连接本地后端…' }
  }
})

/* Expanded rail with page names; remembered across launches. */
const NAV_KEY = 'lp.nav.expanded'
const expanded = ref(readExpanded())
function readExpanded() {
  try {
    const saved = window.localStorage.getItem(NAV_KEY)
    return saved === null ? true : saved === '1'
  } catch { return true }
}
/* After a click the pointer still rests on the button; keep its hint hidden until it leaves. */
const tipQuiet = ref(false)
function toggleNav() {
  tipQuiet.value = true
  expanded.value = !expanded.value
  try { window.localStorage.setItem(NAV_KEY, expanded.value ? '1' : '0') } catch { /* storage unavailable */ }
}
/** Ctrl+B toggles the menu, except while typing (where it could mean something else). */
function navKey(e: KeyboardEvent) {
  if (!(e.ctrlKey || e.metaKey) || e.altKey || e.shiftKey || e.key.toLowerCase() !== 'b') return
  const target = e.target as HTMLElement | null
  if (target?.closest('input, textarea, select, [contenteditable="true"]')) return
  e.preventDefault()
  toggleNav()
}
onMounted(() => window.addEventListener('keydown', navKey))
onBeforeUnmount(() => window.removeEventListener('keydown', navKey))

const workspaceStyle = computed(() =>
  notchY.value === null ? {} : { '--notch-y': `${notchY.value}px`, '--notch-shift': `${notchX.value}px` },
)
</script>

<template>
  <div ref="shell" class="shell" :class="[{ 'is-expanded': expanded }, active ? `is-${active.name}` : '']">
    <aside class="rail">
      <RouterLink to="/" class="rail__brand" title="LP Lora Trainer">
        <span class="rail__logo" aria-hidden="true">LP</span>
        <span class="rail__brand-name">LP Lora Trainer</span>
      </RouterLink>

      <nav class="rail__pill" aria-label="主导航">
        <div class="rail__group">
          <RouterLink
            v-for="(item, i) in topItems"
            :key="item.name"
            :ref="(el) => setItemRef(item.name, el)"
            :to="{ name: item.name }"
            class="rail__item"
            :class="{ 'is-active': active?.name === item.name }"
            :aria-label="item.title"
          >
            <AppIcon :name="item.icon" :size="15" />
            <span class="rail__label" :style="{ '--i': i }">{{ item.title }}</span>
            <span
              v-if="item.name === 'tasks' && (runningCount || batchOutcome)"
              class="rail__badge"
              :class="runningCount ? 'num' : ['is-result', `is-${batchOutcome}`]"
              :title="runningCount ? `${runningCount} 个任务进行中` : batchOutcome === 'ok' ? '上次提交的任务全部成功' : '上次提交的任务有失败或中断'"
            >{{ runningCount || '' }}</span>
            <span class="rail__tip">{{ item.title }}</span>
          </RouterLink>
        </div>
        <div class="rail__group">
          <RouterLink
            v-for="(item, i) in bottomItems"
            :key="item.name"
            :ref="(el) => setItemRef(item.name, el)"
            :to="{ name: item.name }"
            class="rail__item"
            :class="{ 'is-active': active?.name === item.name }"
            :aria-label="item.title"
          >
            <AppIcon :name="item.icon" :size="15" />
            <span class="rail__label" :style="{ '--i': topItems.length + i }">{{ item.title }}</span>
            <span class="rail__tip">{{ item.title }}</span>
          </RouterLink>
          <div class="rail__toggle-wrap" :class="{ 'is-quiet': tipQuiet }" @mouseleave="tipQuiet = false">
            <button
              type="button"
              class="rail__status"
              :class="`is-${connection.tone}`"
              :aria-expanded="expanded"
              :aria-label="`${expanded ? '收起' : '展开'}菜单（Ctrl+B）`"
              @click="toggleNav"
            >
              <span class="rail__status-core" role="status" :aria-label="connection.label" />
              <!-- Expanded: the connection status, swapped for the collapse hint on hover. -->
              <span class="rail__status-text" aria-hidden="true">
                <span class="rail__status-line rail__status-state">{{ connection.label }}</span>
                <span class="rail__status-line rail__status-hint">收起菜单<kbd>Ctrl B</kbd></span>
              </span>
              <!-- One arrow for both states: it rides the button's right edge and flips direction. -->
              <AppIcon class="rail__status-toggle" name="chevron" :size="13" />
            </button>
            <span class="rail__tip rail__tip--toggle" aria-hidden="true">
              展开菜单<kbd>Ctrl B</kbd>
              <small>{{ connection.label }}</small>
            </span>
          </div>
        </div>
      </nav>
    </aside>

    <!-- The notch: active page marker sitting in the workspace cut-out -->
    <div
      v-if="active && notchY !== null"
      class="notch"
      :style="{ top: `calc(${notchY}px + var(--frame-pad))`, '--notch-shift': `${notchX}px` }"
      aria-hidden="true"
    >
      <Transition name="nav-icon" mode="out-in">
        <AppIcon :key="active.name" :name="active.icon" :size="17" />
      </Transition>
    </div>

    <main ref="workspace" class="workspace" :class="{ 'has-notch': notchY !== null }" :style="workspaceStyle">
      <div v-if="store.connection === 'demo'" class="banner banner--info">
        演示模式 · 当前显示的是前端示例数据，不代表真实引擎或任务状态；所有操作均被禁用。
      </div>
      <div v-else-if="store.connection === 'unreachable'" class="banner banner--danger">
        无法连接本地后端服务。请通过桌面启动入口运行应用；界面会在后端可用后自动重连。
      </div>
      <!-- A failed read must never look like "no data": show it until a request succeeds. -->
      <div v-if="store.loadError" class="banner banner--danger banner--dismissible" role="alert">
        <span class="truncate">加载失败：{{ store.loadError }}</span>
        <span class="spacer" />
        <button class="banner__close" aria-label="关闭错误提示" @click="store.loadError = null">
          <AppIcon name="x" :size="13" />
        </button>
      </div>
      <!-- SSE backpressure: bounded buffering drops the oldest events; say so instead of hiding it. -->
      <div v-if="store.droppedEvents" class="banner banner--warn" role="status">
        事件流积压，已丢弃 {{ store.droppedEvents }} 条过期更新；界面会在重连后自动刷新快照。
      </div>
      <div class="workspace__view">
        <slot />
      </div>
    </main>
  </div>
</template>

<style scoped>
/* Registered so the rail width can transition; the grid column and the notch both follow it. */
@property --rail-cur {
  syntax: '<length>';
  inherits: true;
  initial-value: 60px;
}

.shell {
  --frame-pad: 12px;
  --rail-open: 184px;
  --rail-cur: var(--rail-w);
  --notch-r: 24px;
  --notch-cut: 32px;
  /* How far a page may nudge the marker (and its cut-out) towards the rail. */
  --notch-shift: 0px;

  position: relative;
  display: grid;
  grid-template-columns: var(--rail-cur) minmax(0, 1fr);
  gap: calc(var(--gap) + 4px);
  height: 100%;
  padding: var(--frame-pad);
  background: var(--frame);
  border: 1px solid var(--frame-edge);
  overflow: hidden;
  transition: --rail-cur 420ms var(--ease);
}
.shell.is-expanded {
  --rail-cur: var(--rail-open);
}

/* ---------- Rail ---------- */

.rail {
  display: flex;
  flex-direction: column;
  gap: var(--gap);
  min-height: 0;
}
.rail__brand {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  width: 100%;
  height: var(--rail-w);
  margin: 0;
  flex: none;
  box-sizing: border-box;
  /* Same radius as the nav pill: a circle when collapsed, a matching pill when expanded. */
  border-radius: calc(var(--rail-w) / 2);
  background: var(--panel-strong);
  color: var(--light-strong);
  text-decoration: none;
  overflow: hidden;
  transition: background 0.2s;
}
/* Fixed rail-width column: "LP" stays centred in the circle and never moves while the rail opens.
   The left padding balances the trailing letter-spacing so the glyphs are optically centred. */
.rail__logo {
  display: inline-block;
  flex: none;
  width: var(--rail-w);
  padding-left: 0.08em;
  box-sizing: border-box;
  text-align: center;
  color: var(--light-strong);
  font-size: 15px;
  font-weight: 800;
  line-height: 1;
  letter-spacing: 0.08em;
}
.rail__brand-name {
  min-width: 0;
  max-width: 0;
  opacity: 0;
  transform: translateX(-8px);
  color: var(--light-strong);
  font-size: 13px;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  transition: max-width 220ms var(--ease), opacity 160ms, transform 220ms var(--ease);
}
.rail__brand:hover {
  background: #626262;
}
.shell.is-expanded .rail__brand-name {
  max-width: 140px;
  opacity: 1;
  transform: none;
  transition-delay: 120ms;
}
.rail__pill {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 0;
  padding: 26px 0 6px;
  /* Half the collapsed width: a full pill when narrow, still round when expanded. */
  border-radius: calc(var(--rail-w) / 2);
  background: var(--panel-strong);
}
/* Left-aligned so icons keep their column while the rail widens. */
.rail__group {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 18px;
}
.rail__item {
  position: relative;
  display: flex;
  align-items: center;
  flex: none;
  width: 34px;
  height: 34px;
  margin-left: calc((var(--rail-w) - 34px) / 2);
  padding-left: 9.5px;
  border-radius: 10px;
  color: var(--light);
  text-decoration: none;
  transition: width 420ms var(--ease), background 240ms, color 240ms, transform 300ms var(--ease);
}
.shell.is-expanded .rail__item {
  width: calc(var(--rail-open) - var(--rail-w) + 34px);
}
.rail__item > .icon {
  flex: none;
  transition: opacity 200ms;
}
.rail__item::before {
  /* the small rounded squares of the reference, visible behind the icon */
  content: '';
  position: absolute;
  top: 6px;
  left: 6px;
  width: 22px;
  height: 22px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.06);
  transition: opacity 0.15s;
}
.rail__item:hover {
  background: rgba(255, 255, 255, 0.08);
  color: var(--light-strong);
}
.rail__item.is-active > .icon {
  opacity: 0;
}
/* The active item is represented by the workspace notch; hovering must not add a second highlight. */
.rail__item.is-active:hover {
  background: transparent;
}
.rail__label {
  margin-left: 12px;
  font-size: 13px;
  white-space: nowrap;
  overflow: hidden;
  opacity: 0;
  transform: translateX(-8px);
  pointer-events: none;
  transition: opacity 140ms, transform 140ms;
}
.shell.is-expanded .rail__label {
  opacity: 1;
  transform: none;
  /* Cascade in top to bottom after the rail has started opening. */
  transition: opacity 260ms var(--ease) calc(120ms + var(--i) * 35ms),
    transform 360ms var(--ease) calc(120ms + var(--i) * 35ms);
}
.rail__item.is-active .rail__label {
  color: var(--light-strong);
  font-weight: 600;
}
.rail__item.is-active::before {
  opacity: 0;
}
/* The finishing dot is the same badge, so it only shrinks to a dot. */
.rail__badge.is-result {
  top: 2px;
  left: 30px;
  min-width: 10px;
  width: 10px;
  height: 10px;
  padding: 0;
}
.rail__badge.is-ok {
  background: var(--ok);
}
.rail__badge.is-failed {
  background: var(--danger);
}
.rail__badge {
  position: absolute;
  top: -2px;
  left: 24px;
  min-width: 16px;
  height: 16px;
  padding: 0 4px;
  border-radius: 8px;
  background: var(--light-strong);
  color: var(--ink);
  font-size: 10px;
  font-weight: 700;
  line-height: 16px;
  text-align: center;
}
.rail__tip {
  position: absolute;
  left: calc(100% + 30px);
  top: 50%;
  transform: translate(-4px, -50%);
  padding: 4px 10px;
  border-radius: var(--radius-pill);
  background: var(--light-strong);
  color: var(--ink);
  font-size: 12px;
  font-weight: 600;
  white-space: nowrap;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.15s, transform 0.15s;
  z-index: 20;
}
.rail__item:hover .rail__tip,
.rail__item:focus-visible .rail__tip {
  opacity: 1;
  transform: translate(0, -50%);
}
.rail__item.is-active .rail__tip,
.shell.is-expanded .rail__tip {
  display: none;
}

/* The big circle at the bottom of the rail toggles the expanded menu and doubles as
   the backend connection indicator; expanded, it grows into a pill with the status text. */
.rail__status {
  position: relative;
  display: flex;
  align-items: center;
  flex: none;
  width: 48px;
  height: 48px;
  margin: 4px 0 0 calc((var(--rail-w) - 48px) / 2);
  padding-left: 19px;
  border-radius: 24px;
  background: var(--light);
  color: var(--ink);
  overflow: hidden;
  transition: width 420ms var(--ease), background 0.2s, transform 0.2s var(--ease), box-shadow 0.2s;
}
.rail__status:hover {
  background: var(--light-strong);
  box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.08);
}
.rail__status:active {
  transform: scale(0.94);
}
.shell.is-expanded .rail__status {
  width: calc(var(--rail-open) - var(--rail-w) + 48px);
  /* Keep the status text clear of the arrow. */
  padding-right: 32px;
}
.rail__status-core {
  flex: none;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--ink-soft);
  transition: transform 260ms var(--ease);
}
/* Collapsed + hover: the dot steps aside so the "open" arrow reads clearly. */
.shell:not(.is-expanded) .rail__status:hover .rail__status-core {
  transform: translateX(-5px);
}
.rail__status-text {
  display: grid;
  min-width: 0;
  margin-left: 10px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
  opacity: 0;
  transition: opacity 140ms;
}
.rail__status-line {
  grid-area: 1 / 1;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: opacity 180ms var(--ease), transform 220ms var(--ease);
}
.rail__status-hint {
  display: flex;
  align-items: center;
  gap: 6px;
  opacity: 0;
  transform: translateY(8px);
}
.rail__status:hover .rail__status-state,
.rail__status:focus-visible .rail__status-state {
  opacity: 0;
  transform: translateY(-8px);
}
.rail__status:hover .rail__status-hint,
.rail__status:focus-visible .rail__status-hint {
  opacity: 1;
  transform: none;
}
.shell.is-expanded .rail__status-text {
  opacity: 1;
  transition: opacity 260ms var(--ease) 220ms;
}
/* The arrow is always visible: "›" peeks from the collapsed circle, and once expanded
   it sits at the pill's right end turned to "‹". */
.rail__status-toggle {
  position: absolute;
  top: 50%;
  right: 4px;
  margin-top: -6.5px;
  opacity: 0.45;
  transition: right 420ms var(--ease), opacity 160ms, transform 320ms var(--ease);
}
.rail__status:hover .rail__status-toggle {
  opacity: 1;
  transform: translateX(2px);
}
.shell.is-expanded .rail__status-toggle {
  right: 14px;
  opacity: 0.6;
  transform: rotate(180deg);
}
.shell.is-expanded .rail__status:hover .rail__status-toggle {
  opacity: 1;
  transform: rotate(180deg) translateX(3px);
}
.rail__toggle-wrap {
  position: relative;
}
.rail__tip--toggle {
  left: calc(100% + 22px);
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  padding: 7px 12px;
  border-radius: 14px;
}
.rail__tip--toggle small {
  font-size: 10px;
  font-weight: 500;
  color: var(--ink-soft);
}
.shell:not(.is-expanded) .rail__toggle-wrap:not(.is-quiet):hover .rail__tip--toggle,
.shell:not(.is-expanded) .rail__toggle-wrap:has(:focus-visible) .rail__tip--toggle {
  opacity: 1;
  transform: translate(0, -50%);
}
.rail kbd {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 5px;
  border-radius: 5px;
  background: rgba(0, 0, 0, 0.1);
  font-family: var(--font-sans);
  font-size: 10px;
  font-weight: 600;
  line-height: 14px;
}
.rail__status-hint kbd {
  margin-left: 0;
}
.rail__status.is-ok .rail__status-core {
  background: #3f7a4d;
}
.rail__status.is-warn .rail__status-core {
  background: #9a7426;
  animation: pulse 1.2s infinite;
}
.rail__status.is-danger .rail__status-core {
  background: #a3423a;
}
.rail__status.is-info .rail__status-core {
  background: #4b6788;
}

/* ---------- Notch ---------- */

.notch {
  position: absolute;
  left: calc(var(--frame-pad) + var(--rail-cur) + var(--gap) + 4px - var(--notch-shift));
  width: calc(var(--notch-r) * 2);
  height: calc(var(--notch-r) * 2);
  transform: translate(-50%, -50%);
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--light);
  color: var(--ink);
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
  /* Position is animated with the mask via one requestAnimationFrame loop. */
  z-index: 5;
}

/* The dataset list sits closest to the rail; AppShell shifts the marker there in line
   with NOTCH_SHIFT_PAGES, so no page-specific CSS is needed. */

.nav-icon-enter-active, .nav-icon-leave-active {
  transition: opacity 160ms ease, transform 160ms ease;
}
.nav-icon-enter-from { opacity: 0; transform: scale(.65); }
.nav-icon-leave-to { opacity: 0; transform: scale(.65); }

/* ---------- Workspace ---------- */

.workspace {
  display: flex;
  flex-direction: column;
  gap: var(--gap);
  min-width: 0;
  min-height: 0;
}
.workspace.has-notch {
  -webkit-mask-image: radial-gradient(
    circle at calc(0px - var(--notch-shift)) var(--notch-y),
    transparent calc(var(--notch-cut) - 0.5px),
    #000 var(--notch-cut)
  );
  mask-image: radial-gradient(
    circle at calc(0px - var(--notch-shift)) var(--notch-y),
    transparent calc(var(--notch-cut) - 0.5px),
    #000 var(--notch-cut)
  );
}
.workspace__view {
  flex: 1;
  min-height: 0;
}

.banner {
  flex: none;
  padding: 9px 18px 9px 44px;
  border-radius: var(--radius-pill);
  font-size: 12px;
}
.banner--dismissible {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-right: 10px;
}
.banner--dismissible .truncate {
  min-width: 0;
}
.banner__close {
  display: grid;
  place-items: center;
  width: 24px;
  height: 24px;
  flex: none;
  border-radius: 50%;
  color: inherit;
  opacity: 0.8;
  transition: background 0.15s, opacity 0.15s;
}
.banner__close:hover {
  opacity: 1;
  background: rgba(0, 0, 0, 0.12);
}
.banner--info {
  background: var(--info-bg);
  color: var(--info);
}
.banner--danger {
  background: var(--danger-bg);
  color: var(--danger);
}
.banner--warn {
  background: var(--warn-bg);
  color: var(--warn);
}
</style>
