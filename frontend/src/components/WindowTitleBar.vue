<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import { store } from '@/features/store'
import { ACTIVE_TASK_STATES } from '@/features/format'
/* Brand icon lives inside the Vite root so dev and build resolve identically. */
import logoUrl from '@/assets/logo.png'
import {
  beginWindowResize, getWindowState, hasWindowControls, resolveClose, windowAction,
  type CloseAction, type ClosePrompt, type NativeWindowState, type ResizeEdge, type WindowAction,
} from '@/api/bridge'
import { toastError } from '@/features/toast'
import ModalDialog from '@/components/ModalDialog.vue'

const route = useRoute()
const pageName = computed(() => typeof route.meta.title === 'string' ? route.meta.title : 'LP LoRA Trainer')
const native = ref(false)
const maximized = ref(false)
const resizable = ref(false)
const busy = ref(false)
const edges: ResizeEdge[] = ['n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw']
let disposed = false

function applyState(state: NativeWindowState) {
  maximized.value = state.maximized
  resizable.value = state.resizable
}

async function connect() {
  if (disposed || native.value || !hasWindowControls()) return
  native.value = true
  try {
    const state = await getWindowState()
    if (!disposed) applyState(state)
  } catch (error) {
    toastError(error, '读取窗口状态失败')
  }
}

function stateChanged(event: Event) {
  const state = (event as CustomEvent<NativeWindowState>).detail
  if (state && typeof state.maximized === 'boolean' && typeof state.resizable === 'boolean') {
    applyState(state)
  }
}

/* The launcher owns the close decision (it knows why leaving is refused); the page
   only renders the wording it sends, so both close gestures look and read alike. */
const closePrompt = ref<ClosePrompt | null>(null)

function closeRequested(event: Event) {
  const detail = (event as CustomEvent<ClosePrompt>).detail
  if (detail && typeof detail.title === 'string' && typeof detail.body === 'string') {
    closePrompt.value = detail
  }
}

/* Shown only once the window is really going away: not while the prompt is open, not
   for minimize, not for cancel. The dialog is a native <dialog> (top layer), so this
   overlay always sits underneath it. */
const closing = ref(false)

/* Training keeps running in the supervisor, so closing while a task is active is a
   decision the user must make explicitly (the launcher has no task view). */
const trainingCount = computed(
  () => (store.tasks ?? []).filter((task) => ACTIVE_TASK_STATES.includes(task.state)).length,
)
/* The title bar is visible from every page, so the live run belongs there: it answers
   "what is training right now" without navigating to the task list. */
const runningLabels = computed(() =>
  (store.tasks ?? [])
    .filter((task) => ACTIVE_TASK_STATES.includes(task.state))
    .map((task) => task.name || task.task_id)
    .join(' · '),
)
const windowTitle = computed(() => (runningLabels.value ? `${pageName.value} · ${runningLabels.value}` : pageName.value))

function askClose() {
  if (!trainingCount.value) {
    void closeWithVeil()
    return
  }
  closePrompt.value = {
    reason: 'training',
    title: '训练仍在进行',
    body: `${trainingCount.value} 个任务正在训练。关闭窗口会终止训练，已完成的结果会保留。`,
    yes_label: '终止训练并关闭',
    yes_action: 'close',
    no_label: '取消',
    no_action: 'cancel',
  }
}

async function answerClose(action: CloseAction) {
  const prompt = closePrompt.value
  closePrompt.value = null
  if (!prompt) return
  if (action === 'close') {
    closing.value = true
    // Let the veil paint (and be visible for a moment) before the window disappears;
    // otherwise the close happens within the same frame and nothing is ever seen.
    await nextTick()
    await new Promise((resolve) => window.setTimeout(resolve, 500))
  }
  void resolveClose(action).catch((error) => {
    closing.value = false
    toastError(error, '关闭窗口失败')
  })
}

/** The title-bar ✕ closes natively and instantly, so it shows the veil first: that
    covers the backend shutdown round trip while the window is going away. */
async function closeWithVeil() {
  if (busy.value) return
  // busy is set before the wait, so a second click cannot start a second close.
  busy.value = true
  closing.value = true
  await nextTick()
  await new Promise((resolve) => window.setTimeout(resolve, 420))
  try {
    // Not act(): it starts with the busy guard, which is already set here, so the
    // close request was silently dropped and the veil stayed up forever.
    const state = await windowAction('close')
    if (state && !disposed) applyState(state)
  } catch (error) {
    closing.value = false
    toastError(error, '关闭窗口失败')
  } finally {
    busy.value = false
  }
}

async function act(action: WindowAction) {
  if (busy.value) return
  busy.value = true
  try {
    const state = await windowAction(action)
    if (state && !disposed) applyState(state)
  } catch (error) {
    toastError(error, '窗口操作失败')
  } finally {
    busy.value = false
  }
}

function resize(event: PointerEvent, edge: ResizeEdge) {
  if (event.button !== 0 || maximized.value) return
  event.preventDefault()
  event.stopPropagation()
  void beginWindowResize(edge).catch((error) => toastError(error, '窗口缩放失败'))
}

onMounted(() => {
  // Keep listening: the bridge may arrive after the normal path-dialog timeout.
  window.addEventListener('pywebviewready', connect)
  window.addEventListener('lp-window-state', stateChanged)
  window.addEventListener('lp-close-request', closeRequested)
  void connect()
})
onBeforeUnmount(() => {
  disposed = true
  window.removeEventListener('pywebviewready', connect)
  window.removeEventListener('lp-window-state', stateChanged)
  window.removeEventListener('lp-close-request', closeRequested)
})
</script>

<template>
  <header v-if="native" class="window-bar" aria-label="桌面窗口标题栏">
    <div
      class="window-bar__title"
      :class="{ 'pywebview-drag-region': !maximized }"
      @dblclick.left="act('toggle_maximize')"
    >
      <img class="window-bar__logo" :src="logoUrl" width="22" height="18" alt="LP" draggable="false" />
      <span class="window-bar__divider" aria-hidden="true" />
      <strong class="window-bar__name">{{ windowTitle }}</strong>
    </div>
    <div class="window-bar__controls" role="group" aria-label="窗口控制">
        <button type="button" class="window-bar__button" aria-label="最小化" title="最小化" :disabled="busy" @click="act('minimize')">
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h10" /></svg>
      </button>
      <button
        type="button" class="window-bar__button" :aria-label="maximized ? '还原' : '最大化'"
        :title="maximized ? '还原' : '最大化'" :disabled="busy" @click="act('toggle_maximize')"
      >
        <svg v-if="maximized" viewBox="0 0 16 16" aria-hidden="true"><path d="M6 5V3h7v7h-2M3 6h7v7H3z" /></svg>
        <svg v-else viewBox="0 0 16 16" aria-hidden="true"><path d="M3.5 3.5h9v9h-9z" /></svg>
      </button>
      <button
        type="button" class="window-bar__button window-bar__button--close" aria-label="关闭窗口"
          title="关闭窗口并退出后台" :disabled="busy" @click="askClose"
      >
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 4 8 8m0-8-8 8" /></svg>
      </button>

      <!-- The launcher decides and words the close dialog; the page only renders it. -->
      <ModalDialog v-if="closePrompt" :title="closePrompt.title" :width="460" @close="answerClose('cancel')">
        <p>{{ closePrompt.body }}</p>
        <template #footer>
          <button class="btn btn--ghost" @click="answerClose(closePrompt.no_action)">{{ closePrompt.no_label }}</button>
          <button
            class="btn"
            :class="closePrompt.yes_action === 'close' ? 'btn--danger' : 'btn--primary'"
            @click="answerClose(closePrompt.yes_action)"
          >{{ closePrompt.yes_label }}</button>
        </template>
      </ModalDialog>

    <Teleport to="body">
      <div v-if="closing" class="closing-veil" role="status" aria-live="polite">
        <span class="closing-spinner" aria-hidden="true" />
        <p class="closing-text">正在退出…</p>
      </div>
    </Teleport>
    </div>
  </header>
  <template v-if="native && resizable && !maximized">
    <div
      v-for="edge in edges" :key="edge" class="window-resize" :class="`window-resize--${edge}`"
      aria-hidden="true" @pointerdown="resize($event, edge)"
    />
  </template>
</template>

<style scoped>
.window-bar {
  flex: 0 0 40px;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 8px 0 18px;
  background: var(--frame);
  border: 1px solid var(--frame-edge);
  border-bottom: 0;
  color: var(--text-2);
  user-select: none;
}
.window-bar__title {
  align-self: stretch;
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  letter-spacing: 0.025em;
}
.window-bar__title > * { pointer-events: none; }
.window-bar__logo { flex-shrink: 0; object-fit: contain; }
.window-bar__divider {
  width: 1px;
  height: 18px;
  flex: 0 0 1px;
  background: var(--line-strong);
}
.window-bar__name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--light-strong);
  font-weight: 700;
  letter-spacing: 0.015em;
}
.window-bar__controls { display: flex; gap: 4px; }
.window-bar__button {
  display: grid;
  place-items: center;
  width: 38px;
  height: 28px;
  border-radius: 9px;
  color: var(--text-2);
  transition: background 0.15s var(--ease), color 0.15s var(--ease);
}
.window-bar__button:hover { background: var(--panel-raised); color: var(--text); }
.window-bar__button:active { background: var(--panel-strong); }
.window-bar__button--close:hover { background: var(--danger-bg); color: var(--danger); }
.window-bar__button:disabled { opacity: 0.5; }
.window-bar__button svg { width: 14px; height: 14px; fill: none; stroke: currentColor; stroke-width: 1.3; }
.window-resize { position: fixed; z-index: 1000; touch-action: none; }
.window-resize--n, .window-resize--s { left: 12px; right: 12px; height: 5px; cursor: ns-resize; }
.window-resize--n { top: 0; }
.window-resize--s { bottom: 0; }
.window-resize--e, .window-resize--w { top: 12px; bottom: 12px; width: 5px; cursor: ew-resize; }
.window-resize--e { right: 0; }
.window-resize--w { left: 0; }
.window-resize--ne, .window-resize--nw, .window-resize--se, .window-resize--sw { width: 12px; height: 12px; }
.window-resize--ne, .window-resize--sw { cursor: nesw-resize; }
.window-resize--nw, .window-resize--se { cursor: nwse-resize; }
.window-resize--ne { top: 0; right: 0; }
.window-resize--nw { top: 0; left: 0; }
.window-resize--se { bottom: 0; right: 0; }
.window-resize--sw { bottom: 0; left: 0; }
/* Below the native <dialog> by construction: that element lives in the browser's top
   layer, which no z-index here can or should outrank. */
.closing-veil {
  position: fixed;
  inset: 0;
  /* Above the toast layer (100); native <dialog> still wins through the top layer. */
  z-index: 300;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
  background: color-mix(in srgb, #000 38%, transparent);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
}
.closing-spinner {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 3px solid color-mix(in srgb, #fff 28%, transparent);
  border-top-color: #fff;
  animation: closing-spin 900ms linear infinite;
}
.closing-text {
  margin: 0;
  color: #fff;
  font-size: 13px;
  letter-spacing: .02em;
}
@keyframes closing-spin {
  to { transform: rotate(360deg); }
}
@media (prefers-reduced-motion: reduce) {
  .closing-spinner { animation-duration: 2.4s; }
}
</style>
