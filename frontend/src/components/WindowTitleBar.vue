<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
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
  if (!native.value) return
  const detail = (event as CustomEvent<ClosePrompt>).detail
  if (detail && typeof detail.title === 'string' && typeof detail.body === 'string') {
    event.preventDefault() // acknowledge receipt to the native fallback
    closeProgress.value = null
    closePrompt.value = detail
  }
}

const closeProgress = ref<{ message: string; cancellable: boolean } | null>(null)
const closingVeil = ref<HTMLDialogElement | null>(null)
watch(closingVeil, dialog => {
  if (dialog && !dialog.open) dialog.showModal()
}, { flush: 'post' })

function progressChanged(event: Event) {
  if (!native.value) return
  const detail = (event as CustomEvent<{ message: string; cancellable: boolean }>).detail
  closeProgress.value = detail?.message ? detail : null
  event.preventDefault()
}

/* The title bar is visible from every page, so the live run belongs there: it answers
   "what is training right now" without navigating to the task list. */
const runningLabels = computed(() =>
  (store.tasks ?? [])
    .filter((task) => ACTIVE_TASK_STATES.includes(task.state))
    .map((task) => task.name || task.task_id)
    .join(' · '),
)
const windowTitle = computed(() => (runningLabels.value ? `${pageName.value} · ${runningLabels.value}` : pageName.value))

async function answerClose(action: CloseAction) {
  closePrompt.value = null
  try {
    await resolveClose(action)
  } catch (error) {
    toastError(error, '关闭窗口失败')
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
  window.addEventListener('lp-close-progress', progressChanged)
  void connect()
})
onBeforeUnmount(() => {
  disposed = true
  window.removeEventListener('pywebviewready', connect)
  window.removeEventListener('lp-window-state', stateChanged)
  window.removeEventListener('lp-close-request', closeRequested)
  window.removeEventListener('lp-close-progress', progressChanged)
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
          title="关闭窗口" :disabled="busy" @click="act('close')"
      >
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 4 8 8m0-8-8 8" /></svg>
      </button>

      <!-- The launcher decides and words the close dialog; the page only renders it. -->
      <ModalDialog v-if="closePrompt" :title="closePrompt.title" :width="460" @close="answerClose('cancel')">
        <p>{{ closePrompt.body }}</p>
        <template #footer>
          <button v-if="closePrompt.reason !== 'notice'" class="btn btn--ghost" @click="answerClose('cancel')">取消</button>
          <button
            class="btn"
            :class="['training', 'force'].includes(closePrompt.reason) ? 'btn--danger' : 'btn--primary'"
            @click="answerClose('confirm')"
          >{{ closePrompt.yes_label }}</button>
        </template>
      </ModalDialog>

    <Teleport to="body">
      <dialog v-if="closeProgress && !closePrompt" ref="closingVeil" class="closing-veil" aria-label="退出进度" @cancel.prevent>
        <span class="closing-spinner" aria-hidden="true" />
        <p class="closing-text" role="status">{{ closeProgress.message }}</p>
        <button v-if="closeProgress.cancellable" class="btn btn--ghost" title="保留窗口，已发出的停止请求仍然有效" @click="answerClose('cancel_wait')">保留窗口</button>
      </dialog>
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
.closing-veil[open] {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  max-width: none;
  max-height: none;
  box-sizing: border-box;
  margin: 0;
  padding: 24px;
  border: 0;
  background: transparent;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
}
.closing-veil::backdrop {
  background: rgb(0 0 0 / 38%);
  backdrop-filter: blur(6px);
}
.closing-spinner {
  width: 34px;
  height: 34px;
  border-radius: 50%;
  border: 3px solid rgb(255 255 255 / 28%);
  border-top-color: white;
  animation: closing-spin 900ms linear infinite;
}
.closing-text {
  margin: 0;
  color: white;
  font-size: 13px;
  letter-spacing: .02em;
  text-align: center;
}
@keyframes closing-spin {
  to { transform: rotate(360deg); }
}
@media (prefers-reduced-motion: reduce) {
  .closing-spinner { animation: none; }
}
</style>
