<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
/* Brand icon lives inside the Vite root so dev and build resolve identically. */
import logoUrl from '@/assets/logo.png'
import {
  beginWindowResize, getWindowState, hasWindowControls, windowAction,
  type NativeWindowState, type ResizeEdge, type WindowAction,
} from '@/api/bridge'
import { toastError } from '@/features/toast'

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
  void connect()
})
onBeforeUnmount(() => {
  disposed = true
  window.removeEventListener('pywebviewready', connect)
  window.removeEventListener('lp-window-state', stateChanged)
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
      <strong class="window-bar__name">{{ pageName }}</strong>
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
        title="关闭窗口（训练继续运行）" :disabled="busy" @click="act('close')"
      >
        <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 4 8 8m0-8-8 8" /></svg>
      </button>
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
</style>
