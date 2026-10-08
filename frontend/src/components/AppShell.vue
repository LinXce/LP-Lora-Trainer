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

const active = computed(() => navItems.find((i) => i.name === route.name) ?? null)

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
  notchY.value = a.top + a.height / 2 - w.top
}

let ro: ResizeObserver | undefined
onMounted(() => {
  ro = new ResizeObserver(measure)
  if (shell.value) ro.observe(shell.value)
  void nextTick(measure)
})
onBeforeUnmount(() => ro?.disconnect())
watch(() => route.name, () => void nextTick(measure))

const runningCount = computed(() => store.tasks.filter((t) => ACTIVE_TASK_STATES.includes(t.state)).length)

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

const workspaceStyle = computed(() =>
  notchY.value === null ? {} : { '--notch-y': `${notchY.value}px` },
)
</script>

<template>
  <div ref="shell" class="shell">
    <aside class="rail">
      <RouterLink to="/" class="rail__brand" title="LP LoRA Trainer">
        <span class="rail__logo" aria-hidden="true">LP</span>
      </RouterLink>

      <nav class="rail__pill" aria-label="主导航">
        <div class="rail__group">
          <RouterLink
            v-for="item in topItems"
            :key="item.name"
            :ref="(el) => setItemRef(item.name, el)"
            :to="{ name: item.name }"
            class="rail__item"
            :class="{ 'is-active': active?.name === item.name }"
            :aria-label="item.title"
          >
            <AppIcon :name="item.icon" :size="15" />
            <span v-if="item.name === 'tasks' && runningCount" class="rail__badge num">{{ runningCount }}</span>
            <span class="rail__tip">{{ item.title }}</span>
          </RouterLink>
        </div>
        <div class="rail__group">
          <RouterLink
            v-for="item in bottomItems"
            :key="item.name"
            :ref="(el) => setItemRef(item.name, el)"
            :to="{ name: item.name }"
            class="rail__item"
            :class="{ 'is-active': active?.name === item.name }"
            :aria-label="item.title"
          >
            <AppIcon :name="item.icon" :size="15" />
            <span class="rail__tip">{{ item.title }}</span>
          </RouterLink>
          <div class="rail__status" :class="`is-${connection.tone}`" :title="connection.label" role="status" :aria-label="connection.label">
            <span class="rail__status-core" />
          </div>
        </div>
      </nav>
    </aside>

    <!-- The notch: active page marker sitting in the workspace cut-out -->
    <div
      v-if="active && notchY !== null"
      class="notch"
      :style="{ top: `calc(${notchY}px + var(--frame-pad))` }"
      aria-hidden="true"
    >
      <AppIcon :name="active.icon" :size="17" />
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
      <div class="workspace__view">
        <slot />
      </div>
    </main>
  </div>
</template>

<style scoped>
.shell {
  --frame-pad: 12px;
  --notch-r: 24px;
  --notch-cut: 32px;

  position: relative;
  display: grid;
  grid-template-columns: var(--rail-w) minmax(0, 1fr);
  gap: calc(var(--gap) + 4px);
  height: 100%;
  padding: var(--frame-pad);
  background: var(--frame);
  border: 1px solid var(--frame-edge);
  overflow: hidden;
}

/* ---------- Rail ---------- */

.rail {
  display: flex;
  flex-direction: column;
  gap: var(--gap);
  min-height: 0;
}
.rail__brand {
  display: grid;
  place-items: center;
  width: var(--rail-w);
  height: var(--rail-w);
  flex: none;
  border-radius: 50%;
  background: var(--panel-strong);
  color: var(--light-strong);
  transition: transform 0.3s var(--ease), background 0.2s;
}
.rail__logo {
  display: inline-block;
  color: var(--light-strong);
  font-size: 15px;
  font-weight: 800;
  line-height: 1;
  letter-spacing: 0.08em;
}
.rail__brand:hover {
  background: #626262;
  transform: rotate(45deg);
}
.rail__pill {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 0;
  padding: 26px 0 6px;
  border-radius: var(--radius-pill);
  background: var(--panel-strong);
}
.rail__group {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 18px;
}
.rail__item {
  position: relative;
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 10px;
  color: var(--light);
  text-decoration: none;
  transition: background 0.15s, color 0.15s;
}
.rail__item::before {
  /* the small rounded squares of the reference, visible behind the icon */
  content: '';
  position: absolute;
  inset: 6px;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.06);
  transition: opacity 0.15s;
}
.rail__item:hover {
  background: rgba(255, 255, 255, 0.08);
  color: var(--light-strong);
}
.rail__item.is-active {
  color: transparent;
}
/* The active item is represented by the workspace notch; hovering must not add a second highlight. */
.rail__item.is-active:hover {
  background: transparent;
  color: transparent;
}
.rail__item.is-active::before {
  opacity: 0;
}
.rail__badge {
  position: absolute;
  top: -2px;
  right: -4px;
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
.rail__item.is-active .rail__tip {
  display: none;
}

/* The big circle at the bottom of the rail doubles as the backend connection indicator. */
.rail__status {
  display: grid;
  place-items: center;
  width: 48px;
  height: 48px;
  margin-top: 4px;
  border-radius: 50%;
  background: var(--light);
}
.rail__status-core {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--ink-soft);
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
  left: calc(var(--frame-pad) + var(--rail-w) + var(--gap) + 4px);
  width: calc(var(--notch-r) * 2);
  height: calc(var(--notch-r) * 2);
  transform: translate(-50%, -50%);
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--light);
  color: var(--ink);
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
  transition: top 0.38s var(--ease);
  z-index: 5;
}

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
    circle at 0 var(--notch-y),
    transparent calc(var(--notch-cut) - 0.5px),
    #000 var(--notch-cut)
  );
  mask-image: radial-gradient(
    circle at 0 var(--notch-y),
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
</style>
