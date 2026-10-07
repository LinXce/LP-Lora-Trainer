<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import { api } from '@/api'
import { refreshEngines } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import { engineName } from '@/features/format'
import type { InstallationSession } from '@/types/api'
import logo from '@/assets/terminal-logo.txt?raw'

const route = useRoute()
const sessions = ref<InstallationSession[]>([])
const selectedId = ref(typeof route.query.session === 'string' ? route.query.session : '')
const selected = computed(() => sessions.value.find((s) => s.session_id === selectedId.value))
const output = ref('')
const error = ref<string | null>(null)
const autoScroll = ref(true)
const viewport = ref<HTMLElement | null>(null)
const stopping = ref(false)
const logoLines = logo.trimEnd().split('\n')
const colors = ['#f5f5f5', '#f5f5f5', '#22d3ee', '#22d3ee', '#2563eb', '#2563eb']
const states: Record<InstallationSession['state'], string> = {
  queued: '等待安装', running: '正在安装', stopping: '正在停止', verifying: '正在诊断',
  succeeded: '安装完成', failed: '安装失败', cancelled: '已停止', interrupted: '已中断',
}
const canStop = computed(() => selected.value && ['queued', 'running'].includes(selected.value.state))
let cursor = 0
let timer: number | undefined
let disposed = false
let polling = false
let pendingPoll = false

watch(() => route.query.session, (id) => {
  if (typeof id === 'string') selectedId.value = id
})
watch(selectedId, () => {
  cursor = 0
  output.value = ''
  void poll()
})

function plainText(text: string): string {
  // Treat all process output as text: no HTML and no terminal escape execution.
  return text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, '')
    .replace(/\x1b\][^\x07]*(?:\x07|\x1b\\)/g, '')
    .replace(/\r\n/g, '\n').replace(/\r/g, '\n')
    .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '')
}

async function scrollBottom() {
  if (!autoScroll.value) return
  await nextTick()
  if (viewport.value) viewport.value.scrollTop = viewport.value.scrollHeight
}

async function poll() {
  if (disposed) return
  if (polling) { pendingPoll = true; return }
  polling = true
  window.clearTimeout(timer)
  try {
    const before = selected.value?.state
    const list = await api.terminal.sessions()
    if (disposed) return
    sessions.value = list
    if (!selectedId.value && list[0]) selectedId.value = list[0].session_id
    const id = selectedId.value
    if (list.some((s) => s.session_id === id)) {
      const start = cursor
      const chunk = await api.terminal.log(id, start)
      if (disposed) return
      // Ignore a response from a previously selected session or cleared view.
      if (id === selectedId.value && start === cursor) {
        cursor = chunk.offset
        output.value = (output.value + plainText(chunk.text)).slice(-256 * 1024)
        await scrollBottom()
      }
    }
    error.value = null
    if (before && before !== selected.value?.state) void refreshEngines()
  } catch (err) {
    if (!disposed) error.value = err instanceof Error ? err.message : String(err)
  } finally {
    polling = false
    if (!disposed) {
      const active = sessions.value.some((s) => ['queued', 'running', 'stopping', 'verifying'].includes(s.state))
      timer = window.setTimeout(() => void poll(), pendingPoll ? 0 : document.hidden ? 15000 : active ? 1000 : 5000)
      pendingPoll = false
    }
  }
}

function clearDisplay() {
  output.value = '' // Backend log and current byte cursor are retained.
  toast('已清空当前显示，磁盘日志保留', 'ok')
}
async function copyLog() {
  try {
    await navigator.clipboard.writeText(logo.trimEnd() + '\n\n' + output.value)
    toast('已复制当前显示的日志', 'ok')
  } catch (err) { toastError(err, '复制日志') }
}
async function stopInstall() {
  if (!selected.value || stopping.value) return
  stopping.value = true
  try {
    await api.terminal.stop(selected.value.session_id)
    toast('已请求停止安装，部分依赖可能已安装', 'ok')
    await poll()
  } catch (err) { toastError(err, '停止安装') }
  finally { stopping.value = false }
}
watch(autoScroll, () => void scrollBottom())
onMounted(() => void poll())
onBeforeUnmount(() => { disposed = true; window.clearTimeout(timer) })
</script>

<template>
  <div class="page terminal-page">
    <section class="page__main panel terminal-panel">
      <header class="page__header">
        <div class="stack" style="gap: 4px">
          <h1 class="title-lg">终端</h1>
          <p class="muted">查看引擎安装输出；命令由引擎适配器提供，不执行任意系统命令。</p>
        </div>
        <span class="spacer" />
        <button class="btn btn--sm" @click="copyLog"><AppIcon name="copy" :size="14" />复制日志</button>
        <button class="btn btn--sm" @click="clearDisplay">清空显示</button>
      </header>
      <div class="terminal-toolbar row">
        <select v-model="selectedId" class="input session-select" aria-label="安装会话">
          <option v-if="!sessions.length" value="">暂无安装会话</option>
          <option v-for="s in sessions" :key="s.session_id" :value="s.session_id">
            {{ s.label }} · {{ states[s.state] }} · {{ s.started_at.replace('T', ' ').slice(0, 19) }}
          </option>
        </select>
        <span v-if="selected" class="label-pill">{{ states[selected.state] }}</span>
        <span class="spacer" />
        <label class="row" style="gap: 6px"><input v-model="autoScroll" type="checkbox" />自动滚动</label>
        <button class="btn btn--sm" :disabled="!canStop || stopping" @click="stopInstall">停止安装</button>
      </div>
      <div v-if="selected" class="terminal-context muted">
        <strong>{{ engineName(selected.engine_id) }}</strong> · {{ selected.title }}
        <div class="mono selectable">{{ selected.cwd }}</div>
        <div class="mono selectable">{{ selected.python_executable }}</div>
      </div>
      <p v-if="error" class="terminal-error">{{ error }}</p>
      <div ref="viewport" class="terminal-output selectable" role="log" aria-label="引擎安装输出">
        <pre class="terminal-logo"><span v-for="(line, index) in logoLines" :key="index" :style="{ color: colors[index] ?? 'var(--text-2)' }">{{ line }}{{ '\n' }}</span></pre>
        <p v-if="!selected" class="muted">LP LoRA Trainer · 终端已就绪。请在“引擎管理”中选择“安装引擎环境”。</p>
        <pre v-else class="terminal-text">{{ output || '等待安装输出…' }}</pre>
        <p v-if="selected?.error" class="terminal-error">{{ selected.error }}</p>
        <p v-if="selected?.diagnostic_error" class="terminal-error">环境诊断失败：{{ selected.diagnostic_error }}</p>
      </div>
      <footer class="terminal-footer muted">仅保留最近 256 KiB 显示内容，完整日志保存在数据目录 installations/&lt;会话 ID&gt;/output.log。</footer>
    </section>
  </div>
</template>

<style scoped>
.terminal-panel { display: flex; flex-direction: column; min-height: 0; }
.page__header { padding-left: 44px; flex-wrap: wrap; }
.terminal-toolbar { padding: 12px 24px 12px 44px; gap: 12px; flex-wrap: wrap; }
.session-select { width: min(440px, 100%); }
.terminal-context { padding: 0 24px 12px 44px; overflow-wrap: anywhere; }
.terminal-output { flex: 1; min-height: 160px; overflow: auto; margin: 0 24px 0 44px; padding: 24px; border-radius: var(--radius-md); background: var(--code-bg); }
.terminal-logo { font-size: 12px; line-height: 1.4; margin: 0 0 24px; white-space: pre; }
.terminal-text { font-size: 12px; line-height: 1.6; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.terminal-error { color: var(--danger); padding: 8px 24px 8px 44px; white-space: pre-wrap; }
.terminal-output .terminal-error { padding-left: 0; }
.terminal-footer { padding: 12px 24px 16px 44px; font-size: 11px; }
</style>
