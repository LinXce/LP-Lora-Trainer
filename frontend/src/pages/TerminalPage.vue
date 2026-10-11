<script setup lang="ts">
import SelectInput from '@/components/SelectInput.vue'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import ModalDialog from '@/components/ModalDialog.vue'
import AppIcon from '@/components/AppIcon.vue'
import { api } from '@/api'
import { refreshEngines } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import { engineName } from '@/features/format'
import type { InstallationSession } from '@/types/api'
import logo from '@/assets/terminal-logo.txt?raw'

const route = useRoute()
const router = useRouter()
const sessions = ref<InstallationSession[]>([])
const selectedId = ref(typeof route.query.session === 'string' ? route.query.session : '')
const selected = computed(() => sessions.value.find((s) => s.session_id === selectedId.value))
const output = ref('')
const error = ref<string | null>(null)
const AUTO_SCROLL_KEY = 'lp.terminal.autoScroll'
function readAutoScroll(): boolean {
  try { return window.localStorage.getItem(AUTO_SCROLL_KEY) === '1' } catch { return false }
}
function persistAutoScroll(value: boolean): void {
  try { window.localStorage.setItem(AUTO_SCROLL_KEY, value ? '1' : '0') } catch { /* storage unavailable */ }
}
const autoScroll = ref(readAutoScroll())
const viewport = ref<HTMLElement | null>(null)
const stopping = ref(false)
const deleting = ref(false)
const deleteTarget = ref<InstallationSession | null>(null)
const deletedIds = new Set<string>()
const canDelete = computed(() => selected.value && ['succeeded', 'failed', 'cancelled', 'interrupted'].includes(selected.value.state))
const logoLines = logo.trimEnd().split('\n')
const colors = ['#f5f5f5', '#f5f5f5', '#22d3ee', '#22d3ee', '#2563eb', '#2563eb']
const states: Record<InstallationSession['state'], string> = {
  queued: '等待安装', running: '正在安装', stopping: '正在停止', verifying: '正在诊断',
  succeeded: '安装完成', failed: '安装失败', cancelled: '已停止', interrupted: '已中断',
}
const canStop = computed(() => selected.value && ['queued', 'running'].includes(selected.value.state))


const SPINNERS = {
  braille: [
    String.fromCodePoint(0x280b), String.fromCodePoint(0x2819), String.fromCodePoint(0x2839),
    String.fromCodePoint(0x2838), String.fromCodePoint(0x283c), String.fromCodePoint(0x2834),
    String.fromCodePoint(0x2826), String.fromCodePoint(0x2827), String.fromCodePoint(0x2807),
    String.fromCodePoint(0x280f),
  ],
  ascii: ['-', '\\', '|', '/'],
} as const

type DownloadProgress = {
  active: boolean
  packageName: string
  percent: number | null
  transferred: string | null
  total: string | null
}

const spinnerTick = ref(0)
const commandCount = computed(() => Math.max(selected.value?.commands.length ?? 0, 1))

function currentCommandLog(text: string): string {
  const parts = text.split(/\n>\s?/)
  return parts[parts.length - 1] ?? text
}

const currentCommandProgress = computed(() => {
  if (!selected.value || selected.value.state !== 'running') return 0
  const matches = [...currentCommandLog(output.value).matchAll(/(?:^|\s)(\d{1,3}(?:\.\d+)?)\s*%/g)]
  const value = Number(matches[matches.length - 1]?.[1] ?? 0)
  return Number.isFinite(value) ? Math.max(0, Math.min(100, value)) : 0
})

const progressPercent = computed(() => {
  const session = selected.value
  if (!session) return 0
  if (session.state === 'succeeded') return 100
  if (session.state === 'verifying') return 96
  if (session.state === 'queued') return 0

  const completed = Math.max(0, Math.min(commandCount.value, session.command_index ?? 0))
  const current = session.state === 'running' ? currentCommandProgress.value / 100 : 0
  return Math.round(Math.max(0, Math.min(1, (completed + current) / commandCount.value)) * 100)
})

const progressState = computed(() => selected.value?.state ?? 'queued')
const progressLabel = computed(() => {
  switch (selected.value?.state) {
    case 'queued': return '等待安装器启动'
    case 'running': return '正在下载并安装依赖'
    case 'verifying': return '正在验证引擎环境'
    case 'succeeded': return '安装完成'
    case 'failed': return '安装失败'
    case 'stopping': return '正在停止安装'
    case 'cancelled': return '安装已停止'
    case 'interrupted': return '安装已中断'
    default: return '等待安装任务'
  }
})

const spinnerFrame = computed(() => SPINNERS.braille[spinnerTick.value % SPINNERS.braille.length])

const downloadProgress = computed<DownloadProgress>(() => {
  if (!selected.value || selected.value.state !== 'running') {
    return { active: false, packageName: '', percent: null, transferred: null, total: null }
  }

  const log = currentCommandLog(output.value)
  const lines = log.split('\n').map((line) => line.trim()).filter(Boolean)
  const downloadLine = [...lines].reverse().find((line) => /\b(download(?:ing)?|fetch(?:ing)?|install(?:ing)?|resolv(?:ing)?|prepar(?:ing)?|build(?:ing)?)\b/i.test(line))
  if (!downloadLine) {
    return { active: false, packageName: '', percent: null, transferred: null, total: null }
  }

  const percentMatches = [...log.matchAll(/(\d{1,3}(?:\.\d+)?)\s*%/g)]
  const rawPercent = Number(percentMatches[percentMatches.length - 1]?.[1])
  const percent = Number.isFinite(rawPercent) ? Math.max(0, Math.min(100, rawPercent)) : null
  const sizePattern = /(\d+(?:\.\d+)?\s*(?:KiB|MiB|GiB|KB|MB|GB))\s*(?:\/|\bof\b)\s*(\d+(?:\.\d+)?\s*(?:KiB|MiB|GiB|KB|MB|GB))/i
  const sizeMatch = log.match(sizePattern)
  const totalSizeMatch = downloadLine.match(/\((\d+(?:\.\d+)?\s*(?:KiB|MiB|GiB|KB|MB|GB))\)/i)

  const packageMatch = downloadLine.match(/\b(?:download(?:ing)?|fetch(?:ing)?|install(?:ing)?|prepar(?:ing)?|build(?:ing)?)\s+(.+?)(?=\s+(?:from|at|for)\s+|\s*\(|\s+\d|$)/i)
  const packageName = (packageMatch?.[1] ?? 'current package').trim().replace(/[,:;]+$/, '')

  return {
    active: true,
    packageName,
    percent,
    transferred: sizeMatch?.[1] ?? null,
    total: sizeMatch?.[2] ?? totalSizeMatch?.[1] ?? null,
  }
})

const downloadSizeLabel = computed(() => {
  const { transferred, total } = downloadProgress.value
  if (!total) return ''
  return transferred ? `${transferred} / ${total}` : total
})

const downloadBarStyle = computed(() => {
  const percent = downloadProgress.value.percent
  return percent === null ? undefined : { width: `${percent}%` }
})

const progressDetail = computed(() => {
  const session = selected.value
  if (!session) return ''
  const count = commandCount.value
  const index = Math.max(0, Math.min(count - 1, session.command_index ?? 0))
  const command = session.commands[index]?.join(' ') ?? ''
  if (session.state === 'running') {
    const kind = /bootstrap_uv|ensure_engine_venv|venv/i.test(command)
      ? '准备独立 Python 环境'
      : /pip|uv|install|requirements/i.test(command)
        ? '下载并安装引擎依赖'
        : '执行引擎安装步骤'
    return `${kind} · 第 ${Math.min(index + 1, count)}/${count} 个阶段 · 当前阶段 ${currentCommandProgress.value}%`
  }
  if (session.state === 'verifying') return '检查已安装依赖与引擎入口'
  return `${count} 个安装阶段`
})
let cursor = 0
let timer: number | undefined
let spinnerTimer: number | undefined
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
    sessions.value = list.filter(s => !deletedIds.has(s.session_id))
    if (!sessions.value.some(s => s.session_id === selectedId.value)) selectedId.value = sessions.value[0]?.session_id ?? ''
    const id = selectedId.value
    if (sessions.value.some((s) => s.session_id === id)) {
      const start = cursor
      const chunk = await api.terminal.log(id, start)
      if (disposed) return
      // Ignore a response from a previously selected session or cleared view.
      if (!deletedIds.has(id) && id === selectedId.value && start === cursor) {
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
async function deleteSession() {
  const target = deleteTarget.value
  if (!target || deleting.value) return
  deleting.value = true
  try {
    await api.terminal.delete(target.session_id)
    deletedIds.add(target.session_id)
    sessions.value = sessions.value.filter(s => s.session_id !== target.session_id)
    if (selectedId.value === target.session_id) {
      selectedId.value = sessions.value[0]?.session_id ?? ''
      cursor = 0
      output.value = ''
      error.value = null
    }
    if (route.query.session === target.session_id) {
      await router.replace({ query: { ...route.query, session: selectedId.value || undefined } })
    }
    deleteTarget.value = null
    toast('已删除安装会话及磁盘日志，引擎环境保留', 'ok')
    void poll()
  } catch (err) { toastError(err, '删除终端日志') }
  finally { deleting.value = false }
}
watch(autoScroll, (enabled) => {
  persistAutoScroll(enabled)
  if (enabled) void scrollBottom()
})
onMounted(() => {
  spinnerTimer = window.setInterval(() => { spinnerTick.value += 1 }, 120)
  void poll()
})
onBeforeUnmount(() => {
  disposed = true
  window.clearTimeout(timer)
  window.clearInterval(spinnerTimer)
})
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
        <button class="btn btn--danger btn--sm" :disabled="!canDelete || deleting" title="删除已结束安装会话及完整日志，不删除引擎环境" @click="deleteTarget = selected ?? null">删除日志</button>
      </header>
      <div class="terminal-toolbar row">
        <SelectInput v-model="selectedId" class="input session-select" aria-label="安装会话">
          <option v-if="!sessions.length" value="">暂无安装会话</option>
          <option v-for="s in sessions" :key="s.session_id" :value="s.session_id">
            {{ s.label }} · {{ states[s.state] }} · {{ s.started_at.replace('T', ' ').slice(0, 19) }}
          </option>
        </SelectInput>
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
      <section v-if="selected" class="terminal-progress" :class="`terminal-progress--${progressState}`" aria-label="安装进度">
        <div class="terminal-progress__heading">
          <strong>{{ progressLabel }}</strong>
          <span class="terminal-progress__percent">{{ progressPercent }}%</span>
        </div>
        <div
          class="terminal-progress__bar"
          role="progressbar"
          :aria-valuenow="progressPercent"
          aria-valuemin="0"
          aria-valuemax="100"
          :aria-label="`安装进度 ${progressPercent}%`"
          :aria-busy="['queued', 'running', 'verifying', 'stopping'].includes(progressState)"
        >
          <span class="terminal-progress__fill" :style="{ width: `${progressPercent}%` }" />
        </div>
        <div class="terminal-progress__detail muted">
          <span>{{ progressDetail }}</span>
          <span v-if="selected.state === 'running'">{{ currentCommandProgress }}% · {{ Math.min((selected.command_index ?? 0) + 1, commandCount) }}/{{ commandCount }} 个阶段</span>
        </div>
      </section>
      <section
        v-if="downloadProgress.active && selected?.state === 'running'"
        class="terminal-download"
        aria-label="Download progress"
      >
        <div class="terminal-download__heading">
          <span class="terminal-download__spinner" aria-hidden="true">{{ spinnerFrame }}</span>
          <span class="terminal-download__package" :title="downloadProgress.packageName">{{ downloadProgress.packageName }}</span>
          <span v-if="downloadSizeLabel" class="terminal-download__size">
            {{ downloadSizeLabel }}
          </span>
          <span v-if="downloadProgress.percent !== null" class="terminal-download__percent">{{ downloadProgress.percent }}%</span>
        </div>
        <div class="terminal-download__bar" role="progressbar" :aria-valuenow="downloadProgress.percent ?? undefined" aria-valuemin="0" aria-valuemax="100" :aria-label="`Current package ${downloadProgress.packageName}`" :aria-busy="downloadProgress.percent === null">
          <span
            class="terminal-download__fill"
            :class="{ 'terminal-download__fill--indeterminate': downloadProgress.percent === null }"
            :style="downloadBarStyle"
          />
        </div>
      </section>
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
    <ModalDialog v-if="deleteTarget" title="删除终端历史日志？" @close="!deleting && (deleteTarget = null)">
      <p>删除 <strong>{{ deleteTarget.label }}</strong> 此次安装会话的历史记录和磁盘日志。</p>
      <p class="muted">不会卸载引擎、删除 Python 环境或已安装的依赖；此操作不可撤销。“清空显示”仍只清空界面内容。</p>
      <template #footer>
        <button class="btn btn--ghost" :disabled="deleting" @click="deleteTarget = null">取消</button>
        <button class="btn btn--danger" :disabled="deleting" @click="deleteSession">{{ deleting ? '正在删除…' : '确认删除' }}</button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
.terminal-panel { display: flex; flex-direction: column; min-height: 0; }
.page__header { padding-left: 44px; flex-wrap: wrap; }
.terminal-toolbar { padding: 12px 24px 12px 44px; gap: 12px; flex-wrap: wrap; }
.session-select { width: min(440px, 100%); }
.terminal-context { padding: 0 24px 12px 44px; overflow-wrap: anywhere; }
.terminal-progress { margin: 0 24px 16px 44px; padding: 14px 16px; border: 1px solid color-mix(in srgb, var(--border) 70%, transparent); border-radius: var(--radius-md); background: color-mix(in srgb, var(--panel) 78%, var(--code-bg)); }
.terminal-progress__heading, .terminal-progress__detail { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.terminal-progress__heading { color: var(--text-1); font-size: 13px; }
.terminal-progress__percent { color: var(--text-1); font-variant-numeric: tabular-nums; }
.terminal-progress__bar { height: 6px; margin: 10px 0 8px; overflow: hidden; border-radius: 999px; background: color-mix(in srgb, var(--text-3) 20%, transparent); }
.terminal-progress__fill { display: block; height: 100%; border-radius: inherit; background: #9bdcff; transition: width 220ms ease; }
.terminal-progress--running .terminal-progress__fill, .terminal-progress--queued .terminal-progress__fill, .terminal-progress--verifying .terminal-progress__fill { animation: terminal-progress-pulse 1.2s ease-in-out infinite; }
.terminal-progress__detail { font-size: 11px; line-height: 1.5; flex-wrap: wrap; }
.terminal-download { margin: -4px 24px 16px 44px; padding: 0 16px 12px; }
.terminal-download__heading { display: flex; align-items: center; gap: 8px; min-width: 0; color: var(--text-2); font-size: 11px; line-height: 1.4; }
.terminal-download__spinner { width: 1em; color: #55d69b; font-size: 14px; line-height: 1; }
.terminal-download__package { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.terminal-download__size, .terminal-download__percent { flex: none; color: var(--text-3); font-variant-numeric: tabular-nums; }
.terminal-download__percent { color: #55d69b; }
.terminal-download__bar { height: 2px; margin-top: 7px; overflow: hidden; border-radius: 999px; background: color-mix(in srgb, var(--text-3) 20%, transparent); }
.terminal-download__fill { display: block; width: 0; height: 100%; border-radius: inherit; background: #55d69b; transition: width 180ms ease; }
.terminal-download__fill--indeterminate { width: 35%; animation: terminal-download-indeterminate 1.1s ease-in-out infinite; }
@keyframes terminal-progress-pulse { 0%, 100% { opacity: .72; } 50% { opacity: 1; } }
@keyframes terminal-download-indeterminate { 0% { transform: translateX(-120%); } 100% { transform: translateX(320%); } }
@media (prefers-reduced-motion: reduce) { .terminal-progress__fill, .terminal-download__fill { transition: none; animation: none; } }
.terminal-output { flex: 1; min-height: 160px; overflow: auto; margin: 0 24px 0 44px; padding: 24px; border-radius: var(--radius-md); background: var(--code-bg); }
.terminal-logo { font-size: 12px; line-height: 1.4; margin: 0 0 24px; white-space: pre; }
.terminal-text { font-size: 12px; line-height: 1.6; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.terminal-error { color: var(--danger); padding: 8px 24px 8px 44px; white-space: pre-wrap; }
.terminal-output .terminal-error { padding-left: 0; }
.terminal-footer { padding: 12px 24px 16px 44px; font-size: 11px; }
</style>
