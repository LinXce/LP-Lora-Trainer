<script setup lang="ts">
/**
 * New training: fixed base form + capability-driven parameters + native
 * config preview. All validation is server-side; the preview is whatever the
 * adapter produced, never assembled by the frontend.
 */
import { computed, onBeforeUnmount, reactive, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import PathInput from '@/components/PathInput.vue'
import SegTabs from '@/components/SegTabs.vue'
import StatusTag from '@/components/StatusTag.vue'
import EmptyState from '@/components/EmptyState.vue'
import ModalDialog from '@/components/ModalDialog.vue'
import { api } from '@/api'
import { refreshTasks, store } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import { clearDraft, emptyDraft, hasContent, loadDraft, saveDraft, type DraftParams } from '@/features/draftStore'
import { engineName, installStateMeta, shortCommit, verificationMeta } from '@/features/format'
import type { BaseModel, Dataset, EngineCapabilities, ParamGroup, ParamSpec, TrainingDraft, ValidationResult } from '@/types/api'

const router = useRouter()

const importInput = ref<HTMLInputElement | null>(null)

/* The staged form is shared through the backend, so the desktop window and a
   browser tab continue from the same content. */
const draft = reactive<TrainingDraft>(emptyDraft())
let paramsByArchitecture: Record<string, DraftParams> = {}
const restoredForm = ref(false)
const resetOpen = ref(false)
/* Saving stays disabled until the shared draft has been read, so an empty form
   can never overwrite what another client staged. */
let stagingReady = false
let saveFailed = false

async function restoreStagedForm() {
  try {
    const staged = await loadDraft()
    stagingReady = true
    if (staged) {
      // Memory must be in place before applying the draft: loading capabilities
      // rebuilds the parameter set and reads it back from here.
      paramsByArchitecture = { ...staged.paramsByArchitecture }
      Object.assign(draft, { ...emptyDraft(), ...staged.draft, params: { ...staged.draft.params } })
      restoredForm.value = hasContent(staged.draft)
    }
  } catch (err) {
    // An older backend has no draft endpoint (404): report once and keep the
    // form usable instead of retrying a failing save on every keystroke.
    toastError(err, '读取暂存表单')
  }
}
void restoreStagedForm()

/* Base-model type first: it decides which engines are selectable. */
const baseModels = ref<BaseModel[]>([])
async function loadBaseModels() {
  try {
    baseModels.value = await api.baseModels.list()
  } catch {
    baseModels.value = []
  }
}
watch(() => store.engines, () => void loadBaseModels(), { immediate: true })
const baseModel = computed(() => baseModels.value.find((m) => m.id === draft.architecture) ?? null)
watch(
  baseModels,
  (list) => {
    if (!list.some((m) => m.id === draft.architecture)) draft.architecture = list[0]?.id ?? ''
  },
  { immediate: true },
)

/* Only engines that support the selected base-model type are offered. */
const installations = computed(() => {
  const model = baseModel.value
  return [...store.engines]
    .filter((e) => !model || (e.engine_id !== null && model.engines.includes(e.engine_id)))
    .sort((a, b) => Number(b.is_default) - Number(a.is_default))
})
watch(
  installations,
  (list) => {
    if (list.some((e) => e.installation_id === draft.installation_id)) return
    const pick = list.find((e) => e.is_default && e.state === 'ready') ?? list.find((e) => e.state === 'ready') ?? list[0]
    draft.installation_id = pick?.installation_id ?? ''
  },
  { immediate: true },
)
const installation = computed(() => store.engines.find((e) => e.installation_id === draft.installation_id) ?? null)

const datasets = ref<Dataset[]>([])
api.datasets
  .list()
  .then((d) => (datasets.value = d))
  .catch(() => (datasets.value = []))

const result = ref<ValidationResult | null>(null)
const validating = ref(false)
let seq = 0
onBeforeUnmount(() => { seq += 1; flushStagedDraft() })

/* Capabilities */
const caps = ref<EngineCapabilities | null>(null)
const capsError = ref<string | null>(null)
watch(
  () => draft.installation_id,
  async (id) => {
    caps.value = null
    capsError.value = null
    result.value = null
    seq += 1
    if (!id) return
    try {
      const c = await api.engines.capabilities(id)
      if (draft.installation_id !== id) return
      // Prefer what this base-model type used before; otherwise carry over values
      // that still exist, and fall back to the engine default.
      const prev = { ...draft.params }
      const remembered = paramsByArchitecture[draft.architecture] ?? {}
      const next: TrainingDraft['params'] = {}
      for (const p of c.params) {
        if (p.key in remembered) next[p.key] = remembered[p.key]
        else if (p.key in prev) next[p.key] = prev[p.key]
        else next[p.key] = p.default
      }
      const dropped = Object.keys(prev).filter((k) => !c.params.some((p) => p.key === k))
      if (dropped.length) {
        const shown = dropped.slice(0, 6).join('、')
        toast(`已切换引擎参数集，移除不适用参数：${shown}${dropped.length > 6 ? ` 等 ${dropped.length} 项` : ''}`, 'info', 6000)
      }
      draft.params = next
      caps.value = c
    } catch (err) {
      capsError.value = err instanceof Error ? err.message : String(err)
    }
  },
  { immediate: true },
)

type Tab = ParamGroup
const tab = ref<Tab>('basic')
/* Only parameters that apply to the selected base-model type are rendered. */
const applicableParams = computed(() =>
  (caps.value?.params ?? []).filter((p) => !p.architectures?.length || p.architectures.includes(draft.architecture)),
)
const sections = computed(() => {
  const out = new Map<string, ParamSpec[]>()
  for (const p of applicableParams.value) {
    if (p.group !== tab.value) continue
    out.set(p.section, [...(out.get(p.section) ?? []), p])
  }
  return [...out.entries()]
})
const groupCounts = computed(() => {
  const c = { basic: 0, advanced: 0, native: 0 }
  for (const p of applicableParams.value) c[p.group]++
  return c
})

function setParam(p: ParamSpec, raw: string | boolean) {
  if (p.type === 'bool') draft.params[p.key] = raw as boolean
  else if (p.type === 'int' || p.type === 'float') draft.params[p.key] = raw === '' ? null : Number(raw)
  else draft.params[p.key] = raw as string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function hasOwn(value: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key)
}

/** Remove a TOML comment without treating # inside a quoted string as a comment. */
function stripTomlComment(line: string): string {
  let quote: '"' | "'" | null = null
  let escaped = false
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i]
    if (quote === '"') {
      if (escaped) {
        escaped = false
      } else if (char === '\\') {
        escaped = true
      } else if (char === '"') {
        quote = null
      }
      continue
    }
    if (quote === "'") {
      if (char === "'") {
        // TOML literal strings escape a quote by doubling it.
        if (line[i + 1] === "'") i += 1
        else quote = null
      }
      continue
    }
    if (char === '"' || char === "'") quote = char
    else if (char === '#') return line.slice(0, i)
  }
  return line
}

function tomlValueComplete(value: string): boolean {
  let quote: '"' | "'" | null = null
  let escaped = false
  let depth = 0
  for (let i = 0; i < value.length; i += 1) {
    const char = value[i]
    if (quote === '"') {
      if (escaped) escaped = false
      else if (char === '\\') escaped = true
      else if (char === '"') quote = null
      continue
    }
    if (quote === "'") {
      if (char === "'") {
        if (value[i + 1] === "'") i += 1
        else quote = null
      }
      continue
    }
    if (char === '"' || char === "'") quote = char
    else if (char === '[') depth += 1
    else if (char === ']') depth -= 1
  }
  return quote === null && depth === 0
}

function splitTomlArray(value: string): string[] {
  const parts: string[] = []
  let start = 0
  let depth = 0
  let quote: '"' | "'" | null = null
  let escaped = false
  for (let i = 0; i < value.length; i += 1) {
    const char = value[i]
    if (quote === '"') {
      if (escaped) escaped = false
      else if (char === '\\') escaped = true
      else if (char === '"') quote = null
      continue
    }
    if (quote === "'") {
      if (char === "'") {
        if (value[i + 1] === "'") i += 1
        else quote = null
      }
      continue
    }
    if (char === '"' || char === "'") quote = char
    else if (char === '[') depth += 1
    else if (char === ']') depth -= 1
    else if (char === ',' && depth === 0) {
      parts.push(value.slice(start, i).trim())
      start = i + 1
    }
  }
  const last = value.slice(start).trim()
  if (last) parts.push(last)
  return parts
}

function parseTomlValue(raw: string): unknown {
  const value = raw.trim()
  if (!value) return null
  if (value.startsWith('[') && value.endsWith(']')) {
    return splitTomlArray(value.slice(1, -1)).map(parseTomlValue)
  }
  if (value.startsWith('"') && value.endsWith('"')) {
    try {
      return JSON.parse(value)
    } catch {
      return value.slice(1, -1)
    }
  }
  if (value.startsWith("'") && value.endsWith("'")) return value.slice(1, -1).replace(/''/g, "'")
  if (value === 'true' || value === 'false') return value === 'true'
  const number = value.replace(/_/g, '')
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(number)) {
    const parsed = Number(number)
    if (Number.isFinite(parsed)) return parsed
  }
  // This keeps dates and other valid TOML scalar forms importable as text.
  return value
}

/**
 * Parse the flat TOML files emitted by Musubi Tuner. It intentionally stays
 * dependency-free so importing a config works in the packaged desktop build.
 * Basic arrays, comments, sections, booleans and numeric values are supported.
 */
function parseToml(text: string): Record<string, unknown> {
  const result: Record<string, unknown> = {}
  let section = ''
  let pending = ''

  for (const physicalLine of text.replace(/^\uFEFF/, '').split(/\r?\n/)) {
    const line = stripTomlComment(physicalLine).trim()
    if (!line) continue
    if (!pending && line.startsWith('[') && line.endsWith(']')) {
      section = line.slice(1, -1).trim()
      continue
    }
    pending = pending ? `${pending} ${line}` : line
    const equals = pending.indexOf('=')
    if (equals < 1) continue
    const rawValue = pending.slice(equals + 1).trim()
    if (!tomlValueComplete(rawValue)) continue
    const key = pending.slice(0, equals).trim()
    if (key) result[section ? `${section}.${key}` : key] = parseTomlValue(rawValue)
    pending = ''
  }

  if (pending) throw new Error('TOML ???????????')
  return result
}

function importedParams(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) return {}
  const source = isRecord(value.params) ? value.params : isRecord(value.parameters) ? value.parameters : value
  // TOML sections are represented by the lightweight parser as dotted keys
  // (for example `general.batch_size`). Keep the original keys, and expose
  // their leaf names as fallbacks so both flat Musubi files and the generated
  // [general] dataset config can be imported.
  const flattened = { ...source }
  for (const [key, item] of Object.entries(source)) {
    const separator = key.lastIndexOf('.')
    if (separator >= 0) {
      const leaf = key.slice(separator + 1)
      if (leaf && !hasOwn(flattened, leaf)) flattened[leaf] = item
    }
  }
  return flattened
}

function convertImportedValue(p: ParamSpec, value: unknown): string | number | boolean | null | undefined {
  if (value === null) return null
  if (p.type === 'bool') {
    if (typeof value === 'boolean') return value
    if (typeof value === 'number' && (value === 0 || value === 1)) return value === 1
    if (typeof value === 'string') {
      const normalized = value.trim().toLowerCase()
      if (['true', 'yes', 'on', '1'].includes(normalized)) return true
      if (['false', 'no', 'off', '0'].includes(normalized)) return false
    }
    return undefined
  }
  if (p.type === 'int' || p.type === 'float') {
    const number = typeof value === 'number' ? value : typeof value === 'string' && value.trim() ? Number(value.trim()) : NaN
    if (!Number.isFinite(number) || (p.type === 'int' && !Number.isInteger(number))) return undefined
    return number
  }
  if (Array.isArray(value)) return value.map((item) => String(item)).join('\n')
  if (typeof value !== 'string' && typeof value !== 'number' && typeof value !== 'boolean') return undefined
  const text = String(value)
  if (p.type === 'enum' && p.options?.length && !p.options.some((option) => option.value === text)) return undefined
  return text
}

function importedArchitecture(value: unknown): string | null {
  if (typeof value !== 'string') return null
  const normalized = value.trim().toLowerCase()
  const direct = baseModels.value.find((model) => model.id.toLowerCase() === normalized)
  if (direct) return direct.id
  const suffixed = baseModels.value.find((model) =>
    [`${model.id.toLowerCase()}-lora`, `${model.id.toLowerCase()}_lora`].includes(normalized),
  )
  return suffixed?.id ?? null
}

function normalizedPath(value: string): string {
  return value.trim().replaceAll('\\', '/').replace(/\/+$/, '').toLowerCase()
}

function valueFromKeys(source: Record<string, unknown>, keys: string[]): string | null {
  for (const key of keys) {
    if (typeof source[key] === 'string' && source[key].trim()) return source[key] as string
  }
  return null
}

function importResolution(value: unknown): number {
  let values: number[]
  if (Array.isArray(value)) {
    // Musubi accepts either [width, height] or a list of bucket pairs.
    const pair = Array.isArray(value[0]) ? value[0] : value
    values = pair.slice(0, 2).map((item) => typeof item === 'number' ? item : Number(item))
  } else if (typeof value === 'string' || typeof value === 'number') {
    values = String(value).split(/[xX,\s]+/).filter(Boolean).map(Number)
  } else {
    return 0
  }
  if (values.length !== 2 || values.some((item) => !Number.isInteger(item) || item <= 0)) return 0
  const width = caps.value?.params.find((p) => p.key === 'resolution_width')
  const height = caps.value?.params.find((p) => p.key === 'resolution_height')
  if (width) draft.params[width.key] = values[0]
  if (height) draft.params[height.key] = values[1]
  return width && height ? 2 : 0
}

function openImport() {
  if (!caps.value) return
  importInput.value?.click()
}

async function importParams(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file || !caps.value) return

  try {
    const isToml = file.name.toLowerCase().endsWith('.toml')
    const parsed: unknown = isToml ? parseToml(await file.text()) : JSON.parse(await file.text())
    if (!isRecord(parsed)) throw new Error(isToml ? 'TOML ??????????' : 'JSON ????????')

    const params = importedParams(parsed)
    const importedArch = importedArchitecture(parsed.architecture ?? parsed.model_train_type)
    if (importedArch && caps.value.architectures.includes(importedArch)) draft.architecture = importedArch

    const baseModelPath = valueFromKeys(parsed, ['base_model_path', 'dit', 'pretrained_model_name_or_path'])
    const name = valueFromKeys(parsed, ['name', 'output_name'])
    const outputDir = valueFromKeys(parsed, ['output_dir'])
    if (name !== null) draft.name = name
    if (baseModelPath !== null) draft.base_model_path = baseModelPath
    if (outputDir !== null) draft.output_dir = outputDir

    const datasetPath = valueFromKeys(params, ['train_data_dir', 'dataset_dir', 'image_directory'])
    if (datasetPath !== null) {
      const matched = datasets.value.find((dataset) => normalizedPath(dataset.path) === normalizedPath(datasetPath))
      if (matched) draft.dataset_id = matched.dataset_id
    }

    // Names used by Musubi's standalone TOML differ from the UI for a few
    // shared values. Keep these aliases local to importing so the adapter's
    // existing native output contract stays unchanged.
    const importAliases: Record<string, string[]> = {
      batch_size: ['train_batch_size'],
      train_batch_size: ['batch_size'],
    }

    let importedCount = 0
    for (const spec of caps.value.params) {
      if (spec.architectures?.length && !spec.architectures.includes(draft.architecture)) continue
      const candidates = [
        spec.native_key,
        spec.key,
        ...(importAliases[spec.key] ?? []),
        ...(spec.native_key ? importAliases[spec.native_key] ?? [] : []),
      ].filter((key, index, all): key is string => Boolean(key) && all.indexOf(key) === index)
      const importedKey = candidates.find((key) => hasOwn(params, key))
      if (!importedKey) continue
      const value = convertImportedValue(spec, params[importedKey])
      if (value === undefined) continue
      draft.params[spec.key] = value
      importedCount += 1
    }
    importedCount += importResolution(params.resolution)

    seq += 1
    result.value = null
    stale.value = false
    toast(`??? ${importedCount} ?????????`, 'ok')
  } catch (err) {
    toast(`???????${err instanceof Error ? err.message : String(err)}`, 'danger')
  }
}

/* Validation is explicit: it runs when the user presses 检测参数 or submits, never
   on every keystroke. Typing only marks the previous result as outdated, and
   stages the form so leaving the page does not clear it. */
const stale = ref(false)
let saveTimer: number | undefined

function persistStagedDraft() {
  window.clearTimeout(saveTimer)
  saveTimer = undefined
  if (!stagingReady) return
  if (draft.architecture) paramsByArchitecture[draft.architecture] = { ...draft.params }
  void saveDraft(JSON.parse(JSON.stringify(draft)) as TrainingDraft, paramsByArchitecture)
    .then(() => { saveFailed = false })
    .catch((err) => {
      // One notice per failure streak: a draft is convenience data, not a task.
      if (saveFailed) return
      saveFailed = true
      toastError(err, '暂存表单')
    })
}

/** Save immediately when the user navigates away inside the debounce window. */
function flushStagedDraft() {
  if (saveTimer !== undefined) persistStagedDraft()
}

watch(
  draft,
  () => {
    if (result.value) stale.value = true
    if (!stagingReady) return
    window.clearTimeout(saveTimer)
    saveTimer = window.setTimeout(persistStagedDraft, 500)
  },
  { deep: true },
)

async function resetStagedForm() {
  resetOpen.value = false
  window.clearTimeout(saveTimer)
  saveTimer = undefined
  try {
    await clearDraft()
  } catch (err) {
    toastError(err, '清空暂存')
    return
  }
  paramsByArchitecture = {}
  // Keep the page usable without re-staging the cleared content.
  stagingReady = false
  Object.assign(draft, emptyDraft())
  draft.architecture = baseModels.value[0]?.id ?? ''
  result.value = null
  stale.value = false
  restoredForm.value = false
  stagingReady = true
  toast('已清空暂存的表单内容', 'ok')
}

async function runCheck(): Promise<ValidationResult | null> {
  if (!caps.value) { result.value = null; validating.value = false; return null }
  const my = ++seq
  validating.value = true
  try {
    const r = await api.training.validate(payload())
    if (my !== seq) return null
    result.value = r
    stale.value = false
    return r
  } catch (err) {
    if (my !== seq) return null
    result.value = { ok: false, issues: [{ field: null, level: 'error', message: (err as Error).message }], native_config: null, native_format: null, argv: null, pipeline: null, submittable: false }
    stale.value = false
    return null
  } finally {
    if (my === seq) validating.value = false
  }
}

/* List arguments are multi-line in the UI but space separated for the engines;
   multi-resolution sets keep their lines (each line is one bucket resolution). */
function payload(): TrainingDraft {
  const copy = JSON.parse(JSON.stringify(draft)) as TrainingDraft
  for (const p of caps.value?.params ?? []) {
    if (!p.list_arg) continue
    const value = copy.params[p.key]
    if (typeof value === 'string') copy.params[p.key] = value.split(/\s+/).filter(Boolean).join(' ')
  }
  return copy
}

const fieldIssue = (key: string) => result.value?.issues.find((i) => i.field === key) ?? null
const errors = computed(() => result.value?.issues.filter((i) => i.level === 'error') ?? [])
const warnings = computed(() => result.value?.issues.filter((i) => i.level === 'warning') ?? [])

/* Submitting is allowed whenever the engine itself can run; parameter errors are
   reported by the check that submit performs first. */
const canSubmit = computed(
  () =>
    !!caps.value &&
    caps.value.submittable !== false &&
    !validating.value &&
    !submitting.value &&
    installation.value?.state === 'ready',
)
/* Some engines only describe parameters/config; they cannot queue a job yet. */
const previewOnly = computed(() => caps.value?.submittable === false)

const submitting = ref(false)
async function submit() {
  if (!canSubmit.value || submitting.value) return
  const checked = await runCheck()
  if (!checked || !checked.ok) {
    toast('参数检测未通过，请修正右侧列出的错误后再提交', 'danger', 6000)
    return
  }
  if (checked.submittable === false) {
    toast('该引擎当前仅支持配置预览，尚不能提交训练', 'danger')
    return
  }
  submitting.value = true
  try {
    const task = await api.training.submit(payload())
    toast(`已提交任务 ${task.name}，安装实例已固定`, 'ok')
    await refreshTasks()
    void router.push({ name: 'tasks', params: { taskId: task.task_id } })
  } catch (err) {
    toastError(err, '提交任务')
  } finally {
    submitting.value = false
  }
}

const previewMode = ref<'config' | 'argv'>('config')

function formatCommand(cmd: string[]) {
  return cmd.map((a) => (a.includes(' ') ? JSON.stringify(a) : a)).join('\n  ')
}
/* The supervisor may run caching stages before training; show them in order. */
const argvPreview = computed(() => {
  const stages = result.value?.pipeline
  if (stages?.length) {
    const total = stages.length
    return stages
      .map((cmd, i) => `# 阶段 ${i + 1}/${total}${i === total - 1 ? '（训练）' : '（缓存）'}\n${formatCommand(cmd)}`)
      .join('\n\n')
  }
  return result.value?.argv ? formatCommand(result.value.argv) : ''
})

async function copyPreview() {
  const text = previewMode.value === 'config' ? result.value?.native_config : argvPreview.value
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
    toast('已复制', 'ok', 1800)
  } catch {
    toast('复制失败：剪贴板不可用', 'danger')
  }
}
</script>

<template>
  <div class="page">
    <section class="page__main panel">
      <header class="page__header">
        <h1 class="title-lg">新建训练</h1>
        <span class="spacer" />
        <button class="btn btn--ghost btn--sm" title="清除本机暂存的表单内容" @click="resetOpen = true">清空暂存</button>
        <StatusTag
          v-if="result"
          :tone="validating ? 'info' : stale ? 'warn' : result.ok ? 'ok' : 'danger'"
          :label="validating ? '检测中…' : stale ? '参数已改动，需重新检测' : result.ok ? '配置有效' : `${errors.length} 个错误`"
        />
        <StatusTag v-else :tone="validating ? 'info' : 'neutral'" :label="validating ? '检测中…' : '尚未检测参数'" />
      </header>

      <div class="page__body form">
        <p v-if="restoredForm" class="restored">
          <AppIcon name="refresh" :size="13" />
          <span>已恢复上次填写的内容（含该底模类型的参数）。如需从空白开始，点右上角「清空暂存」。</span>
          <span class="spacer" />
          <button class="btn btn--ghost btn--sm" @click="restoredForm = false">知道了</button>
        </p>
        <!-- Base form -->
        <div class="block">
          <span class="label-pill">基础信息</span>
          <div class="grid2">
            <label class="field">
              <span class="field__label">任务名称 <span class="field__hint">同时作为输出文件名</span></span>
              <input v-model="draft.name" class="input" :class="{ 'is-invalid': fieldIssue('name')?.level === 'error' }" placeholder="例如 character_aki_v3" spellcheck="false" />
              <span v-if="fieldIssue('name')" class="field__error">{{ fieldIssue('name')!.message }}</span>
            </label>

            <label class="field">
              <span class="field__label">底模类型 <span class="field__hint">决定可选的引擎与参数</span></span>
              <div class="row" style="flex-wrap: wrap">
                <button
                  v-for="m in baseModels"
                  :key="m.id"
                  type="button"
                  class="chip"
                  :class="{ 'is-on': draft.architecture === m.id }"
                  @click="draft.architecture = m.id"
                >
                  {{ m.label }}
                </button>
                <span v-if="!baseModels.length" class="muted">暂无已登记引擎支持的底模类型</span>
              </div>
            </label>

            <label class="field">
              <span class="field__label">引擎安装实例</span>
              <select v-model="draft.installation_id" class="input">
                <option value="" disabled>选择安装实例</option>
                <option v-for="e in installations" :key="e.installation_id" :value="e.installation_id" :disabled="e.state !== 'ready'">
                  {{ engineName(e.engine_id) }} / {{ e.label }}{{ e.is_default ? '（默认）' : '' }}{{ e.state !== 'ready' ? ' — 环境未就绪' : '' }}
                </option>
              </select>
              <span v-if="capsError" class="field__error">{{ capsError }}</span>
            </label>

            <label class="field">
              <span class="field__label">数据集</span>
              <select v-model="draft.dataset_id" class="input" :class="{ 'is-invalid': fieldIssue('dataset_id')?.level === 'error' }">
                <option value="" disabled>选择数据集</option>
                <option v-for="d in datasets" :key="d.dataset_id" :value="d.dataset_id">
                  {{ d.name }}{{ d.image_count !== null ? `（${d.image_count} 张）` : '（未扫描）' }}
                </option>
              </select>
            </label>

            <label class="field span2">
              <span class="field__label">底模文件</span>
              <PathInput
                v-model="draft.base_model_path"
                kind="file"
                dialog-title="选择底模"
                :file-types="['模型文件 (*.safetensors;*.ckpt)']"
                :invalid="fieldIssue('base_model_path')?.level === 'error'"
              />
              <span v-if="fieldIssue('base_model_path')" class="field__error">{{ fieldIssue('base_model_path')!.message }}</span>
            </label>

            <label class="field span2">
              <span class="field__label">输出目录</span>
              <PathInput v-model="draft.output_dir" kind="directory" dialog-title="选择输出目录" :invalid="fieldIssue('output_dir')?.level === 'error'" />
              <span v-if="fieldIssue('output_dir')" class="field__error">{{ fieldIssue('output_dir')!.message }}</span>
            </label>
          </div>
        </div>

        <!-- Capability-driven params -->
        <div v-if="caps" class="block">
          <div class="row">
            <span class="label-pill">训练参数</span>
            <span class="spacer" />
            <input
              ref="importInput"
              type="file"
              accept=".json,.toml,application/json,application/toml"
              hidden
              @change="importParams"
            />
            <button
              class="btn btn--ghost btn--sm"
              type="button"
              :disabled="!caps"
              title="导入项目参数 JSON 或扁平引擎参数 JSON"
              @click="openImport"
            >
              <AppIcon name="upload" :size="13" />
              导入参数
            </button>
            <SegTabs
              v-model="tab"
              :options="[
                { value: 'basic', label: '基础', count: groupCounts.basic },
                { value: 'advanced', label: '高级', count: groupCounts.advanced },
                { value: 'native', label: `原生 · ${engineName(installation?.engine_id ?? null)}`, count: groupCounts.native },
              ]"
            />
          </div>
          <p v-if="tab === 'native'" class="field__hint">
            引擎专属参数，不与其他引擎的同名参数视为等价。
          </p>

          <div v-for="[section, params] in sections" :key="section" class="section">
            <span class="eyebrow">{{ section }}</span>
            <div class="grid3">
              <div
                v-for="p in params"
                :key="p.key"
                class="field"
                :class="{ span3: p.type === 'string', 'is-off': p.unsupported_reason }"
                :title="p.unsupported_reason ?? p.help ?? ''"
              >
                <span class="field__label">
                  <span>{{ p.label }}</span>
                  <code v-if="tab !== 'native'" class="field__key">{{ p.key }}</code>
                </span>

                <button
                  v-if="p.type === 'bool'"
                  type="button"
                  class="bool"
                  :disabled="!!p.unsupported_reason"
                  @click="setParam(p, !draft.params[p.key])"
                >
                  <span class="switch" role="switch" :aria-checked="!!draft.params[p.key]" />
                  <span class="muted">{{ draft.params[p.key] ? '开启' : '关闭' }}</span>
                </button>
                <select
                  v-else-if="p.type === 'enum'"
                  class="input"
                  :value="draft.params[p.key] as string"
                  :disabled="!!p.unsupported_reason"
                  @change="setParam(p, ($event.target as HTMLSelectElement).value)"
                >
                  <option v-for="o in p.options" :key="o.value" :value="o.value">{{ o.label }}</option>
                </select>
                <textarea
                  v-else-if="p.multiline"
                  class="input area mono"
                  :class="{ 'is-invalid': fieldIssue(p.key)?.level === 'error' }"
                  :value="draft.params[p.key] as string ?? ''"
                  :disabled="!!p.unsupported_reason"
                  rows="3"
                  spellcheck="false"
                  placeholder="每行一项，也可用空格分隔"
                  @input="setParam(p, ($event.target as HTMLTextAreaElement).value)"
                />
                <input
                  v-else
                  class="input num"
                  :class="{ mono: p.type === 'string', 'is-invalid': fieldIssue(p.key)?.level === 'error' }"
                  :type="p.type === 'string' ? 'text' : 'number'"
                  :min="p.min"
                  :max="p.max"
                  :step="p.step ?? (p.type === 'int' ? 1 : 'any')"
                  :value="draft.params[p.key] ?? ''"
                  :disabled="!!p.unsupported_reason"
                  spellcheck="false"
                  @input="setParam(p, ($event.target as HTMLInputElement).value)"
                />
                <span v-if="p.unsupported_reason" class="field__hint">不可用：{{ p.unsupported_reason }}</span>
                <span v-else-if="fieldIssue(p.key)" :class="fieldIssue(p.key)!.level === 'error' ? 'field__error' : 'field__warn'">
                  {{ fieldIssue(p.key)!.message }}
                </span>
                <span v-else-if="p.help" class="field__hint">{{ p.help }}</span>
              </div>
            </div>
          </div>
        </div>
        <EmptyState v-else-if="!installations.length && store.enginesLoaded" icon="cpu" title="当前底模类型没有可用的引擎安装实例" text="请先在“引擎管理”中接入支持该底模类型的引擎并完成环境核实。">
          <RouterLink :to="{ name: 'engines' }" class="btn btn--primary">前往引擎管理</RouterLink>
        </EmptyState>
      </div>
    </section>

    <aside class="page__side">
      <!-- Binding summary -->
      <section class="panel panel--pad stack bind">
        <span class="label-pill" style="align-self: flex-start">固定的安装实例</span>
        <template v-if="installation">
          <strong>{{ engineName(installation.engine_id) }} / {{ installation.label }}</strong>
          <div class="row" style="flex-wrap: wrap; gap: 6px">
            <span class="chip mono">{{ shortCommit(installation.revision.commit) ?? '源码指纹' }}</span>
            <span class="chip">Py {{ installation.python_version ?? '?' }}</span>
            <StatusTag v-bind="installStateMeta[installation.state]" />
            <StatusTag v-bind="verificationMeta[installation.verification]" :dot="false" />
          </div>
          <p class="field__hint">提交时绑定该实例与环境快照，之后切换默认版本不会影响此任务。</p>
        </template>
        <p v-else class="muted">未选择</p>
      </section>

      <!-- Issues + preview -->
      <section class="panel preview">
        <div class="preview__head">
          <SegTabs
            v-model="previewMode"
            :options="[
              { value: 'config', label: result?.native_format ? `原生配置 .${result.native_format}` : '原生配置' },
              { value: 'argv', label: '启动参数' },
            ]"
          />
          <span class="spacer" />
          <button class="btn btn--ghost btn--icon btn--sm" title="复制" @click="copyPreview">
            <AppIcon name="link" :size="13" />
          </button>
        </div>

        <div v-if="errors.length || warnings.length" class="preview__issues">
          <div v-for="(i, n) in [...errors, ...warnings]" :key="n" class="pissue" :class="`is-${i.level}`">
            <AppIcon :name="i.level === 'error' ? 'x' : 'alert'" :size="12" />
            <span>{{ i.message }}</span>
          </div>
        </div>

        <pre class="preview__code scroll" :class="{ 'is-stale': validating || stale }">{{
          previewMode === 'config'
            ? result?.native_config ?? '— 尚未检测：点击下方“检测参数”生成 —'
            : argvPreview || '— 尚未检测：点击下方“检测参数”生成 —'
        }}</pre>

        <div class="preview__foot">
          <button class="btn btn--primary submit" :disabled="!canSubmit || submitting" @click="submit">
            <AppIcon name="star" :size="14" />
            提交到队列
          </button>
          <button class="btn btn--ghost" :disabled="validating || !caps" title="手动检测表单与参数是否合法" @click="runCheck">
            <AppIcon name="search" :size="14" />
            {{ validating ? '检测中…' : '检测参数' }}
          </button>
          <p v-if="installation && installation.state !== 'ready'" class="field__error">安装实例环境未就绪，不能进入队列。</p>
          <p v-else-if="previewOnly" class="field__warn">该引擎当前仅支持配置预览，尚不能提交训练。</p>
          <p v-else-if="stale" class="field__warn">参数已改动，提交前会自动重新检测；也可点“检测参数”立即查看。</p>
        </div>
      </section>
    </aside>

    <ModalDialog v-if="resetOpen" title="清空暂存的表单内容？" @close="resetOpen = false">
      <p>将清除本机暂存的任务名称、路径与全部训练参数；底模类型与引擎实例会保留并重新填入默认参数。</p>
      <p class="muted">只影响此界面暂存的内容，不影响已提交任务与引擎环境。</p>
      <template #footer>
        <button class="btn btn--ghost" @click="resetOpen = false">取消</button>
        <button class="btn btn--danger" @click="resetStagedForm">确认清空</button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
.page {
  grid-template-columns: minmax(0, 1fr) 360px;
}
.page__header,
.page__body {
  padding-left: 44px;
}
.form {
  display: flex;
  flex-direction: column;
  gap: 28px;
}
.block {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.block > .label-pill {
  align-self: flex-start;
}
.restored {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  background: var(--info-bg);
  color: var(--info);
  font-size: 12px;
}
.grid2 {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px 18px;
}
.grid3 {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px 18px;}
.span2 {
  grid-column: span 2;
}
.span3 {
  grid-column: 1 / -1;
}
.section {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px 18px;
  border-radius: var(--radius-md);
  background: rgba(0, 0, 0, 0.12);
}
.field__key {
  font-size: 10px;
  color: var(--text-3);
}
.field__warn {
  font-size: 11px;
  color: var(--warn);
}
.field.is-off {
  opacity: 0.55;
}
.bool {
  display: flex;
  align-items: center;
  gap: 10px;
  height: 36px;
}

.bind {
  gap: 10px;
}
.preview {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
}
.preview__head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 14px 10px;
}
.preview__head .btn--icon {
  width: 28px;
}
.preview__issues {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 0 14px 10px;
}
.pissue {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  font-size: 11.5px;
}
.pissue .icon {
  margin-top: 3px;
}
.pissue.is-error {
  color: var(--danger);
}
.pissue.is-warning {
  color: var(--warn);
}
.preview__code {
  flex: 1;
  min-height: 0;
  margin: 0 14px;
  padding: 14px;
  border-radius: var(--radius-md);
  background: var(--code-bg);
  color: var(--text-2);
  font-size: 11.5px;
  line-height: 1.6;
  white-space: pre;
  transition: opacity 0.2s;
}
.preview__code.is-stale {
  opacity: 0.55;
}
.preview__foot {
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.submit {
  width: 100%;
  height: 44px;
}

@media (max-width: 1180px) {
  .page {
    grid-template-columns: minmax(0, 1fr) 300px;
  }
  .grid3 {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
