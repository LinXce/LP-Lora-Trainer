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
import { api } from '@/api'
import { refreshTasks, store } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import { engineName, installStateMeta, shortCommit, verificationMeta } from '@/features/format'
import type { BaseModel, Dataset, EngineCapabilities, ParamGroup, ParamSpec, TrainingDraft, ValidationResult } from '@/types/api'

const router = useRouter()

const draft = reactive<TrainingDraft>({
  name: '',
  installation_id: '',
  architecture: '',
  base_model_path: '',
  dataset_id: '',
  output_dir: '',
  params: {},
})

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
onBeforeUnmount(() => { seq += 1 })

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
      // Carry over values whose key still exists; report the ones that don't.
      const prev = { ...draft.params }
      const next: TrainingDraft['params'] = {}
      for (const p of c.params) next[p.key] = p.key in prev ? prev[p.key] : p.default
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

/* Validation is explicit: it runs when the user presses 检测参数 or submits, never
   on every keystroke. Typing only marks the previous result as outdated. */
const stale = ref(false)
watch(
  draft,
  () => {
    if (result.value) stale.value = true
  },
  { deep: true },
)

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
        <StatusTag
          v-if="result"
          :tone="validating ? 'info' : stale ? 'warn' : result.ok ? 'ok' : 'danger'"
          :label="validating ? '检测中…' : stale ? '参数已改动，需重新检测' : result.ok ? '配置有效' : `${errors.length} 个错误`"
        />
        <StatusTag v-else :tone="validating ? 'info' : 'neutral'" :label="validating ? '检测中…' : '尚未检测参数'" />
      </header>

      <div class="page__body form">
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
