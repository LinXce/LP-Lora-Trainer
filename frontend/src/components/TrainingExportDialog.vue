<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import ModalDialog from './ModalDialog.vue'
import AppIcon from './AppIcon.vue'
import { api } from '@/api'
import { saveTextFile } from '@/api/bridge'
import { exportFilename, serializeTrainingDraft } from '@/features/trainingExport'
import { toast, toastError } from '@/features/toast'
import type { NativeParameterExport, TrainingExportSource } from '@/types/api'

const props = defineProps<{ source: TrainingExportSource; initialNative?: boolean }>()
const emit = defineEmits<{ close: [] }>()
const selected = ref<'portable' | 'native'>(props.initialNative ? 'native' : 'portable')
const native = ref<NativeParameterExport | null>(null)
const loading = ref(false)
const saving = ref(false)
const nativeError = ref('')
const suffix = computed(() => selected.value === 'portable' ? '.params.json' : `.${native.value?.native_format ?? 'toml'}`)
const filename = computed(() => exportFilename(props.source.name, suffix.value))
const canSave = computed(() => !saving.value && (selected.value === 'portable' || !!native.value))

async function loadNative() {
  if (loading.value) return
  if (!props.source.engine_id) { nativeError.value = '请先选择已识别类型的引擎；LP 完整备份仍可导出。'; return }
  loading.value = true
  nativeError.value = ''
  try { native.value = await api.training.exportParameters(props.source) }
  catch (err) { nativeError.value = err instanceof Error ? err.message : String(err) }
  finally { loading.value = false }
}
async function saveCompanion(file: { filename: string; text: string }) {
  if (saving.value) return
  saving.value = true
  try { if (await saveTextFile(file.filename, file.text)) toast('配套提示词已导出', 'ok') }
  catch (err) { toastError(err, '导出提示词') }
  finally { saving.value = false }
}
async function save() {
  if (!canSave.value) return
  const text = selected.value === 'portable'
    ? serializeTrainingDraft(props.source, props.source.engine_id, props.source.dataset_path ?? undefined)
    : native.value!.text
  saving.value = true
  try {
    if (await saveTextFile(filename.value, text)) {
      toast('参数已导出', 'ok')
      emit('close')
    }
  } catch (err) { toastError(err, '导出参数') }
  finally { saving.value = false }
}
onMounted(loadNative)
</script>

<template>
  <ModalDialog title="导出训练参数" :width="560" @close="!saving && emit('close')">
    <p class="muted">选择用途与格式。引擎原生格式由对应适配器生成；更换训练器或底模类型时，仍需确认参数含义与路径。</p>
    <fieldset class="export-options" :disabled="saving">
      <legend class="field__label">导出格式</legend>
      <label class="export-option" :class="{ 'is-selected': selected === 'portable' }">
        <input v-model="selected" type="radio" name="training-export-format" value="portable" />
        <span><strong>LP 完整备份 · JSON</strong><small>推荐用于本工具的导入、迁移与历史任务复用；保留全部表单参数及数据集路径。不要求环境就绪。</small></span>
      </label>
      <label class="export-option" :class="{ 'is-selected': selected === 'native', 'is-disabled': !native }">
        <input v-model="selected" type="radio" name="training-export-format" value="native" :disabled="!native" />
        <span><strong>{{ native ? `${native.engine_name} 原生参数 · ${native.native_format.toUpperCase()}` : '当前引擎原生参数' }}</strong>
          <small>{{ loading ? '正在通过适配器生成原生格式…' : native ? '用于当前引擎的参数配置，按原生键名与结构输出；不保留全部 LP 表单信息。' : '原生导出暂不可用，可继续导出 LP 完整备份。' }}</small></span>
      </label>
    </fieldset>
    <div v-if="nativeError" class="export-error" role="status">
      <span>{{ nativeError }}</span>
      <button v-if="source.engine_id" class="btn btn--ghost btn--sm" :disabled="saving || loading" @click="loadNative">重试</button>
    </div>
    <template v-if="selected === 'native' && native">
      <div v-if="native.files?.length" class="field">
        <span class="field__label">配套提示词文件（原生引擎必需）</span>
        <span class="field__hint">请先下载下列文件，再导出参数。运行原生引擎时将它们放在工作目录中，或在原生配置中调整路径；LP 完整备份不需要额外文件。</span>
        <button v-for="file in native.files" :key="file.filename" class="btn btn--ghost btn--sm" :disabled="saving" @click="saveCompanion(file)">
          <AppIcon name="download" :size="14" />下载 {{ file.filename }}
        </button>
      </div>
      <p v-for="(warning, i) in native.warnings" :key="i" class="export-notice">{{ warning }}</p>
      <p class="muted">此处导出参数文件，不包含模型、图片、缓存或训练环境；开始训练前请在目标训练器检查配置。</p>
    </template>
    <p v-if="selected === 'portable' || native" class="export-filename"><span class="muted">建议文件名</span><code>{{ filename }}</code></p>
    <template #footer>
      <button class="btn btn--ghost" :disabled="saving" @click="emit('close')">取消</button>
      <button class="btn btn--primary" :disabled="!canSave" @click="save">
        <AppIcon name="download" :size="14" />{{ saving ? '正在保存…' : '选择位置并导出' }}
      </button>
    </template>
  </ModalDialog>
</template>

<style scoped>
.export-options { display: flex; flex-direction: column; gap: 10px; border: 0; padding: 0; margin: 0; min-width: 0; }
.export-options legend { margin-bottom: 8px; }
.export-option { display: flex; align-items: flex-start; gap: 10px; padding: 12px; border: 1px solid var(--line); border-radius: var(--radius-md); background: var(--field); cursor: pointer; transition: border-color 0.2s var(--ease), background 0.2s var(--ease); }
.export-option.is-selected { border-color: var(--focus); }
.export-option.is-disabled { opacity: 0.6; cursor: default; }
.export-option input { margin-top: 3px; accent-color: var(--focus); }
.export-option span { display: flex; flex-direction: column; gap: 5px; }
.export-option strong { font-size: 13px; }
.export-option small { color: var(--text-3); line-height: 1.6; }
.export-option:focus-within { outline: 2px solid var(--focus); outline-offset: 2px; }
.export-notice, .export-error { padding: 10px 12px; border-radius: var(--radius-sm); background: var(--warn-bg); color: var(--warn); font-size: 12px; line-height: 1.7; }
.export-error { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.export-filename { display: flex; flex-direction: column; gap: 6px; }
.export-filename code { overflow-wrap: anywhere; font-size: 12px; }
</style>
