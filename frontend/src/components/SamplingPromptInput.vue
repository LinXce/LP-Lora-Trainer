<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import ModalDialog from './ModalDialog.vue'
import SelectInput from './SelectInput.vue'
import { api } from '@/api'
import { toast, toastError } from '@/features/toast'
import type { PromptTemplate } from '@/types/api'

const props = defineProps<{ modelValue: string; disabled?: boolean; invalid?: boolean }>()
const emit = defineEmits<{ 'update:modelValue': [value: string] }>()
const mode = ref<'pick' | 'save' | null>(null)
const templates = ref<PromptTemplate[]>([])
const selectedId = ref('')
const selected = computed(() => templates.value.find(t => t.template_id === selectedId.value))
const name = ref('')
const saving = ref(false)
const loading = ref(false)
const error = ref('')
const snapshot = ref('')

async function openPicker() {
  mode.value = 'pick'; error.value = ''; loading.value = true
  try {
    templates.value = await api.promptTemplates.list()
    selectedId.value = templates.value[0]?.template_id ?? ''
  } catch (err) { error.value = err instanceof Error ? err.message : String(err) }
  finally { loading.value = false }
}
function apply(append = false) {
  if (!selected.value || props.disabled) return
  const text = selected.value.prompt
  emit('update:modelValue', append && props.modelValue.trim() ? props.modelValue.trimEnd() + '\n' + text : text)
  mode.value = null
  toast('提示词已复制到当前任务，可继续编辑', 'ok')
}
function openSave() {
  name.value = ''; snapshot.value = props.modelValue; mode.value = 'save'
}
async function save() {
  if (saving.value || !name.value.trim() || !snapshot.value.trim()) return
  saving.value = true
  try {
    await api.promptTemplates.create({ name: name.value, prompt: snapshot.value })
    mode.value = null; toast('提示词模板已保存', 'ok')
  } catch (err) { toastError(err, '保存提示词模板') }
  finally { saving.value = false }
}
</script>

<template>
  <div class="sampling-prompt field">
    <textarea class="input area mono" :class="{ 'is-invalid': invalid }" :value="modelValue"
      :disabled="disabled" rows="6" maxlength="100000" aria-label="采样提示词" spellcheck="false"
      placeholder="直接输入一次性提示词，每行一个采样。例如：portrait, soft lighting"
      @input="emit('update:modelValue', ($event.target as HTMLTextAreaElement).value)" />
    <div class="row" style="flex-wrap: wrap; gap: 8px">
      <button type="button" class="btn btn--ghost btn--sm" :disabled="disabled" @click="openPicker">从模板添加</button>
      <button type="button" class="btn btn--ghost btn--sm" :disabled="disabled || !modelValue.trim()" @click="openSave">保存为模板</button>
      <RouterLink class="btn btn--ghost btn--sm" :to="{ name: 'prompt-templates' }">管理提示词模板</RouterLink>
    </div>
    <ModalDialog v-if="mode === 'pick'" title="选择提示词模板" :width="640" @close="mode = null">
      <p class="muted">模板内容复制到当前任务，不会随模板的修改而改变。</p>
      <p v-if="loading" class="muted">正在加载…</p>
      <div v-else-if="error" class="field"><p class="field__error">{{ error }}</p><button class="btn btn--ghost" @click="openPicker">重试</button></div>
      <template v-else-if="templates.length">
        <SelectInput class="input" :value="selectedId" aria-label="提示词模板" @change="selectedId = ($event.target as HTMLSelectElement).value">
          <option v-for="item in templates" :key="item.template_id" :value="item.template_id">{{ item.name }}</option>
        </SelectInput>
        <textarea class="input area mono" rows="8" :value="selected?.prompt ?? ''" aria-label="模板内容预览" readonly />
      </template>
      <p v-else class="muted">还没有模板。可以先输入一次性提示词并保存为模板。</p>
      <template #footer>
        <button class="btn btn--ghost" @click="mode = null">取消</button>
        <button class="btn btn--ghost" :disabled="loading || !!error || !selected" @click="apply(true)">追加到末尾</button>
        <button class="btn btn--primary" :disabled="loading || !!error || !selected" @click="apply()">使用（替换当前内容）</button>
      </template>
    </ModalDialog>
    <ModalDialog v-if="mode === 'save'" title="保存提示词模板" :width="640" @close="!saving && (mode = null)">
      <label class="field"><span class="field__label">模板名称</span><input v-model="name" class="input" maxlength="200" :disabled="saving" placeholder="例如：人物肖像采样" /></label>
      <label class="field"><span class="field__label">提示词内容</span><textarea v-model="snapshot" class="input area mono" rows="8" maxlength="100000" :disabled="saving" /></label>
      <span class="field__hint">保存模板不会修改当前任务；当前任务仍使用编辑框中的提示词。</span>
      <template #footer>
        <button class="btn btn--ghost" :disabled="saving" @click="mode = null">取消</button>
        <button class="btn btn--primary" :disabled="saving || !name.trim() || !snapshot.trim()" @click="save">{{ saving ? '保存中…' : '保存模板' }}</button>
      </template>
    </ModalDialog>
  </div>
</template>
