<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import ModalDialog from './ModalDialog.vue'
import AppIcon from './AppIcon.vue'
import PromptListEditor from './PromptListEditor.vue'
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
function lines(prompt: string) {
  return prompt.split(/\r?\n/).filter(line => line.trim())
}
const currentCount = computed(() => lines(props.modelValue).length)
function apply(append = false) {
  if (!selected.value || props.disabled) return
  const text = selected.value.prompt
  emit('update:modelValue', append && props.modelValue.trim() ? props.modelValue.trimEnd() + '\n' + text : text)
  mode.value = null
  toast(append ? `已追加 ${lines(text).length} 个采样，可继续编辑` : '提示词已复制到当前任务，可继续编辑', 'ok')
}
function openSave() {
  name.value = ''; snapshot.value = props.modelValue; mode.value = 'save'
}
async function save() {
  if (saving.value || !name.value.trim() || !snapshot.value.trim()) return
  saving.value = true
  try {
    await api.promptTemplates.create({ name: name.value, prompt: lines(snapshot.value).join('\n') })
    mode.value = null; toast('提示词模板已保存', 'ok')
  } catch (err) { toastError(err, '保存提示词模板') }
  finally { saving.value = false }
}
</script>

<template>
  <div class="sampling-prompt field">
    <PromptListEditor
      :model-value="modelValue"
      :disabled="disabled"
      :invalid="invalid"
      placeholder="例如：portrait, soft lighting"
      @update:model-value="emit('update:modelValue', $event)"
    />
    <div class="sampling-prompt__bar">
      <button type="button" class="chip" :disabled="disabled" @click="openPicker"><AppIcon name="file" :size="13" />从模板添加</button>
      <button type="button" class="chip" :disabled="disabled || !modelValue.trim()" @click="openSave"><AppIcon name="download" :size="13" />保存为模板</button>
      <span class="spacer" />
      <RouterLink class="sampling-prompt__manage" :to="{ name: 'prompt-templates' }">管理提示词模板<AppIcon name="chevron" :size="12" /></RouterLink>
    </div>
    <p class="sampling-prompt__hint">每一项对应一次采样，留空则不生成预览图；Enter 新建下一条，粘贴多行会自动拆分。</p>
    <ModalDialog v-if="mode === 'pick'" title="选择提示词模板" :width="640" @close="mode = null">
      <p class="muted">模板内容复制到当前任务，不会随模板的修改而改变。</p>
      <p v-if="loading" class="muted">正在加载…</p>
      <div v-else-if="error" class="field"><p class="field__error">{{ error }}</p><button class="btn btn--ghost" @click="openPicker">重试</button></div>
      <div v-else-if="templates.length" class="picker">
        <div class="picker__list" role="listbox" aria-label="提示词模板">
          <button
            v-for="item in templates"
            :key="item.template_id"
            type="button"
            role="option"
            class="picker__item"
            :class="{ 'is-on': item.template_id === selectedId }"
            :aria-selected="item.template_id === selectedId"
            @click="selectedId = item.template_id"
            @dblclick="apply()"
          >
            <span class="picker__name truncate">{{ item.name }}</span>
            <span class="picker__count num">{{ lines(item.prompt).length }} 个采样</span>
          </button>
        </div>
        <ol v-if="selected" class="picker__preview mono" aria-label="模板内容预览">
          <li v-for="(line, n) in lines(selected.prompt)" :key="n">
            <span class="picker__no num">{{ n + 1 }}</span>
            <span class="picker__text">{{ line }}</span>
          </li>
        </ol>
      </div>
      <p v-else class="muted">还没有模板。可以先输入一次性提示词并保存为模板。</p>
      <template #footer>
        <button class="btn btn--ghost" @click="mode = null">取消</button>
        <button class="btn btn--ghost" :disabled="loading || !!error || !selected" @click="apply(true)"><AppIcon name="plus" :size="14" />追加到列表末尾</button>
        <button class="btn btn--primary" :disabled="loading || !!error || !selected" @click="apply()">{{ currentCount ? `替换当前 ${currentCount} 个采样` : '使用此模板' }}</button>
      </template>
    </ModalDialog>
    <ModalDialog v-if="mode === 'save'" title="保存提示词模板" :width="640" @close="!saving && (mode = null)">
      <label class="field"><span class="field__label">模板名称</span><input v-model="name" class="input" maxlength="200" :disabled="saving" placeholder="例如：人物肖像采样" /></label>
      <div class="field"><span class="field__label">采样提示词（{{ lines(snapshot).length }} 个）</span><div class="save__list"><PromptListEditor v-model="snapshot" :disabled="saving" placeholder="例如：portrait, soft lighting" /></div></div>
      <span class="field__hint">可在此调整后再保存；保存模板不会修改当前任务的采样列表。</span>
      <template #footer>
        <button class="btn btn--ghost" :disabled="saving" @click="mode = null">取消</button>
        <button class="btn btn--primary" :disabled="saving || !name.trim() || !snapshot.trim()" @click="save">{{ saving ? '保存中…' : '保存模板' }}</button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
.sampling-prompt__bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  border-top: 1px solid var(--line);
  padding-top: 10px;
}
.sampling-prompt__bar .chip {
  height: 26px;
  font-size: 11px;
}
.sampling-prompt__manage {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  color: var(--text-3);
  font-size: 11px;
  text-decoration: none;
  transition: color 0.15s;
}
.sampling-prompt__manage:hover {
  color: var(--text);
}
.sampling-prompt__hint {
  margin: 0;
  color: var(--text-3);
  font-size: 11px;
  line-height: 1.5;
}

/* Template picker: list on the left, numbered sample preview on the right. */
.picker {
  display: grid;
  grid-template-columns: 220px minmax(0, 1fr);
  gap: 12px;
  height: min(46vh, 380px);
}
.picker__list {
  display: flex;
  flex-direction: column;
  gap: 4px;
  overflow-y: auto;
  min-height: 0;
}
.picker__item {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  padding: 9px 12px;
  border-radius: var(--radius-sm);
  text-align: left;
  color: var(--text-2);
  transition: background 0.15s, color 0.15s;
}
.picker__item:hover {
  background: rgba(255, 255, 255, 0.06);
  color: var(--text);
}
.picker__item.is-on {
  background: var(--light);
  color: var(--ink);
}
.picker__name {
  max-width: 100%;
  font-size: 13px;
  font-weight: 600;
}
.picker__count {
  font-size: 11px;
  opacity: 0.65;
}
.picker__preview {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 0;
  margin: 0;
  padding: 12px 14px;
  overflow-y: auto;
  list-style: none;
  border-radius: var(--radius-sm);
  background: var(--field);
  font-size: 12px;
  line-height: 1.5;
}
.picker__preview li {
  display: flex;
  gap: 10px;
}
.picker__no {
  flex: none;
  width: 18px;
  text-align: right;
  color: var(--text-3);
}
.picker__text {
  min-width: 0;
  color: var(--text-2);
  overflow-wrap: anywhere;
}
.save__list {
  max-height: min(40vh, 360px);
  overflow-y: auto;
  padding-right: 4px;
}
</style>
