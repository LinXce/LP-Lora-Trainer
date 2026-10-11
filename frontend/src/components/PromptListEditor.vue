<script setup lang="ts">
/**
 * One row per sample prompt. The value stays the newline-joined string the
 * backend and adapters already use ("one line = one sample"), so a row can
 * never contain a line break: Enter inserts a new row, and pasting several
 * lines splits them into rows.
 */
import { nextTick, ref, watch } from 'vue'
import AppIcon from './AppIcon.vue'

const props = withDefaults(defineProps<{
  modelValue: string
  disabled?: boolean
  invalid?: boolean
  placeholder?: string
  addLabel?: string
}>(), {
  disabled: false,
  invalid: false,
  placeholder: '输入一个采样提示词，可包含引擎支持的行内选项',
  addLabel: '添加采样',
})

const emit = defineEmits<{ 'update:modelValue': [value: string] }>()

interface Row { id: number; text: string }
let nextId = 0
const rows = ref<Row[]>([])
const root = ref<HTMLElement>()

function parse(value: string): string[] {
  const lines = value.replace(/\r\n?/g, '\n').split('\n')
  return lines.length === 1 && !lines[0].trim() ? [] : lines
}
function serialize() {
  return rows.value.map(r => r.text).join('\n')
}
// Rebuild rows only for outside changes (template applied, draft restored), so
// row identity — and therefore focus — survives normal typing.
watch(() => props.modelValue, (value) => {
  if (value === serialize()) return
  rows.value = parse(value).map(text => ({ id: nextId++, text }))
}, { immediate: true })

function commit() {
  emit('update:modelValue', serialize())
}
async function focusRow(index: number, caret: 'start' | 'end' = 'end') {
  await nextTick()
  const el = root.value?.querySelectorAll<HTMLTextAreaElement>('.prompt-list__input')[index]
  if (!el) return
  el.focus()
  const pos = caret === 'start' ? 0 : el.value.length
  el.setSelectionRange(pos, pos)
  autosize(el)
}
function autosize(el: HTMLTextAreaElement) {
  el.style.height = 'auto'
  el.style.height = `${el.scrollHeight + 2}px`
}

function insert(index: number, texts: string[] = ['']) {
  rows.value.splice(index, 0, ...texts.map(text => ({ id: nextId++, text })))
  commit()
}
function add() {
  insert(rows.value.length)
  void focusRow(rows.value.length - 1)
}
function remove(index: number) {
  rows.value.splice(index, 1)
  commit()
  if (rows.value.length) void focusRow(Math.min(index, rows.value.length - 1))
}
function duplicate(index: number) {
  insert(index + 1, [rows.value[index].text])
  void focusRow(index + 1)
}
function move(index: number, delta: -1 | 1) {
  const target = index + delta
  if (target < 0 || target >= rows.value.length) return
  const [row] = rows.value.splice(index, 1)
  rows.value.splice(target, 0, row)
  commit()
  void focusRow(target)
}

function onInput(index: number, event: Event) {
  const el = event.target as HTMLTextAreaElement
  // Line breaks can still arrive via IME or drag-and-drop; turn them into rows.
  const parts = el.value.replace(/\r\n?/g, '\n').split('\n')
  rows.value[index].text = parts[0]
  if (parts.length > 1) {
    insert(index + 1, parts.slice(1))
    void focusRow(index + parts.length - 1)
    return
  }
  autosize(el)
  commit()
}
function onPaste(index: number, event: ClipboardEvent) {
  const text = event.clipboardData?.getData('text/plain') ?? ''
  if (!/[\r\n]/.test(text)) return
  event.preventDefault()
  const el = event.target as HTMLTextAreaElement
  const row = rows.value[index]
  const before = row.text.slice(0, el.selectionStart)
  const after = row.text.slice(el.selectionEnd)
  const parts = text.replace(/\r\n?/g, '\n').replace(/\n+$/, '').split('\n')
  parts[0] = before + parts[0]
  const last = parts.length - 1
  parts[last] = parts[last] + after
  row.text = parts[0]
  insert(index + 1, parts.slice(1))
  void focusRow(index + last)
}
function onKeydown(index: number, event: KeyboardEvent) {
  if (event.isComposing) return
  const el = event.target as HTMLTextAreaElement
  if (event.key === 'Enter') {
    // Enter splits at the caret into a new sample, like a list editor.
    event.preventDefault()
    const row = rows.value[index]
    const rest = row.text.slice(el.selectionEnd)
    row.text = row.text.slice(0, el.selectionStart)
    insert(index + 1, [rest])
    void focusRow(index + 1, 'start')
  } else if (event.key === 'Backspace' && !el.value && rows.value.length > 1) {
    event.preventDefault()
    rows.value.splice(index, 1)
    commit()
    void focusRow(Math.max(0, index - 1))
  } else if (event.altKey && (event.key === 'ArrowUp' || event.key === 'ArrowDown')) {
    event.preventDefault()
    move(index, event.key === 'ArrowUp' ? -1 : 1)
  }
}

const vAutosize = { mounted: autosize, updated: autosize }
</script>

<template>
  <div ref="root" class="prompt-list" :class="{ 'is-invalid': invalid, 'is-disabled': disabled }">
    <TransitionGroup name="prompt-row" tag="ol" class="prompt-list__rows">
      <li v-for="(row, index) in rows" :key="row.id" class="prompt-list__item">
        <span class="prompt-list__index num">{{ index + 1 }}</span>
        <textarea
          v-autosize
          class="input mono prompt-list__input"
          :value="row.text"
          :disabled="disabled"
          :placeholder="placeholder"
          rows="1"
          maxlength="100000"
          spellcheck="false"
          :aria-label="`采样提示词 ${index + 1}`"
          @input="onInput(index, $event)"
          @paste="onPaste(index, $event)"
          @keydown="onKeydown(index, $event)"
        />
        <div class="prompt-list__tools">
          <button type="button" class="prompt-list__tool" :disabled="disabled || index === 0" title="上移（Alt+↑）" :aria-label="`上移第 ${index + 1} 个采样`" @click="move(index, -1)">
            <AppIcon name="chevron" :size="12" class="is-up" />
          </button>
          <button type="button" class="prompt-list__tool" :disabled="disabled || index === rows.length - 1" title="下移（Alt+↓）" :aria-label="`下移第 ${index + 1} 个采样`" @click="move(index, 1)">
            <AppIcon name="chevron" :size="12" class="is-down" />
          </button>
          <button type="button" class="prompt-list__tool" :disabled="disabled" title="复制" :aria-label="`复制第 ${index + 1} 个采样`" @click="duplicate(index)">
            <AppIcon name="layers" :size="12" />
          </button>
          <button type="button" class="prompt-list__tool is-danger" :disabled="disabled" title="删除" :aria-label="`删除第 ${index + 1} 个采样`" @click="remove(index)">
            <AppIcon name="trash" :size="12" />
          </button>
        </div>
      </li>
    </TransitionGroup>
    <p v-if="!rows.length" class="prompt-list__empty">暂无采样提示词</p>
    <div class="prompt-list__foot">
      <button type="button" class="btn btn--ghost btn--sm" :disabled="disabled" @click="add">
        <AppIcon name="plus" :size="14" />{{ addLabel }}
      </button>
      <span v-if="rows.length" class="prompt-list__count num">共 {{ rows.length }} 个采样 · Enter 新建下一条</span>
    </div>
  </div>
</template>

<style scoped>
.prompt-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.prompt-list__rows {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
}
.prompt-list__item {
  position: relative;
  display: grid;
  grid-template-columns: 22px minmax(0, 1fr) auto;
  align-items: start;
  gap: 8px;
}
.prompt-list__index {
  padding-top: 10px;
  color: var(--text-3);
  font-size: 11px;
  text-align: right;
}
.prompt-list__item:focus-within .prompt-list__index {
  color: var(--text);
}
.prompt-list__input {
  display: block;
  min-height: 36px;
  height: 36px;
  padding: 8px 12px;
  line-height: 1.45;
  resize: none;
  overflow: hidden;
}
.prompt-list__tools {
  display: flex;
  align-items: center;
  gap: 2px;
  height: 36px;
  opacity: 0;
  transition: opacity 0.15s;
}
.prompt-list__item:hover .prompt-list__tools,
.prompt-list__item:focus-within .prompt-list__tools {
  opacity: 1;
}
.prompt-list__tool {
  display: grid;
  place-items: center;
  width: 24px;
  height: 24px;
  border-radius: 7px;
  color: var(--text-3);
  transition: background 0.15s, color 0.15s;
}
.prompt-list__tool:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.08);
  color: var(--text);
}
.prompt-list__tool.is-danger:hover:not(:disabled) {
  background: var(--danger-bg);
  color: var(--danger);
}
.prompt-list__tool:disabled {
  opacity: 0.3;
}
.prompt-list__tool .is-up {
  transform: rotate(-90deg);
}
.prompt-list__tool .is-down {
  transform: rotate(90deg);
}
.prompt-list__empty {
  margin: 0;
  padding: 14px;
  border: 1px dashed var(--line);
  border-radius: var(--radius-sm);
  color: var(--text-3);
  font-size: 12px;
  text-align: center;
}
.prompt-list__foot {
  display: flex;
  align-items: center;
  gap: 12px;
}
.prompt-list__count {
  color: var(--text-3);
  font-size: 11px;
}
.prompt-list.is-invalid .prompt-list__input,
.prompt-list.is-invalid .prompt-list__empty {
  border-color: var(--danger);
}
.prompt-list.is-disabled .prompt-list__tools {
  display: none;
}
@media (hover: none) {
  .prompt-list__tools {
    opacity: 1;
  }
}

.prompt-row-enter-active,
.prompt-row-leave-active {
  transition: opacity 0.18s var(--ease), transform 0.22s var(--ease);
}
.prompt-row-enter-from,
.prompt-row-leave-to {
  opacity: 0;
  transform: translateY(-4px);
}
.prompt-row-leave-active {
  position: absolute;
  left: 0;
  right: 0;
}
.prompt-row-move {
  transition: transform 0.22s var(--ease);
}
</style>
