<script setup lang="ts">
/** Styled popup; native options remain the single source for existing callers. */
import { computed, nextTick, onBeforeUnmount, onMounted, ref, shallowRef, useAttrs, watch } from 'vue'
import AppIcon from './AppIcon.vue'
defineOptions({ inheritAttrs: false })
// Boolean union props otherwise default to false when omitted, masking :value bindings.
const props = withDefaults(defineProps<{
  modelValue?: string | number | boolean | null
  value?: string | number | boolean | null
  disabled?: boolean
  placeholder?: string
}>(), { modelValue: undefined, value: undefined })
const emit = defineEmits<{
  'update:modelValue': [string]
  change: [Event]
}>()
const attrs = useAttrs()
const root = ref<HTMLElement>()
const trigger = ref<HTMLButtonElement>()
const native = ref<HTMLSelectElement>()
const popup = ref<HTMLElement>()
// A body-level popup is behind a modal dialog and inert, regardless of z-index.
// Keep it in the owning dialog; native popovers also escape clipping/transforms.
const popupTarget = shallowRef<HTMLElement | string>('body')
const supportsPopover = typeof HTMLElement !== 'undefined' && typeof HTMLElement.prototype.showPopover === 'function'
const open = ref(false)
const activeIndex = ref(-1)
const options = ref<{ value: string; label: string; disabled: boolean }[]>([])
const selected = computed(() => String(props.modelValue ?? props.value ?? ''))
const label = computed(() => options.value.find(o => o.value === selected.value)?.label ?? props.placeholder ?? '\u8bf7\u9009\u62e9')
const menuStyle = ref<Record<string, string>>({})
const above = ref(false)
const id = `lp-select-${Math.random().toString(36).slice(2)}`
let observer: MutationObserver | undefined
let search = ''
let searchAt = 0

function syncOptions() {
  if (!native.value) return
  options.value = Array.from(native.value.options).map(o => ({ value: o.value, label: o.label, disabled: o.disabled || !!o.closest('optgroup')?.disabled }))
  native.value.value = selected.value
  if (open.value) { activeIndex.value = options.value.findIndex(o => o.value === selected.value && !o.disabled); position() }
}
function position() {
  if (!root.value || !open.value) return
  const rect = root.value.getBoundingClientRect()
  // Older webviews without popovers keep the list inside the modal scrollport.
  // Local absolute coordinates remain correct during the dialog's transform animation.
  const modal = !supportsPopover && popupTarget.value instanceof HTMLElement ? popupTarget.value : undefined
  const modalRect = modal?.getBoundingClientRect()
  const top = modal && modalRect ? modalRect.top + modal.clientTop : 0
  const bottom = modal ? top + modal.clientHeight : window.innerHeight
  const below = bottom - rect.bottom - 8
  const before = rect.top - top - 8
  above.value = below < 160 && before > below
  const space = Math.max(40, above.value ? before : below)
  menuStyle.value = {
    position: modal ? 'absolute' : 'fixed',
    left: `${modal && modalRect ? rect.left - modalRect.left - modal.clientLeft + modal.scrollLeft : rect.left}px`,
    width: `${rect.width}px`, maxHeight: `${Math.min(280, space)}px`,
    ...(above.value
      ? { bottom: `${modal ? modal.clientHeight - (rect.top - top + modal.scrollTop) - 1 : window.innerHeight - rect.top - 1}px` }
      : { top: `${modal ? rect.bottom - top + modal.scrollTop - 1 : rect.bottom - 1}px` }),
  }
}
function close(focus = false) {
  open.value = false
  if (focus) trigger.value?.focus()
}
async function show() {
  if (props.disabled) return
  popupTarget.value = root.value?.closest('dialog') ?? 'body'
  syncOptions()
  open.value = true
  activeIndex.value = options.value.findIndex(o => o.value === selected.value && !o.disabled)
  if (activeIndex.value < 0) activeIndex.value = options.value.findIndex(o => !o.disabled)
  position()
  await nextTick()
  if (!open.value || !popup.value?.isConnected) return
  if (supportsPopover) popup.value.showPopover()
  position()
  reveal()
}
function reveal() {
  popup.value?.querySelector(`[data-index="${activeIndex.value}"]`)?.scrollIntoView({ block: 'nearest' })
}
function choose(index: number) {
  const option = options.value[index]
  if (!option || option.disabled || !native.value) return
  native.value.value = option.value
  // Preserve both v-model and existing native @change handlers (including target.value).
  native.value.dispatchEvent(new Event('change', { bubbles: true }))
  close(true)
}
function changed(event: Event) {
  emit('update:modelValue', (event.target as HTMLSelectElement).value)
  emit('change', event)
}
function move(direction: number) {
  const count = options.value.length
  for (let step = 1; step <= count; step++) {
    const i = (activeIndex.value + direction * step + count * 2) % count
    if (!options.value[i].disabled) { activeIndex.value = i; break }
  }
  void nextTick(reveal)
}
function keydown(event: KeyboardEvent) {
  if (props.disabled) return
  if (event.key === 'Tab') { close(); return }
  if (event.key === 'Escape') {
    if (open.value) {
      event.preventDefault()
      event.stopPropagation() // Close this list, not the owning installation dialog.
      close(true)
    }
    return
  }
  if (['Enter', ' ', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
    event.preventDefault()
    if (!open.value) { void show(); return }
    if (event.key === 'Enter' || event.key === ' ') { choose(activeIndex.value); return }
    if (event.key === 'Home' || event.key === 'End') {
      activeIndex.value = event.key === 'Home' ? -1 : options.value.length
      move(event.key === 'Home' ? 1 : -1)
    } else move(event.key === 'ArrowDown' ? 1 : -1)
  } else if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
    event.preventDefault()
    if (!open.value) void show()
    const now = performance.now()
    search = (now - searchAt > 700 ? '' : search) + event.key.toLocaleLowerCase()
    searchAt = now
    const found = options.value.findIndex(o => !o.disabled && o.label.toLocaleLowerCase().startsWith(search))
    if (found >= 0) { activeIndex.value = found; void nextTick(reveal) }
  }
}
function outside(event: Event) {
  const target = event.target as Node
  if (!root.value?.contains(target) && !popup.value?.contains(target)) close()
}
function focusout(event: FocusEvent) {
  const target = event.relatedTarget as Node | null
  if (!target || (!root.value?.contains(target) && !popup.value?.contains(target))) close()
}
watch(selected, () => {
  if (native.value) native.value.value = selected.value
  if (open.value) {
    activeIndex.value = options.value.findIndex(o => o.value === selected.value && !o.disabled)
    void nextTick(reveal)
  }
})
watch(() => props.disabled, value => { if (value) close() })
onMounted(() => {
  syncOptions()
  observer = new MutationObserver(syncOptions)
  if (native.value) observer.observe(native.value, { childList: true, subtree: true, attributes: true, characterData: true })
  document.addEventListener('pointerdown', outside, true)
  window.addEventListener('resize', position)
  window.addEventListener('scroll', position, true)
})
onBeforeUnmount(() => {
  observer?.disconnect()
  document.removeEventListener('pointerdown', outside, true)
  window.removeEventListener('resize', position)
  window.removeEventListener('scroll', position, true)
})
</script>

<template>
  <div ref="root" class="select-control" :class="{ 'is-open': open, 'is-above': above, 'is-disabled': disabled }" v-bind="attrs" @focusout="focusout">
    <select ref="native" :value="selected" :disabled="disabled" hidden aria-hidden="true" tabindex="-1" @change="changed"><slot /></select>
    <button ref="trigger" type="button" class="select-control__trigger" role="combobox" aria-haspopup="listbox"
      :aria-expanded="open" :aria-controls="id" :aria-label="attrs['aria-label'] as string | undefined"
      :aria-activedescendant="open && activeIndex >= 0 ? `${id}-${activeIndex}` : undefined"
      :disabled="disabled" @click="open ? close() : show()" @keydown="keydown">
      <span class="truncate">{{ label }}</span><AppIcon name="chevron-down" :size="13" />
    </button>
    <Teleport :to="popupTarget">
      <Transition name="select-popup">
        <div v-if="open" :id="id" ref="popup" :popover="supportsPopover ? 'manual' : undefined" :inert="!open" role="listbox" class="select-popup" :class="{ 'is-above': above }" :style="menuStyle">
          <button v-for="(option, index) in options" :id="`${id}-${index}`" :key="`${option.value}-${index}`" type="button"
            role="option" class="select-popup__option" :class="{ 'is-active': activeIndex === index }"
            :aria-selected="option.value === selected" :disabled="option.disabled" :data-index="index" tabindex="-1"
            @pointerdown.prevent @mouseenter="!option.disabled && (activeIndex = index)" @click="choose(index)">
            <span>{{ option.label }}</span><AppIcon v-if="option.value === selected" name="check" :size="13" />
          </button>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<style scoped>
.select-control.input { padding: 0; background-image: none; }
.select-control { position: relative; min-width: 0; }
.select-control__trigger { display: flex; align-items: center; justify-content: space-between; gap: 10px; width: 100%; height: 34px; padding: 0 14px; text-align: left; outline: none; }
.select-control__trigger svg { flex: none; transition: transform 220ms ease; }
.select-control.is-open { border-color: var(--line-strong); border-bottom-left-radius: 0; border-bottom-right-radius: 0; }
.select-control.is-open.is-above { border-radius: 0 0 var(--radius-sm) var(--radius-sm); }
.select-control.is-open svg { transform: rotate(180deg); }
.select-control:focus-within { border-color: var(--line-strong); background: var(--field-focus); }
.select-control.is-disabled { opacity: .5; }
.select-popup { margin: 0; inset: auto; color: var(--text); box-sizing: border-box; position: fixed; z-index: 3000; overflow-y: auto; padding: 4px; background: var(--panel-raised); border: 1px solid var(--line-strong); border-top: 0; border-radius: 0 0 var(--radius-sm) var(--radius-sm); box-shadow: 0 12px 24px rgba(0,0,0,.32); transform-origin: top; }
.select-popup.is-above { border-top: 1px solid var(--line-strong); border-bottom: 0; border-radius: var(--radius-sm) var(--radius-sm) 0 0; transform-origin: bottom; }
.select-popup__option { display: flex; align-items: center; justify-content: space-between; gap: 10px; width: 100%; padding: 8px 10px; border-radius: 5px; text-align: left; font-size: 12px; white-space: normal; overflow-wrap: anywhere; }
.select-popup__option.is-active:not(:disabled) { background: rgba(255,255,255,.1); }
.select-popup__option[aria-selected='true'] { color: var(--light-strong); font-weight: 600; }
.select-popup__option:disabled { opacity: .45; cursor: not-allowed; }
.select-popup-enter-active, .select-popup-leave-active { transition: opacity 180ms ease, transform 180ms ease; }
.select-popup-enter-from, .select-popup-leave-to { opacity: 0; transform: scaleY(.96); }
</style>
