<script lang="ts">
/** Module scope: every dialog instance gets its own deterministic title id. */
let titleSeq = 0
</script>

<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'
import AppIcon from './AppIcon.vue'

const props = defineProps<{ title: string; width?: number }>()
const emit = defineEmits<{ close: [] }>()

const dialog = ref<HTMLDialogElement>()
const titleId = `modal-title-${++titleSeq}`

function onKey(e: KeyboardEvent) {
  if (e.key !== 'Escape') return
  // Every open dialog listens on the window; nested dialogs must not all close
  // at once, so only the dialog containing the event target reacts.
  const root = dialog.value
  const target = e.target
  if (!root || !(target instanceof Node) || !root.contains(target)) return
  emit('close')
}
onMounted(() => {
  dialog.value?.showModal()
  window.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<template>
  <dialog
    ref="dialog"
    class="modal"
    aria-modal="true"
    :aria-labelledby="titleId"
    :style="{ width: `${props.width ?? 480}px` }"
    @cancel.prevent="emit('close')"
    @click.self="emit('close')"
  >
    <div class="modal__inner">
      <header class="modal__head">
        <h3 :id="titleId" class="title-md">{{ title }}</h3>
        <button class="btn btn--ghost btn--icon btn--sm" aria-label="关闭" @click="emit('close')">
          <AppIcon name="x" :size="14" />
        </button>
      </header>
      <div class="modal__body"><slot /></div>
      <footer v-if="$slots.footer" class="modal__foot"><slot name="footer" /></footer>
    </div>
  </dialog>
</template>

<style scoped>
.modal {
  max-width: calc(100vw - 48px);
  padding: 0;
  border: 1px solid var(--frame-edge);
  border-radius: var(--radius-lg);
  background: var(--panel-raised);
  color: var(--text);
  box-shadow: 0 30px 80px rgba(0, 0, 0, 0.55);
}
.modal::backdrop {
  background: rgba(0, 0, 0, 0.55);
  backdrop-filter: blur(3px);
}
.modal[open] {
  animation: fade-up 0.22s var(--ease);
}
.modal__inner {
  display: flex;
  flex-direction: column;
}
.modal__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 20px 6px;
}
.modal__head .btn--icon {
  width: 28px;
}
.modal__body {
  padding: 12px 20px 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.modal__foot {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 14px 20px;
  border-top: 1px solid var(--line);
}
</style>
