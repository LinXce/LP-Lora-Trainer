<script setup lang="ts">
import AppIcon from './AppIcon.vue'
import { dismissToast, toasts } from '@/features/toast'
</script>

<template>
  <div class="toasts" aria-live="polite">
    <TransitionGroup name="toast">
      <div v-for="t in toasts" :key="t.id" class="toast" :class="`toast--${t.tone}`">
        <AppIcon :name="t.tone === 'ok' ? 'check' : t.tone === 'danger' ? 'alert' : 'info'" :size="15" />
        <span class="toast__text selectable">{{ t.text }}</span>
        <button class="toast__x" aria-label="关闭" @click="dismissToast(t.id)">
          <AppIcon name="x" :size="12" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.toasts {
  position: fixed;
  right: 28px;
  bottom: 28px;
  z-index: 100;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
}
.toast {
  pointer-events: auto;
  display: flex;
  align-items: center;
  gap: 10px;
  max-width: 420px;
  padding: 10px 12px 10px 16px;
  border-radius: var(--radius-pill);
  background: var(--light-strong);
  color: var(--ink);
  font-weight: 500;
  box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45);
}
.toast--danger {
  background: #f1d2ce;
  color: #4a1a15;
}
.toast--ok {
  background: #d4e9d8;
  color: #183a20;
}
.toast__text {
  flex: 1;
  font-size: 12px;
}
.toast__x {
  display: grid;
  place-items: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  opacity: 0.6;
}
.toast__x:hover {
  opacity: 1;
  background: rgba(0, 0, 0, 0.08);
}
.toast-enter-active,
.toast-leave-active {
  transition: all 0.25s var(--ease);
}
.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateY(10px) scale(0.98);
}
</style>
