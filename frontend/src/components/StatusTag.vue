<script setup lang="ts">
import type { Tone } from '@/features/format'

withDefaults(defineProps<{ tone?: Tone; label: string; dot?: boolean }>(), { tone: 'neutral', dot: true })
</script>

<template>
  <span class="tag" :class="`tag--${tone}`">
    <span v-if="dot" class="tag__dot" />
    {{ label }}
  </span>
</template>

<style scoped>
.tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 22px;
  padding: 0 10px;
  border-radius: var(--radius-pill);
  font-size: 11px;
  font-weight: 500;
  white-space: nowrap;
  background: rgba(255, 255, 255, 0.07);
  color: var(--text-2);
}
.tag__dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: currentColor;
}
.tag--ok {
  background: var(--ok-bg);
  color: var(--ok);
}
.tag--warn {
  background: var(--warn-bg);
  color: var(--warn);
}
.tag--danger {
  background: var(--danger-bg);
  color: var(--danger);
}
.tag--info {
  background: var(--info-bg);
  color: var(--info);
}
.tag--live {
  background: var(--light);
  color: var(--ink);
}
.tag--live .tag__dot {
  animation: pulse 1.4s ease-in-out infinite;
}
</style>
