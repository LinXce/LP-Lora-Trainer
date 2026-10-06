<script setup lang="ts" generic="T extends string">
/** Pill segmented control — the outline/filled chip rows in the reference. */
defineProps<{ options: { value: T; label: string; count?: number }[]; modelValue: T }>()
const emit = defineEmits<{ 'update:modelValue': [T] }>()
</script>

<template>
  <div class="seg" role="tablist">
    <button
      v-for="o in options"
      :key="o.value"
      type="button"
      role="tab"
      class="chip"
      :class="{ 'is-on': o.value === modelValue }"
      :aria-selected="o.value === modelValue"
      @click="emit('update:modelValue', o.value)"
    >
      {{ o.label }}
      <span v-if="o.count !== undefined" class="seg__count num">{{ o.count }}</span>
    </button>
  </div>
</template>

<style scoped>
.seg {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.seg__count {
  font-size: 11px;
  opacity: 0.65;
}
</style>
