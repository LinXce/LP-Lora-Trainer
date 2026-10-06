<script setup lang="ts">
import AppIcon from './AppIcon.vue'

defineProps<{ icon?: string; title: string; text?: string }>()
</script>

<template>
  <div class="empty">
    <div class="empty__art">
      <AppIcon :name="icon ?? 'image'" :size="34" />
    </div>
    <p class="title-md">{{ title }}</p>
    <p v-if="text" class="empty__text">{{ text }}</p>
    <div class="empty__actions"><slot /></div>
  </div>
</template>

<style scoped>
.empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 40px 24px;
  text-align: center;
  height: 100%;
}
.empty__art {
  display: grid;
  place-items: center;
  width: 76px;
  height: 76px;
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.06);
  color: var(--light);
  margin-bottom: 6px;
}
.empty__text {
  max-width: 380px;
  color: var(--text-3);
  font-size: 12px;
}
.empty__actions {
  display: flex;
  gap: 8px;
  margin-top: 6px;
}
.empty__actions:empty {
  display: none;
}
</style>
