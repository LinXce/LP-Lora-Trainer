<script setup lang="ts">
import { onBeforeUnmount, onMounted } from 'vue'
import AppShell from '@/components/AppShell.vue'
import WindowTitleBar from '@/components/WindowTitleBar.vue'
import ToastHost from '@/components/ToastHost.vue'
import { startLiveUpdates, stopLiveUpdates } from '@/features/store'

onMounted(startLiveUpdates)
onBeforeUnmount(stopLiveUpdates)
</script>

<template>
  <div class="app-window">
    <WindowTitleBar />
    <div class="app-workspace">
      <AppShell>
        <RouterView v-slot="{ Component, route }">
          <Transition name="page" mode="out-in">
            <component :is="Component" :key="route.name" />
          </Transition>
        </RouterView>
      </AppShell>
    </div>
  </div>
  <ToastHost />
</template>

<style>
.app-window {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.app-workspace {
  flex: 1;
  min-height: 0;
}
.page-enter-active,
.page-leave-active {
  transition: opacity 0.18s var(--ease), transform 0.22s var(--ease);
}
.page-enter-from {
  opacity: 0;
  transform: translateY(6px);
}
.page-leave-to {
  opacity: 0;
}
</style>
