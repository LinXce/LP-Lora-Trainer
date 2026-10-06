<script setup lang="ts">
/**
 * Path field: native dialog when running inside pywebview, manual input
 * otherwise. The returned path is still validated by the backend.
 */
import { onMounted, ref } from 'vue'
import AppIcon from './AppIcon.vue'
import { bridgeReady, pickPath } from '@/api/bridge'

const props = defineProps<{
  modelValue: string
  kind: 'directory' | 'file'
  placeholder?: string
  dialogTitle?: string
  fileTypes?: string[]
  invalid?: boolean
  disabled?: boolean
}>()
const emit = defineEmits<{ 'update:modelValue': [string] }>()

const canPick = ref(false)
onMounted(async () => {
  canPick.value = await bridgeReady()
})

async function browse() {
  const p = await pickPath(props.kind, props.dialogTitle ?? (props.kind === 'directory' ? '选择目录' : '选择文件'), props.fileTypes)
  if (p) emit('update:modelValue', p)
}
</script>

<template>
  <div class="path">
    <input
      class="input mono"
      :class="{ 'is-invalid': invalid }"
      :value="modelValue"
      :placeholder="placeholder ?? (kind === 'directory' ? '目录绝对路径' : '文件绝对路径')"
      :disabled="disabled"
      spellcheck="false"
      @input="emit('update:modelValue', ($event.target as HTMLInputElement).value)"
    />
    <button
      type="button"
      class="btn btn--ghost btn--icon"
      :disabled="disabled || !canPick"
      :title="canPick ? '浏览…' : '仅桌面窗口中可使用系统选择对话框'"
      @click="browse"
    >
      <AppIcon :name="kind === 'directory' ? 'folder' : 'file'" :size="16" />
    </button>
  </div>
</template>

<style scoped>
.path {
  display: flex;
  gap: 8px;
}
.path .input {
  flex: 1;
  font-size: 12px;
}
</style>
