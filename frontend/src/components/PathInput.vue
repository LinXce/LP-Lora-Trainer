<script setup lang="ts">
/**
 * Path field: native dialog when running inside pywebview, manual input
 * otherwise. The returned path is still validated by the backend.
 */
import { onBeforeUnmount, onMounted, ref } from 'vue'
import AppIcon from './AppIcon.vue'
import { bridgeReady, hasNativeBridge, pickPath } from '@/api/bridge'
import { toastError } from '@/features/toast'

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
const picking = ref(false)
const updateBridgeState = () => {
  canPick.value = hasNativeBridge()
}

onMounted(() => {
  updateBridgeState()
  window.addEventListener('pywebviewready', updateBridgeState)
  void bridgeReady().then(updateBridgeState)
})
onBeforeUnmount(() => {
  window.removeEventListener('pywebviewready', updateBridgeState)
})

async function browse() {
  if (picking.value) return
  picking.value = true
  try {
    if (!hasNativeBridge()) {
      canPick.value = await bridgeReady(3000)
    }
    if (!canPick.value) {
      throw new Error('\u8bf7\u5728\u684c\u9762\u7a97\u53e3\u4e2d\u4f7f\u7528\u8def\u5f84\u6d4f\u89c8\u6309\u94ae')
    }
    const p = await pickPath(
      props.kind,
      props.dialogTitle ?? (props.kind === 'directory' ? '\u9009\u62e9\u76ee\u5f55' : '\u9009\u62e9\u6587\u4ef6'),
      props.fileTypes,
    )
    if (p) emit('update:modelValue', p)
  } catch (err) {
    toastError(err, '\u6d4f\u89c8\u8def\u5f84')
  } finally {
    picking.value = false
  }
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
      :disabled="disabled || picking"
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
