<script setup lang="ts">
/**
 * Path field: native dialog when running inside pywebview, manual input
 * otherwise. The returned path is still validated by the backend.
 */
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import AppIcon from './AppIcon.vue'
import { bridgeReady, hasNativeBridge, pickPath } from '@/api/bridge'
import { toastError } from '@/features/toast'

const props = defineProps<{
  modelValue: string
  kind: 'directory' | 'file' | 'file_or_directory'
  placeholder?: string
  dialogTitle?: string
  fileTypes?: string[]
  invalid?: boolean
  disabled?: boolean
  /** Accessible name for the text field and its browse button. */
  ariaLabel?: string
  /** Id of the element that explains an invalid value. */
  ariaDescribedby?: string
}>()
const emit = defineEmits<{ 'update:modelValue': [string] }>()

const canPick = ref(false)
const picking = ref(false)
const updateBridgeState = () => {
  canPick.value = hasNativeBridge()
}

/* Each field remembers its own last folder: the native dialog only knows one global
   "last used" directory, which made every browse button jump to the same place. */
const STORE_PREFIX = 'lp:path-start:'
const memoryKey = computed(() => `${STORE_PREFIX}${props.dialogTitle ?? props.kind}`)

function parentOf(value: string): string {
  const trimmed = value.trim().replace(/[\\/]+$/, '')
  const cut = Math.max(trimmed.lastIndexOf('\\'), trimmed.lastIndexOf('/'))
  return cut > 0 ? trimmed.slice(0, cut) : trimmed
}

function startDirectory(kind: 'file' | 'directory'): string | undefined {
  try {
    const remembered = localStorage.getItem(memoryKey.value)
    if (remembered) return remembered
  } catch {
    // Storage disabled: fall back to the current value below.
  }
  const current = props.modelValue.trim()
  if (!current) return undefined
  return kind === 'directory' ? current : parentOf(current)
}

function rememberDirectory(picked: string, kind: 'file' | 'directory'): void {
  try {
    localStorage.setItem(memoryKey.value, kind === 'directory' ? picked : parentOf(picked))
  } catch {
    // Remembering is a convenience; a full quota must not break browsing.
  }
}

onMounted(() => {
  updateBridgeState()
  window.addEventListener('pywebviewready', updateBridgeState)
  void bridgeReady().then(updateBridgeState)
})
onBeforeUnmount(() => {
  window.removeEventListener('pywebviewready', updateBridgeState)
})

async function browse(kind: 'file' | 'directory' = props.kind === 'directory' ? 'directory' : 'file') {
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
      kind,
      props.dialogTitle ?? (kind === 'directory' ? '\u9009\u62e9\u76ee\u5f55' : '\u9009\u62e9\u6587\u4ef6'),
      props.fileTypes,
      startDirectory(kind),
    )
    if (p) {
      emit('update:modelValue', p)
      rememberDirectory(p, kind)
    }
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
      :aria-label="ariaLabel"
      :aria-invalid="invalid || undefined"
      :aria-describedby="ariaDescribedby"
      spellcheck="false"
      @input="emit('update:modelValue', ($event.target as HTMLInputElement).value)"
    />
    <button
      type="button"
      class="btn btn--ghost btn--icon"
      :disabled="disabled || picking"
      :title="canPick ? '浏览…' : '仅桌面窗口中可使用系统选择对话框'"
      :aria-label="canPick ? '浏览…' : '仅桌面窗口中可使用系统选择对话框'"
      @click="browse()"
    >
      <AppIcon :name="kind === 'directory' ? 'folder' : 'file'" :size="16" />
    </button>
    <button v-if="kind === 'file_or_directory'" type="button" class="btn btn--ghost btn--icon"
      :disabled="disabled || picking" title="浏览模型目录…" aria-label="浏览模型目录…" @click="browse('directory')">
      <AppIcon name="folder" :size="16" />
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
  min-width: 0;
  font-size: 12px;
}
</style>
