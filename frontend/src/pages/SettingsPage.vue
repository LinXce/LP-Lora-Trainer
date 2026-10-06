<script setup lang="ts">
import { computed, ref } from 'vue'
import PathInput from '@/components/PathInput.vue'
import { api } from '@/api'
import { store } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import type { AppSettings } from '@/types/api'

const saved = ref<AppSettings | null>(null)
const form = ref<AppSettings | null>(null)
const error = ref<string | null>(null)

api.settings
  .get()
  .then((s) => {
    saved.value = s
    form.value = { ...s }
  })
  .catch((err) => (error.value = (err as Error).message))

const dirty = computed(() => JSON.stringify(form.value) !== JSON.stringify(saved.value))
const saving = ref(false)

async function save() {
  if (!form.value) return
  saving.value = true
  try {
    const s = await api.settings.save({ ...form.value, comfyui_lora_dir: form.value.comfyui_lora_dir?.trim() || null })
    saved.value = s
    form.value = { ...s }
    toast('设置已保存', 'ok')
  } catch (err) {
    toastError(err, '保存设置')
  } finally {
    saving.value = false
  }
}
</script>

<template>
  <div class="page">
    <section class="page__main panel">
      <header class="page__header">
        <h1 class="title-lg">设置</h1>
        <span class="spacer" />
        <button class="btn btn--ghost" :disabled="!dirty" @click="form = saved && { ...saved }">还原</button>
        <button class="btn btn--primary" :disabled="!dirty || saving" @click="save">保存</button>
      </header>

      <div class="page__body">
        <p v-if="error" class="field__error">无法读取设置：{{ error }}</p>
        <div v-else-if="form" class="sets">
          <div class="set">
            <div class="set__info">
              <strong>引擎目录 engine_root</strong>
              <p class="muted">复制进来的引擎安装实例所在目录，可迁移到其他磁盘。应用更新不会修改或删除其中内容。</p>
            </div>
            <PathInput v-model="form.engine_root" kind="directory" />
          </div>
          <div class="set">
            <div class="set__info">
              <strong>数据目录 data_root</strong>
              <p class="muted">环境、清单、任务记录、缓存与 state.sqlite。模型与数据集按外部路径引用，不复制到这里。</p>
            </div>
            <PathInput v-model="form.data_root" kind="directory" />
          </div>
          <div class="set">
            <div class="set__info">
              <strong>ComfyUI LoRA 目录（可选）</strong>
              <p class="muted">仅作为“发布”时的默认目标。训练器不依赖 ComfyUI，也不会调用其接口。</p>
            </div>
            <PathInput
              :model-value="form.comfyui_lora_dir ?? ''"
              kind="directory"
              placeholder="未配置"
              @update:model-value="form.comfyui_lora_dir = $event"
            />
          </div>
          <div class="set set--inline">
            <div class="set__info">
              <strong>GPU 监控</strong>
              <p class="muted">单一低频采集器；关闭后界面显示“未知”。</p>
            </div>
            <button class="switch" role="switch" :aria-checked="form.gpu_monitor" @click="form.gpu_monitor = !form.gpu_monitor" />
          </div>
          <div class="set set--inline">
            <div class="set__info">
              <strong>日志显示行数</strong>
              <p class="muted">界面只加载日志尾部；完整日志始终保存在任务目录中。</p>
            </div>
            <input v-model.number="form.log_tail_lines" class="input num" type="number" min="100" max="5000" step="100" style="width: 120px" />
          </div>
        </div>
      </div>
    </section>

    <aside class="page__side">
      <section class="panel panel--pad stack">
        <span class="label-pill" style="align-self: flex-start">关于</span>
        <dl class="kv selectable">
          <dt>应用</dt>
          <dd>LP LoRA Trainer</dd>
          <dt>后端</dt>
          <dd>{{ store.system?.backend_version ?? '未知' }}</dd>
          <dt>监管进程</dt>
          <dd>{{ { running: '运行中', idle: '空闲', unreachable: '无法连接' }[store.system?.supervisor ?? 'unreachable'] }}</dd>
          <dt>GPU</dt>
          <dd>{{ store.system?.gpu?.name ?? '未知' }}</dd>
        </dl>
      </section>
      <section class="panel panel--pad stack">
        <span class="label-pill" style="align-self: flex-start">说明</span>
        <p class="muted" style="font-size: 12px">
          关闭窗口不会停止训练。训练在独立监管进程中运行，再次启动应用会自动重新连接。停止训练请在“训练任务”中操作。
        </p>
      </section>
    </aside>
  </div>
</template>

<style scoped>
.page__header,
.page__body {
  padding-left: 44px;
}
.sets {
  display: flex;
  flex-direction: column;
  max-width: 760px;
}
.set {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 18px 0;
  border-bottom: 1px solid var(--line);
}
.set:last-child {
  border-bottom: 0;
}
.set--inline {
  flex-direction: row;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
}
.set__info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.set__info p {
  font-size: 12px;
}
</style>
