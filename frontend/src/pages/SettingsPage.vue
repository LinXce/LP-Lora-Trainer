<script setup lang="ts">
import { computed, ref } from 'vue'
import { onBeforeRouteLeave, useRouter } from 'vue-router'
import PathInput from '@/components/PathInput.vue'
import ModalDialog from '@/components/ModalDialog.vue'
import AppIcon from '@/components/AppIcon.vue'
import { api } from '@/api'
import { store } from '@/features/store'
import { ACTIVE_TASK_STATES } from '@/features/format'
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

/** The URL input is always a string; an unset proxy is the empty string. */
const proxyUrlInput = computed({
  get: () => form.value?.proxy_url ?? '',
  set: (value: string) => {
    if (form.value) form.value.proxy_url = value.trim() || null
  },
})

// The backend refuses to exit while training is active or an orphan is
// unverified, so the entry point is disabled under exactly the same conditions.
const blockedTasks = computed(() =>
  store.tasks.filter((t) => ACTIVE_TASK_STATES.includes(t.state) || t.state === 'connection_lost'),
)
const shuttingDown = ref(false)
const shutdownOpen = ref(false)

/* uv keeps its download cache under runtime/uv/cache; clearing it only means the
   next engine environment install downloads again. */
const uvCache = ref<{ path: string; exists: boolean; count: number; bytes: number } | null>(null)
const uvClearOpen = ref(false)
async function loadUvCache() {
  try {
    uvCache.value = await api.system.uvCache()
  } catch {
    uvCache.value = null
  }
}
async function clearUvCache() {
  try {
    const result = await api.system.clearUvCache()
    uvClearOpen.value = false
    toast(`已清除 uv 缓存，释放 ${(result.bytes / 1048576).toFixed(1)} MB`, 'ok')
    await loadUvCache()
  } catch (err) {
    toastError(err, '清除 uv 缓存')
  }
}
void loadUvCache()

async function shutdown() {
  shuttingDown.value = true
  try {
    await api.system.shutdown()
    shutdownOpen.value = false
    toast('已请求优雅退出：后端与监管进程将结束，训练队列不再接收新任务。', 'ok')
  } catch (err) {
    toastError(err, '退出后端')
  } finally {
    shuttingDown.value = false
  }
}

async function save() {
  if (!form.value) return false
  if (form.value.proxy_enabled && !form.value.proxy_url?.trim()) {
    toast('启用代理后需要填写代理地址', 'danger')
    return false
  }
  saving.value = true
  try {
    const s = await api.settings.save({
      ...form.value,
      comfyui_lora_dir: form.value.comfyui_lora_dir?.trim() || null,
      proxy_url: form.value.proxy_url?.trim() || null,
    })
    saved.value = s
    form.value = { ...s }
    toast('设置已保存', 'ok')
    return true
  } catch (err) {
    toastError(err, '保存设置')
    return false
  } finally {
    saving.value = false
  }
}

/* Leaving with unsaved edits asks first instead of silently dropping them. */
const router = useRouter()
const unsavedOpen = ref(false)
let pendingPath: string | null = null
let leaving = false

onBeforeRouteLeave((to) => {
  if (leaving || !dirty.value) return true
  pendingPath = to.fullPath
  unsavedOpen.value = true
  return false
})

function stayOnPage() {
  unsavedOpen.value = false
  pendingPath = null
}

function discardAndLeave() {
  const path = pendingPath
  unsavedOpen.value = false
  pendingPath = null
  if (!path) return
  leaving = true
  void router.push(path).finally(() => { leaving = false })
}

async function saveAndLeave() {
  if (await save()) discardAndLeave()
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
            <PathInput v-model="form.engine_root" kind="directory" aria-label="引擎目录 engine_root" />
          </div>
          <div class="set">
            <div class="set__info">
              <strong>数据目录 data_root</strong>
              <p class="muted">环境、清单、任务记录、缓存与 state.sqlite。模型与数据集按外部路径引用，不复制到这里。</p>
            </div>
            <PathInput v-model="form.data_root" kind="directory" aria-label="数据目录 data_root" />
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
              aria-label="ComfyUI LoRA 目录"
              @update:model-value="form.comfyui_lora_dir = $event"
            />
          </div>
          <div class="set set--inline">
            <div class="set__info">
              <strong>GPU 监控</strong>
              <p class="muted">单一低频采集器；关闭后界面显示“未知”。</p>
            </div>
            <button class="switch" role="switch" aria-label="GPU 监控" :aria-checked="form.gpu_monitor" @click="form.gpu_monitor = !form.gpu_monitor" />
          </div>
          <div class="set set--inline">
            <div class="set__info">
              <strong>日志显示行数</strong>
              <p class="muted">界面只加载日志尾部；完整日志始终保存在任务目录中。</p>
            </div>
            <input v-model.number="form.log_tail_lines" class="input num" type="number" min="100" max="5000" step="100" aria-label="日志显示行数" style="width: 120px" />
          </div>
          <div class="set set--inline">
            <div class="set__info">
              <strong>网络代理</strong>
              <p class="muted">引擎环境安装与 Git 拉取默认走此代理。关闭时沿用系统环境变量里的代理；每个下载对话框仍可单独覆盖。</p>
            </div>
            <button class="switch" role="switch" aria-label="启用网络代理" :aria-checked="form.proxy_enabled" @click="form.proxy_enabled = !form.proxy_enabled" />
          </div>
          <div class="set">
            <div class="set__info">
              <strong>代理地址</strong>
              <p class="muted">形如 <code>http://主机:端口</code>，支持 <code>账号:密码@</code>。仅保存在本机 state.sqlite，不会写入安装日志或会话记录（日志只记录是否启用）。</p>
            </div>
            <input
              v-model="proxyUrlInput"
              class="input mono"
              type="url"
              spellcheck="false"
              placeholder="http://127.0.0.1:7890"
              aria-label="代理地址"
            />
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
        <button
          class="btn btn--danger btn--sm"
          :disabled="blockedTasks.length > 0 || shuttingDown"
          @click="shutdownOpen = true"
        >
          退出后端与监督
        </button>
        <button
          class="btn btn--sm"
          :disabled="!uvCache?.exists"
          :title="uvCache?.path ? `清除 ${uvCache.path}` : 'uv 缓存目录'"
          @click="uvClearOpen = true"
        >
          <AppIcon name="trash" :size="14" />
          清除 uv 缓存
        </button>
        <p class="field__hint">
          {{ uvCache?.exists
            ? `runtime/uv/cache：${uvCache.count} 个文件，${(uvCache.bytes / 1048576).toFixed(0)} MB；清除后下次安装引擎环境会重新下载`
            : '暂无 uv 缓存（安装引擎环境时会自动生成）' }}
        </p>
        <p v-if="blockedTasks.length" class="muted" style="font-size: 12px">
          存在 {{ blockedTasks.length }} 个活动或需人工核实的任务，退出已被禁止。
        </p>
        <p v-else class="muted" style="font-size: 12px">
          仅在没有任何活动任务时可用；退出会同时结束后端与监管进程，不会中断训练。
        </p>
      </section>
      <section class="panel panel--pad stack">
        <span class="label-pill" style="align-self: flex-start">说明</span>
        <p class="muted" style="font-size: 12px">
          关闭窗口不会停止训练。训练在独立监管进程中运行，再次启动应用会自动重新连接。停止训练请在“训练任务”中操作。
        </p>
      </section>
    </aside>

    <ModalDialog v-if="uvClearOpen" title="清除 uv 缓存？" :width="470" @close="uvClearOpen = false">
      <p class="mono selectable">{{ uvCache?.path }}</p>
      <p class="muted">
        共 {{ uvCache?.count ?? 0 }} 个文件、{{ ((uvCache?.bytes ?? 0) / 1048576).toFixed(0) }} MB。<strong>只删除这个下载缓存目录</strong>：已装好的引擎环境和应用运行都不受影响，只是下次安装引擎环境需要重新下载依赖。
      </p>
      <template #footer>
        <button class="btn btn--ghost" @click="uvClearOpen = false">取消</button>
        <button class="btn btn--danger" @click="clearUvCache">清除缓存</button>
      </template>
    </ModalDialog>

    <ModalDialog v-if="shutdownOpen" title="退出后端与监管进程？" @close="shutdownOpen = false">
      <p>将优雅结束后端 API 与独立监管进程（释放 supervisor.lock），并停止接收新的训练任务。</p>
      <p class="warnbox">
        <AppIcon name="alert" :size="14" />
        当前没有任何活动任务，退出不会中断训练。下次启动应用会自动重新连接并拉起监管进程。
      </p>
      <template #footer>
        <button class="btn btn--ghost" @click="shutdownOpen = false">取消</button>
        <button class="btn btn--danger" :disabled="shuttingDown" @click="shutdown">确认退出</button>
      </template>
    </ModalDialog>

    <ModalDialog v-if="unsavedOpen" title="设置有未保存的修改" @close="stayOnPage">
      <p>离开「设置」会丢弃尚未保存的修改。</p>
      <p class="warnbox">
        <AppIcon name="alert" :size="14" />
        未保存的改动不会生效：目录、代理等设置需要保存后才会应用到引擎安装与 Git 拉取。
      </p>
      <template #footer>
        <button class="btn btn--ghost" @click="stayOnPage">留在本页</button>
        <span class="spacer" />
        <button class="btn btn--danger" @click="discardAndLeave">放弃更改并离开</button>
        <button class="btn btn--primary" :disabled="saving" @click="saveAndLeave">{{ saving ? '正在保存…' : '保存并离开' }}</button>
      </template>
    </ModalDialog>
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
.warnbox {
  display: flex;
  gap: 8px;
  padding: 10px 12px;
  border-radius: var(--radius-sm);
  background: var(--warn-bg);
  color: var(--warn);
  font-size: 12px;
}
.warnbox .icon {
  margin-top: 2px;
}
</style>
