<script setup lang="ts">
/**
 * Engine management: installation instances discovered under engine/.
 * Discovery ≠ execution; source present ≠ environment ready. The page shows
 * exactly what the backend reports and never upgrades a state on its own.
 */
import { computed, ref, watch } from 'vue'
import AppIcon from '@/components/AppIcon.vue'
import StatusTag from '@/components/StatusTag.vue'
import EmptyState from '@/components/EmptyState.vue'
import ModalDialog from '@/components/ModalDialog.vue'
import PathInput from '@/components/PathInput.vue'
import SegTabs from '@/components/SegTabs.vue'
import { api } from '@/api'
import { refreshEngines, store } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import { engineName, engineNames, fmtRelative, installStateMeta, shortCommit, verificationMeta } from '@/features/format'
import type { EngineInstallation } from '@/types/api'

type Filter = 'all' | 'ready' | 'attention'
const filter = ref<Filter>('all')

const filtered = computed(() => {
  const list = store.engines
  if (filter.value === 'ready') return list.filter((e) => e.state === 'ready')
  if (filter.value === 'attention') return list.filter((e) => e.state !== 'ready' || !e.engine_id)
  return list
})

const groups = computed(() => {
  const map = new Map<string, EngineInstallation[]>()
  for (const e of filtered.value) {
    const k = e.engine_id ?? '__unknown'
    map.set(k, [...(map.get(k) ?? []), e])
  }
  return [...map.entries()].map(([k, items]) => ({
    key: k,
    title: k === '__unknown' ? '待确认类型' : engineName(k),
    items,
  }))
})

const selectedId = ref<string | null>(null)
watch(
  () => store.engines,
  (list) => {
    if (!list.find((e) => e.installation_id === selectedId.value)) selectedId.value = list[0]?.installation_id ?? null
  },
  { immediate: true },
)
const selected = computed(() => store.engines.find((e) => e.installation_id === selectedId.value) ?? null)

const busy = ref<string | null>(null)
async function run(label: string, fn: () => Promise<unknown>, okText: string) {
  busy.value = label
  try {
    await fn()
    toast(okText, 'ok')
    await refreshEngines()
  } catch (err) {
    toastError(err, label)
  } finally {
    busy.value = null
  }
}

const rescan = () => run('刷新发现', () => api.engines.rescan(), '已重新扫描 engine/ 目录')
const setDefault = (e: EngineInstallation) =>
  run('设为默认', () => api.engines.setDefault(e.installation_id), `已将 ${e.label} 设为默认（不影响已提交任务）`)
const diagnose = (e: EngineInstallation) => run('环境诊断', () => api.engines.diagnose(e.installation_id), '已提交诊断，结果会自动更新')

/* Bind interpreter dialog */
const pyDialog = ref(false)
const pyPath = ref('')
function openPython() {
  pyPath.value = selected.value?.python_executable ?? ''
  pyDialog.value = true
}
async function savePython() {
  const e = selected.value
  if (!e || !pyPath.value.trim()) return
  await run('绑定解释器', () => api.engines.bindPython(e.installation_id, pyPath.value.trim()), '已提交，后端将核实解释器与依赖')
  pyDialog.value = false
}

/* Confirm engine type dialog */
const typeDialog = ref(false)
const typeChoice = ref('')
function openType() {
  const e = selected.value
  typeChoice.value = e?.candidate_engines[0] ?? Object.keys(engineNames)[0]
  typeDialog.value = true
}
async function saveType() {
  const e = selected.value
  if (!e || !typeChoice.value) return
  await run('确认引擎类型', () => api.engines.confirmType(e.installation_id, typeChoice.value), '已登记引擎类型')
  typeDialog.value = false
}

const modeLabel: Record<EngineInstallation['management_mode'], string> = {
  user_managed: '用户管理源码',
  application_managed: '应用托管',
  external_reference: '外部引用',
}

const counts = computed(() => ({
  all: store.engines.length,
  ready: store.engines.filter((e) => e.state === 'ready').length,
  attention: store.engines.filter((e) => e.state !== 'ready' || !e.engine_id).length,
}))
</script>

<template>
  <div class="page">
    <section class="page__main panel">
      <header class="page__header">
        <div class="stack" style="gap: 4px">
          <h1 class="title-lg">引擎管理</h1>
          <p class="muted">engine/ 下每个一级目录是一个安装实例；目录名只是标签，不代表引擎类型或版本。</p>
        </div>
        <span class="spacer" />
        <button class="btn" :disabled="busy !== null" @click="rescan">
          <AppIcon name="refresh" :size="15" />
          刷新发现
        </button>
      </header>

      <div class="page__body">
        <SegTabs
          v-model="filter"
          :options="[
            { value: 'all', label: '全部', count: counts.all },
            { value: 'ready', label: '就绪', count: counts.ready },
            { value: 'attention', label: '待处理', count: counts.attention },
          ]"
          style="margin-bottom: 16px"
        />

        <EmptyState
          v-if="store.enginesLoaded && !store.engines.length"
          icon="cpu"
          title="未发现任何引擎安装实例"
          text="将完整的 Kohya（sd-scripts）或 AI Toolkit 目录复制到 engine/ 下的任意子目录，然后点击“刷新发现”。识别只读取文件，不会执行引擎代码。"
        >
          <button class="btn btn--primary" @click="rescan">刷新发现</button>
        </EmptyState>

        <div v-for="g in groups" :key="g.key" class="group">
          <div class="group__head">
            <span class="label-pill">{{ g.title }}</span>
            <span class="muted num">{{ g.items.length }} 个实例</span>
          </div>
          <div class="inst-list">
            <button
              v-for="e in g.items"
              :key="e.installation_id"
              class="inst"
              :class="{ 'is-selected': e.installation_id === selectedId }"
              @click="selectedId = e.installation_id"
            >
              <span class="inst__knob" :class="`is-${installStateMeta[e.state].tone}`" />
              <span class="inst__main">
                <span class="inst__title">
                  <strong class="truncate">{{ e.label }}</strong>
                  <span v-if="e.is_default" class="inst__default"><AppIcon name="pin" :size="11" />默认</span>
                </span>
                <span class="inst__path mono truncate">{{ e.source_path }}</span>
              </span>
              <span class="inst__rev mono">
                {{ shortCommit(e.revision.commit) ?? (e.revision.fingerprint ? '指纹' : '—') }}
              </span>
              <StatusTag v-bind="installStateMeta[e.state]" />
              <StatusTag v-bind="verificationMeta[e.verification]" :dot="false" />
            </button>
          </div>
        </div>
      </div>
    </section>

    <aside class="page__side">
      <section v-if="selected" class="panel panel--pad detail stack enter" :key="selected.installation_id">
        <div class="row">
          <span class="eyebrow">{{ engineName(selected.engine_id) }}</span>
          <span class="spacer" />
          <StatusTag v-bind="installStateMeta[selected.state]" />
        </div>
        <h2 class="title-md selectable">{{ selected.label }}</h2>

        <div v-if="selected.issues.length" class="issues">
          <div v-for="i in selected.issues" :key="i" class="issue">
            <AppIcon name="alert" :size="13" />
            <span>{{ i }}</span>
          </div>
        </div>

        <dl class="kv selectable">
          <dt>管理方式</dt>
          <dd>{{ modeLabel[selected.management_mode] }}</dd>
          <dt>代码版本</dt>
          <dd class="mono">
            <template v-if="selected.revision.commit">
              {{ selected.revision.requested_ref ?? '' }} {{ shortCommit(selected.revision.commit) }}
            </template>
            <template v-else-if="selected.revision.fingerprint">
              {{ selected.revision.fingerprint }}<br /><span class="muted">上游版本未核实</span>
            </template>
            <template v-else>未知</template>
          </dd>
          <dt>Python</dt>
          <dd>{{ selected.python_version ?? '未绑定' }}</dd>
          <dt>PyTorch</dt>
          <dd>{{ selected.torch_version ?? '未知' }}</dd>
          <dt>CUDA wheel</dt>
          <dd>{{ selected.cuda_wheel ?? '未知' }}</dd>
          <dt>兼容状态</dt>
          <dd>{{ verificationMeta[selected.verification].label }}</dd>
          <dt>任务引用</dt>
          <dd class="num">{{ selected.referenced_by }}</dd>
          <dt>发现于</dt>
          <dd>{{ fmtRelative(selected.discovered_at) }}</dd>
        </dl>

        <div v-if="selected.python_executable" class="pathbox mono selectable">{{ selected.python_executable }}</div>

        <hr class="divider" />

        <div class="actions">
          <button v-if="!selected.engine_id" class="btn btn--primary btn--sm" :disabled="busy !== null" @click="openType">
            确认引擎类型
          </button>
          <button class="btn btn--sm" :class="{ 'btn--primary': selected.engine_id && !selected.python_executable }" :disabled="busy !== null" @click="openPython">
            <AppIcon name="python" :size="14" />
            {{ selected.python_executable ? '更换解释器' : '选择解释器' }}
          </button>
          <button class="btn btn--sm" :disabled="busy !== null || !selected.python_executable" @click="diagnose(selected)">
            环境诊断
          </button>
          <button
            class="btn btn--sm"
            :disabled="busy !== null || selected.is_default || selected.state !== 'ready'"
            :title="selected.state !== 'ready' ? '环境就绪后才能设为默认' : ''"
            @click="setDefault(selected)"
          >
            设为默认
          </button>
        </div>
        <p class="field__hint">
          切换默认只影响之后新建的配置；已排队或运行中的任务仍使用其提交时固定的安装实例。
        </p>
      </section>
      <section v-else class="panel">
        <EmptyState icon="cpu" title="未选择实例" />
      </section>

      <section class="panel panel--pad stack note">
        <span class="label-pill">接入方式</span>
        <ol>
          <li>复制完整引擎目录到 <code>engine/&lt;任意名称&gt;/</code></li>
          <li>刷新发现，确认识别出的引擎类型</li>
          <li>选择已有 Python 解释器或创建独立环境</li>
          <li>诊断通过后即可用于训练</li>
        </ol>
      </section>
    </aside>

    <ModalDialog v-if="pyDialog" title="选择 Python 解释器" @close="pyDialog = false">
      <p class="muted">
        选择该安装实例使用的 python.exe。复制来的 .venv 不会被默认视为可用，后端会核实解释器、路径绑定和必要依赖。
      </p>
      <label class="field">
        <span class="field__label">解释器路径</span>
        <PathInput v-model="pyPath" kind="file" dialog-title="选择 python.exe" :file-types="['Python (python.exe)']" />
      </label>
      <template #footer>
        <button class="btn btn--ghost" @click="pyDialog = false">取消</button>
        <button class="btn btn--primary" :disabled="!pyPath.trim() || busy !== null" @click="savePython">提交核实</button>
      </template>
    </ModalDialog>

    <ModalDialog v-if="typeDialog" title="确认引擎类型" @close="typeDialog = false">
      <p class="muted">无法仅凭目录结构可靠识别时，由你确认引擎类型。不会根据文件夹名称自动绑定。</p>
      <div class="type-list">
        <button
          v-for="(name, id) in engineNames"
          :key="id"
          class="type-opt"
          :class="{ 'is-on': typeChoice === id }"
          @click="typeChoice = id"
        >
          <strong>{{ name }}</strong>
          <span v-if="selected?.candidate_engines.includes(id)" class="muted">适配器匹配</span>
        </button>
      </div>
      <template #footer>
        <button class="btn btn--ghost" @click="typeDialog = false">取消</button>
        <button class="btn btn--primary" :disabled="!typeChoice || busy !== null" @click="saveType">确认</button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
.page__header {
  padding-left: 44px;
}
.page__body {
  padding-left: 44px;
}
.group + .group {
  margin-top: 22px;
}
.group__head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.inst-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.inst {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) 90px auto auto;
  align-items: center;
  gap: 14px;
  padding: 12px 16px 12px 10px;
  border-radius: var(--radius-md);
  background: var(--panel-raised);
  text-align: left;
  transition: background 0.15s, box-shadow 0.15s;
}
.inst:hover {
  background: #4d4d4d;
}
.inst.is-selected {
  box-shadow: inset 0 0 0 1.5px var(--light);
}
.inst__knob {
  width: 18px;
  height: 30px;
  border-radius: 9px;
  border: 1px solid var(--line-strong);
  background: rgba(255, 255, 255, 0.06);
  position: relative;
}
.inst__knob::after {
  content: '';
  position: absolute;
  left: 50%;
  top: 6px;
  width: 6px;
  height: 6px;
  margin-left: -3px;
  border-radius: 50%;
  background: var(--text-3);
}
.inst__knob.is-ok::after {
  background: var(--ok);
}
.inst__knob.is-warn::after {
  background: var(--warn);
}
.inst__knob.is-danger::after {
  background: var(--danger);
}
.inst__knob.is-info::after {
  background: var(--info);
}
.inst__main {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}
.inst__title {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}
.inst__default {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  padding: 1px 8px;
  border-radius: var(--radius-pill);
  background: var(--light);
  color: var(--ink);
  font-size: 10px;
  font-weight: 700;
}
.inst__path {
  font-size: 11px;
  color: var(--text-3);
}
.inst__rev {
  color: var(--text-2);
}

.detail {
  gap: 14px;
}
.issues {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.issue {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 12px;
  border-radius: var(--radius-sm);
  background: var(--warn-bg);
  color: var(--warn);
  font-size: 12px;
}
.pathbox {
  padding: 8px 12px;
  border-radius: var(--radius-sm);
  background: var(--field);
  font-size: 11px;
  color: var(--text-2);
  overflow-wrap: anywhere;
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.note ol {
  margin: 0;
  padding-left: 18px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  color: var(--text-2);
  font-size: 12px;
}
.note code {
  padding: 1px 6px;
  border-radius: 6px;
  background: var(--field);
}
.type-list {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.type-opt {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  padding: 14px 16px;
  border-radius: var(--radius-md);
  border: 1px solid var(--line-strong);
  text-align: left;
}
.type-opt.is-on {
  background: var(--light);
  color: var(--ink);
  border-color: var(--light);
}
.type-opt.is-on .muted {
  color: var(--ink-soft);
}
</style>
