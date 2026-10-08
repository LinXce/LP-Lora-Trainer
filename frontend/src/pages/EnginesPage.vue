<script setup lang="ts">
/**
 * Engine management: installation instances discovered under engine/.
 * Discovery ≠ execution; source present ≠ environment ready. The page shows
 * exactly what the backend reports and never upgrades a state on its own.
 */
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import StatusTag from '@/components/StatusTag.vue'
import EmptyState from '@/components/EmptyState.vue'
import ModalDialog from '@/components/ModalDialog.vue'
import PathInput from '@/components/PathInput.vue'
import SegTabs from '@/components/SegTabs.vue'
import { api } from '@/api'
import { refreshEngines, store } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import { ENGINE_IDS, candidateFallback, engineName, fmtRelative, installStateMeta, shortCommit, verificationMeta } from '@/features/format'
import type { EngineInstallation, InstallEnvironmentOptions } from '@/types/api'

const displayEngineId = (e: EngineInstallation): string | null => candidateFallback(e)

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
    const k = displayEngineId(e) ?? '__unknown'
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

/* Engine-owned installation, explicit consent before executing copied code. */
const router = useRouter()
const installDialog = ref(false)
const torchSource = ref<InstallEnvironmentOptions['torch_source']>('cu124')
const installConfirmed = ref(false)
const packageMirror = ref<'official' | 'tuna' | 'aliyun' | 'custom'>('official')
const customMirrorUrl = ref('')
const mirrorUrl = computed(() => ({
  official: null,
  tuna: 'https://pypi.tuna.tsinghua.edu.cn/simple',
  aliyun: 'https://mirrors.aliyun.com/pypi/simple',
  custom: customMirrorUrl.value.trim() || null,
}[packageMirror.value]))
const installSources = computed(() => selected.value?.installation_sources ?? [])
const cudaLabel = (source: string) => `CUDA ${source.slice(2, -1)}.${source.slice(-1)}（Musubi 官方 extra）`
function openInstall() {
  torchSource.value = installSources.value[0] ?? (selected.value?.engine_id === 'musubi_tuner' ? 'existing' : 'cu124')
  installConfirmed.value = false
  packageMirror.value = 'official'
  customMirrorUrl.value = ''
  installDialog.value = true
}
async function installEnvironment() {
  const e = selected.value
  if (!e || !installConfirmed.value) return
  if (packageMirror.value === 'custom' && !customMirrorUrl.value.trim()) {
    toast('请输入自定义镜像 URL', 'danger')
    return
  }
  busy.value = '安装引擎环境'
  try {
    const session = await api.engines.installEnvironment(e.installation_id, {
      confirmed: true,
      torch_source: torchSource.value,
      mirror_url: mirrorUrl.value,
    })
    installDialog.value = false
    toast('已启动引擎环境安装，输出显示在终端', 'ok')
    await router.push({ name: 'terminal', query: { session: session.session_id } })
    await refreshEngines()
  } catch (err) {
    toastError(err, '安装引擎环境')
  } finally {
    busy.value = null
  }
}

/* Bind interpreter dialog */
const pyDialog = ref(false)
const pyPath = ref('')
function openPython() {
  pyPath.value = selected.value?.python_candidates?.[0] ?? selected.value?.python_executable ?? ''
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
  typeChoice.value = e?.candidate_engines[0] ?? ENGINE_IDS[0]
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
          text="将 Kohya、AI Toolkit 或 Musubi Tuner 源码目录复制到 engine/ 下的任意子目录，然后点击“刷新发现”。识别只读取文件，不会执行引擎代码。"
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
          <span class="eyebrow">{{ engineName(displayEngineId(selected)) }}</span>
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
          <button class="btn btn--sm" :disabled="busy !== null || !selected.engine_id || selected.state === 'preparing' || selected.state === 'missing'" @click="openInstall">
            <AppIcon name="terminal" :size="14" />
            安装引擎环境
          </button>
          <button class="btn btn--sm" :disabled="busy !== null || !selected.python_executable || selected.state === 'preparing'" @click="diagnose(selected)">
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
          <li>确认源码可信后，通过“安装引擎环境”调用该引擎自身的安装流程</li>
          <li>在终端查看安装输出；诊断通过后，按适配器已支持的训练能力使用</li>
        </ol>
      </section>
    </aside>

    <ModalDialog v-if="installDialog" title="安装引擎环境" @close="installDialog = false">
      <p class="muted">将调用 {{ engineName(selected?.engine_id ?? '') }} 的安装流程，不会统一猜测依赖。训练依赖只安装到当前引擎目录内的独立环境，不安装进应用运行时或系统 Python。</p>
      <p class="field__hint">LP 会调用 {{ engineName(selected?.engine_id ?? '') }} 自身提供的安装入口。Kohya 由项目内 uv 准备独立 Python 后调用官方安装器；Musubi 使用官方 uv sync；AI Toolkit 由其 manager 自行创建环境，不使用系统 Python 或应用运行时安装训练依赖。</p>
      <label v-if="selected?.engine_id === 'musubi_tuner'" class="field">
        <span class="field__label">PyTorch CUDA wheel 来源</span>
        <select v-model="torchSource" class="input">
          <option v-for="source in installSources" :key="source" :value="source">{{ cudaLabel(source) }}</option>
          <option value="existing">不指定 CUDA extra（按项目默认依赖同步，可能移除未声明的包）</option>
        </select>
        <span class="field__hint">仅显示 Musubi Tuner 在当前 pyproject.toml 中声明的安装配置；LP 会原样调用其 uv 项目流程。</span>
      </label>
      <p v-else-if="selected?.engine_id === 'ai_toolkit'" class="field__hint">AI Toolkit 由官方 manager 自动检测硬件、选择 PyTorch、创建环境并同步依赖；LP 不替它指定 CUDA wheel。</p>
      <p v-else class="field__hint">Kohya 调用 setup/setup_windows.py --headless，由官方安装器初始化 sd-scripts 子模块并选择 CUDA 依赖。</p>
      <label class="field">
        <span class="field__label">Python / uv 镜像源</span>
        <select v-model="packageMirror" class="input">
          <option value="official">官方 PyPI 源</option>
          <option value="tuna">清华大学镜像</option>
          <option value="aliyun">阿里云镜像</option>
          <option value="custom">自定义镜像 URL</option>
        </select>
        <input
          v-if="packageMirror === 'custom'"
          v-model="customMirrorUrl"
          class="input"
          type="url"
          placeholder="https://example.com/simple"
        />
        <span class="field__hint">仅设置依赖包索引，会传递给引擎自己的 pip/uv 安装流程；Musubi 声明的 PyTorch CUDA 专用源仍由项目配置管理。</span>
      </label>
      <label class="row" style="gap: 10px; margin-top: 16px">
        <input v-model="installConfirmed" type="checkbox" />
        <span>我信任这份引擎源码，并允许其安装流程联网下载依赖及修改所选环境。</span>
      </label>
      <template #footer>
        <button class="btn btn--ghost" @click="installDialog = false">取消</button>
        <button class="btn btn--primary" :disabled="!installConfirmed || busy !== null" @click="installEnvironment">安装并打开终端</button>
      </template>
    </ModalDialog>

    <ModalDialog v-if="pyDialog" title="选择 Python 解释器" @close="pyDialog = false">
      <p class="muted">
        选择该安装实例使用的 python.exe。复制来的 .venv 不会被默认视为可用，后端会核实解释器、路径绑定和必要依赖。
      </p>
      <label class="field">
        <span class="field__label">解释器路径</span>
        <PathInput v-model="pyPath" kind="file" dialog-title="选择 python.exe" :file-types="['Python executable (*.exe)', 'All files (*.*)']" />
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
          v-for="id in ENGINE_IDS"
          :key="id"
          class="type-opt"
          :class="{ 'is-on': typeChoice === id }"
          @click="typeChoice = id"
        >
          <strong>{{ engineName(id) }}</strong>
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
