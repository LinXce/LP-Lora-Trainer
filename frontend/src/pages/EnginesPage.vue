<script setup lang="ts">
import SelectInput from '@/components/SelectInput.vue'
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
import { openTerminal } from '@/api/bridge'
import { refreshEngines, store } from '@/features/store'
import { toast, toastError } from '@/features/toast'
import { candidateFallback, engineCompatMeta, engineName, fmtRelative, installStateMeta, shortCommit, verificationMeta } from '@/features/format'
import type { EngineInstallation, InstallOption, InstallPlan } from '@/types/api'

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
  return [...map.entries()]
    .map(([k, items]) => ({
      key: k,
      title: k === '__unknown' ? '待确认类型' : engineName(k),
      items: [...items].sort((a, b) => (a.label || '').localeCompare(b.label || '', 'zh-Hans-CN', { numeric: true })),
    }))
    // Alphabetical by engine name; the unknown group always sinks to the bottom.
    .sort((a, b) => (a.key === '__unknown' ? 1 : 0) - (b.key === '__unknown' ? 1 : 0)
      || a.title.localeCompare(b.title, 'zh-Hans-CN', { numeric: true }))
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
    if (!await refreshEngines()) return false
    toast(okText, 'ok')
    return true
  } catch (err) {
    toastError(err, label)
    return false
  } finally {
    busy.value = null
  }
}

const rescan = () => run('刷新发现', () => api.engines.rescan(), '已重新扫描 engine/ 目录')
const diagnose = (e: EngineInstallation) => run('环境诊断', () => api.engines.diagnose(e.installation_id), '已提交诊断，结果会自动更新')

/* Forget one entry whose folder is gone (or is no longer wanted). Files are never touched. */
const removeDialog = ref(false)
async function removeEngine() {
  const target = selected.value
  if (!target) return
  busy.value = '删除引擎条目'
  try {
    await api.engines.remove(target.installation_id)
    removeDialog.value = false
    toast('已删除该引擎条目（engine/ 下的文件未改动）', 'ok')
    await refreshEngines()
    selectedId.value = store.engines[0]?.installation_id ?? null
  } catch (err) {
    toastError(err, '删除引擎条目')
  } finally {
    busy.value = null
  }
}

/* Manual dependency installs: LP only opens the console, the user types the commands. */
const terminalDialog = ref(false)
const terminalPython = ref('')
const terminalTarget = ref<EngineInstallation | null>(null)
function openTerminalDialog(e: EngineInstallation) {
  terminalPython.value = e.python_executable ?? ''
  terminalTarget.value = e
  terminalDialog.value = true
}
async function confirmTerminal() {
  try {
    const target = terminalTarget.value
    const python = terminalPython.value.trim()
    let opened = false
    try {
      opened = await openTerminal(python)
    } catch (nativeError) {
      // A present-but-broken pywebview bridge must not hide the working local
      // fallback; re-raise only when there is no engine to ask the backend for.
      if (!target?.installation_id) throw nativeError
    }
    // Browser mode has no pywebview bridge; let the local backend open the same
    // console. Keep the native path when it works so the window stays attached.
    if (!opened && target?.installation_id) {
      await api.engines.openTerminal(target.installation_id, python)
      opened = true
    }
    if (opened) terminalDialog.value = false
    else toast('请在桌面窗口中使用该功能（浏览器模式没有原生桥）', 'info', 6000)
  } catch (err) {
    toastError(err, '打开 Python 终端')
  }
}

/* Engine-owned installation: hardware-aware recommendation, explicit consent. */
const router = useRouter()
const installDialog = ref(false)
const installPlan = ref<InstallPlan | null>(null)
const torchSource = ref('existing')
const installConfirmed = ref(false)
/* Optional pip extras the engine declares; nothing is pre-ticked, so the default
   install stays exactly the engine's own base set. */
const installExtras = ref<string[]>([])
const mirrorPreset = ref('official')
const customMirrorUrl = ref('')
const torchMirrorPreset = ref('project_default')
const customTorchMirrorUrl = ref('')
const pythonMirrorPreset = ref('github')
const customPythonMirrorUrl = ref('')

/* The saved setting seeds both download toggles; each dialog can override it. */
const proxyConfigured = ref(false)
const installUseProxy = ref(false)
const cloneUseProxy = ref(false)
const updateUseProxy = ref(false)
api.settings
  .get()
  .then((settings) => {
    proxyConfigured.value = Boolean(settings.proxy_url)
    const on = settings.proxy_enabled && Boolean(settings.proxy_url)
    installUseProxy.value = on
    cloneUseProxy.value = on
    updateUseProxy.value = on
  })
  .catch(() => { /* leave the toggles off; the backend still applies its setting */ })

// Used only until the first install-plan response arrives (or if it fails).
const FALLBACK_PYPI: InstallOption[] = [
  { value: 'official', label: '官方 PyPI 源', url: null },
  { value: 'tuna', label: '清华大学镜像', url: 'https://pypi.tuna.tsinghua.edu.cn/simple' },
  { value: 'aliyun', label: '阿里云镜像', url: 'https://mirrors.aliyun.com/pypi/simple' },
  { value: 'ustc', label: '中科大镜像', url: 'https://mirrors.ustc.edu.cn/pypi/simple' },
  { value: 'custom', label: '自定义镜像 URL', url: null },
]
const FALLBACK_TORCH: InstallOption[] = [
  { value: 'project_default', label: '跟随引擎项目默认', url: null },
  { value: 'custom', label: '自定义 PyTorch / CUDA 源', url: null },
]
const FALLBACK_PYTHON: InstallOption[] = [
  { value: 'github', label: '默认（GitHub）', url: null },
  { value: 'nju', label: '南京大学镜像', url: 'https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone' },
  { value: 'tuna', label: '清华大学镜像', url: 'https://mirrors.tuna.tsinghua.edu.cn/github-release/astral-sh/python-build-standalone' },
  { value: 'custom', label: '自定义镜像 URL', url: null },
]

const installSources = computed(() => selected.value?.installation_sources ?? [])
const selectedAdapter = computed(() => store.adapters.find(a => a.engine_id === selected.value?.engine_id))
const selectedCompat = computed(() => selected.value?.compat ?? null)
const selectedCompatMeta = computed(() => selectedCompat.value ? engineCompatMeta[selectedCompat.value.status] : null)
const sourceLabel = (source: string) => /^cu\d{3}$/.test(source)
  ? `CUDA ${source.slice(2, -1)}.${source.slice(-1)}` : source
const mirrorPresets = computed(() => installPlan.value?.mirror_presets ?? FALLBACK_PYPI)
const torchMirrorPresets = computed(() => installPlan.value?.torch_mirror_presets ?? FALLBACK_TORCH)
const mirrorUrl = computed(() => mirrorPreset.value === 'custom'
  ? customMirrorUrl.value.trim() || null
  : mirrorPresets.value.find(p => p.value === mirrorPreset.value)?.url ?? null)
const torchMirrorUrl = computed(() => torchMirrorPreset.value === 'custom'
  ? customTorchMirrorUrl.value.trim() || null
  : torchMirrorPresets.value.find(p => p.value === torchMirrorPreset.value)?.url ?? null)
const pythonMirrorPresets = computed(() => installPlan.value?.python_mirror_presets ?? FALLBACK_PYTHON)
const pythonMirrorUrl = computed(() => pythonMirrorPreset.value === 'custom'
  ? customPythonMirrorUrl.value.trim() || null
  : pythonMirrorPresets.value.find(p => p.value === pythonMirrorPreset.value)?.url ?? null)

async function loadInstallPlan(e: EngineInstallation, source?: string) {
  try {
    installPlan.value = await api.engines.installPlan(e.installation_id, source)
  } catch (err) {
    installPlan.value = null
    toastError(err, '获取安装建议')
  }
}

function toggleExtra(name: string, checked: boolean) {
  installExtras.value = checked
    ? [...installExtras.value.filter((n) => n !== name), name]
    : installExtras.value.filter((n) => n !== name)
}

function applyPlan(plan: InstallPlan | null) {
  installConfirmed.value = false
  installExtras.value = []
  customMirrorUrl.value = ''
  customTorchMirrorUrl.value = ''
  customPythonMirrorUrl.value = ''
  torchSource.value = plan?.torch_source
    ?? installSources.value[0] ?? selectedAdapter.value?.default_installation_source ?? 'existing'
  mirrorPreset.value = plan?.mirror_preset ?? 'official'
  torchMirrorPreset.value = plan?.torch_mirror_preset ?? 'project_default'
  pythonMirrorPreset.value = plan?.python_mirror_preset ?? 'github'
}

/* Fetch the machine-aware recommendation, then prefill the dialog for confirmation. */
async function openInstall() {
  const e = selected.value
  if (e) await loadInstallPlan(e)
  applyPlan(installPlan.value)
  installDialog.value = true
}

/* Changing the CUDA source changes the wheel-mirror URLs; refresh the plan for the new tag. */
watch(torchSource, (value) => {
  const e = selected.value
  if (installDialog.value && e && value !== installPlan.value?.torch_source) void loadInstallPlan(e, value)
})

async function installEnvironment() {
  const e = selected.value
  if (!e || !installConfirmed.value) return
  if (mirrorPreset.value === 'custom' && !customMirrorUrl.value.trim()) {
    toast('请输入自定义镜像 URL', 'danger')
    return
  }
  if (torchMirrorPreset.value === 'custom' && !customTorchMirrorUrl.value.trim()) {
    toast('请填写 PyTorch / CUDA wheel 镜像 URL', 'danger')
    return
  }
  if (pythonMirrorPreset.value === 'custom' && !customPythonMirrorUrl.value.trim()) {
    toast('请填写 Python 下载镜像 URL', 'danger')
    return
  }
  busy.value = '安装引擎环境'
  try {
    const session = await api.engines.installEnvironment(e.installation_id, {
      confirmed: true,
      torch_source: torchSource.value,
      mirror_url: mirrorUrl.value,
      torch_mirror_url: torchMirrorUrl.value,
      python_mirror_url: pythonMirrorUrl.value,
      use_proxy: installUseProxy.value,
      extras: installExtras.value,
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

/** Source update: the backend only fetches and fast-forwards; nothing is staged. */
const updateDialog = ref(false)
const updateTarget = ref<EngineInstallation | null>(null)
const updateConfirmed = ref(false)
function openUpdateDialog(engine: EngineInstallation | null) {
  if (!engine) return
  updateTarget.value = engine
  updateConfirmed.value = false
  updateDialog.value = true
}
async function confirmUpdateSource() {
  if (!updateTarget.value || !updateConfirmed.value) return
  const target = updateTarget.value
  updateDialog.value = false
  await updateSource(target)
}
async function updateSource(engine: { installation_id: string }) {
  busy.value = '更新源码'
  try {
    const session = await api.engines.update(engine.installation_id, {
      use_proxy: updateUseProxy.value,
    })
    toast('已开始更新引擎源码（仅 fetch + merge --ff-only，不改动本地环境）', 'ok')
    await router.push({ name: 'terminal', query: { session: session.session_id } })
    await refreshEngines()
  } catch (err) {
    toastError(err, '更新引擎源码')
  } finally {
    busy.value = null
  }
}

/* Git clone into engine/. The terminal shows git's own progress and offers Stop. */
const cloneDialog = ref(false)
const cloneUrl = ref('')
const cloneName = ref('')
const cloneRef = ref('')
const cloneTrusted = ref(false)
/* Presets come from the adapter catalog: the shell never names an engine.
   The flatMap narrows clone_url to a string for the template's handler. */
const REPO_PRESETS = computed(() =>
  store.adapters.flatMap((a) =>
    a.clone_url ? [{ engine_id: a.engine_id, label: a.display_name || a.engine_id, url: a.clone_url }] : [],
  ),
)
const adapterFor = (engineId: string) => store.adapters.find((a) => a.engine_id === engineId)
/** Adapter-owned short status (e.g. 正在适配); empty means "show nothing". */
const statusFor = (engineId: string | null) => (engineId ? adapterFor(engineId)?.adapter_status ?? '' : '')
/** Full adapter notice, used as the badge tooltip. */
const noticeFor = (engineId: string | null) => (engineId ? adapterFor(engineId)?.training_notice ?? '' : '')
/** Adapter-declared upstream reference: the version LP is written against. */
const presetRef = (engineId: string) => adapterFor(engineId)?.clone_branch ?? ''
const presetCommit = (engineId: string) => adapterFor(engineId)?.clone_commit ?? ''
const cloneCommit = ref('')

function applyPreset(preset: { engine_id: string; url: string }) {
  cloneUrl.value = preset.url
  cloneRef.value = presetRef(preset.engine_id)
  cloneCommit.value = presetCommit(preset.engine_id)
}
/** Folder git would create, shown as the placeholder so the default is visible. */
const cloneDefaultName = computed(() => {
  const tail = cloneUrl.value.trim().replace(/\/+$/, '').split(/[/:]/).pop() ?? ''
  return tail.replace(/\.git$/, '').replace(/[^A-Za-z0-9._-]/g, '-').replace(/^[.-]+|[.-]+$/g, '') || 'engine'
})
const cloneNameBase = computed(() => cloneName.value.trim() || cloneDefaultName.value)
const cloneNameWillSuffix = computed(() => {
  const base = cloneNameBase.value.toLowerCase()
  return store.engines.some((e) => e.source_path.replace(/\\/g, '/').split('/').pop()?.toLowerCase() === base)
})
const cloneResolvedName = computed(() => {
  const base = cloneNameBase.value
  if (!cloneNameWillSuffix.value) return base
  for (let index = 2; index < 1000; index++) {
    const suffix = `-${index}`
    const candidate = base.slice(0, Math.max(1, 100 - suffix.length)) + suffix
    if (!store.engines.some((e) => e.source_path.replace(/\\/g, '/').split('/').pop()?.toLowerCase() === candidate.toLowerCase())) {
      return candidate
    }
  }
  return base
})
const cloneUrlValid = computed(() => /^(https?:\/\/|ssh:\/\/|file:\/\/\/|git@)\S+$/.test(cloneUrl.value.trim()))
const cloneUsesSsh = computed(() => /^(ssh:\/\/|git@)/.test(cloneUrl.value.trim()))
/** Same rule the backend enforces: empty means "branch head", else 7-40 hex characters. */
const cloneCommitValid = computed(() => {
  const value = cloneCommit.value.trim()
  return !value || /^[0-9a-fA-F]{7,40}$/.test(value)
})
function openClone() {
  cloneUrl.value = ''
  cloneName.value = ''
  cloneRef.value = ''
  cloneTrusted.value = false
  cloneCommit.value = ''
  // Prefill from the selected engine's adapter: the branch and commit LP targets.
  const engineId = selected.value?.engine_id ?? ''
  const declared = adapterFor(engineId)?.clone_url
  if (declared) {
    cloneUrl.value = declared
    cloneRef.value = presetRef(engineId)
    cloneCommit.value = presetCommit(engineId)
  }
  cloneDialog.value = true
}
async function repairSource(e: EngineInstallation) {
  busy.value = '补齐训练源码'
  try {
    // Follows the saved proxy setting: this panel has no per-operation dialog.
    const session = await api.engines.repairSource(e.installation_id)
    toast('已开始补齐训练源码，进度显示在终端', 'ok')
    await router.push({ name: 'terminal', query: { session: session.session_id } })
    await refreshEngines()
  } catch (err) {
    toastError(err, '补齐训练源码')
  } finally {
    busy.value = null
  }
}

async function cloneEngine() {
  if (!cloneUrlValid.value || !cloneCommitValid.value || !cloneTrusted.value) return
  busy.value = '拉取引擎'
  try {
    const session = await api.engines.clone({
      url: cloneUrl.value.trim(),
      name: cloneName.value.trim() || null,
      ref: cloneRef.value.trim() || null,
      commit: cloneCommit.value.trim() || null,
      use_proxy: cloneUseProxy.value,
    })
    cloneDialog.value = false
    toast(`已开始拉取引擎，目标目录 engine/${session.label}`, 'ok')
    await router.push({ name: 'terminal', query: { session: session.session_id } })
  } catch (err) {
    toastError(err, '拉取引擎')
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
  typeChoice.value = e?.engine_id ?? e?.candidate_engines[0] ?? store.adapters[0]?.engine_id ?? ''
  typeDialog.value = true
}
async function saveType() {
  const e = selected.value
  if (!e || !typeChoice.value) return
  const saved = await run('确认引擎类型', () => api.engines.confirmType(e.installation_id, typeChoice.value), '已登记引擎类型')
  if (saved) typeDialog.value = false
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
          <AppIcon name="refresh" :size="15" :class="{ spin: busy === '刷新发现' }" />
          {{ busy === '刷新发现' ? '正在刷新…' : '刷新发现' }}
        </button>
        <button class="btn btn--primary" :disabled="busy !== null" @click="openClone">
          <AppIcon name="download" :size="15" />
          Git 拉取引擎
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

        <p v-if="store.engineLoadError" class="field__error selectable" role="alert">{{ store.engineLoadError }}</p>
        <p v-for="error in store.adapterErrors" :key="error.directory" class="field__error selectable" role="alert">
          适配器 {{ error.directory }} 加载失败：{{ error.message }}
        </p>
        <EmptyState
          v-if="store.enginesLoaded && !store.engineLoadError && !store.engines.length"
          icon="cpu"
          title="未发现任何引擎安装实例"
          text="将已适配的训练器源码目录复制到 engine/ 下的任意子目录，然后点击“刷新发现”。识别只读取文件，不会执行引擎代码。"
        >
          <button class="btn btn--primary" @click="rescan">刷新发现</button>
        </EmptyState>

        <div v-for="g in groups" :key="g.key" class="group">
          <div class="group__head">
            <span class="label-pill">{{ g.title }}</span>
            <!-- Adapter-owned status badge. Once an instance is genuinely ready the
                 warning is noise: it describes the adaptation, not this environment. -->
            <span
              v-if="statusFor(g.key) && !g.items.some((e) => e.state === 'ready')"
              class="chip chip--warn"
              :title="noticeFor(g.key)"
            >{{ statusFor(g.key) }}</span>
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
                </span>
                <span class="inst__path mono truncate">{{ e.source_path }}</span>
              </span>
              <span class="inst__rev mono">
                {{ shortCommit(e.revision.commit) ?? (e.revision.fingerprint ? '指纹' : '—') }}
              </span>
              <StatusTag v-bind="installStateMeta[e.state]" />
              <StatusTag v-bind="verificationMeta[e.verification]" :dot="false" />
              <StatusTag
                v-if="e.optional_modules?.triton !== undefined"
                :tone="e.optional_modules.triton ? 'ok' : 'warn'"
                label="Triton"
                :title="e.optional_modules.triton
                  ? 'Triton 已就绪：环境可导入 triton/tilelang，COMPILE、VAE 编译与 triton 内核后端可用'
                  : 'Triton 未安装：缺少 triton/tilelang，启用 COMPILE、VAE 编译或 triton 内核会报 TritonMissing'"
              />
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

        <!-- A ready engine's remarks are advisory (python source, adaptation scope), so
             they stay folded: the badge above already answered "can I train". -->
        <details v-if="selected.issues.length && selected.state === 'ready'" class="issues issues--folded">
          <summary>{{ selected.issues.length }} 条备注</summary>
          <div class="issues__list">
            <div v-for="i in selected.issues" :key="i" class="issue">
              <AppIcon name="alert" :size="13" />
              <span>{{ i }}</span>
            </div>
          </div>
        </details>
        <div v-else-if="selected.issues.length" class="issues">
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
          <dt>适配基线</dt>
          <dd>
            <div v-if="selectedCompat && selectedCompatMeta" class="row" style="gap: 8px; align-items: center; flex-wrap: wrap">
              <StatusTag v-bind="selectedCompatMeta" :dot="false" />
              <span v-if="selectedCompat.baseline" class="mono">
                {{ selectedCompat.version }} · {{ selectedCompat.baseline.ref ?? '' }}
                {{ shortCommit(selectedCompat.baseline.commit) }}
              </span>
              <span v-else class="muted">当前项目版本未登记该引擎基线</span>
            </div>
            <span v-else class="muted">未提供</span>
            <div v-if="selectedCompat?.reasons.length" class="field__hint" style="margin-top: 4px">
              {{ selectedCompat.reasons.join('；') }}
            </div>
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
          <button
            class="btn btn--sm btn--primary"
            :disabled="busy !== null || !selected.engine_id || selected.state === 'preparing' || selected.state === 'missing'"
            title="按本机显卡与系统给出推荐值，确认后安装"
            @click="openInstall"
          >
            <AppIcon name="terminal" :size="14" />
            安装引擎环境
          </button>
          <button
            class="btn btn--sm"
            :disabled="busy !== null || !selected.engine_id || selected.state === 'preparing' || selected.state === 'missing'"
            title="git fetch + merge --ff-only：只更新上游源码，不会暂存或改动 .lp/python（embedded 环境）、.venv 或缓存目录"
            @click="openUpdateDialog(selected)"
          >
            <AppIcon name="refresh" :size="14" />
            更新源码
          </button>
          <button
            v-if="selected.source_repair"
            class="btn btn--sm"
            :disabled="busy !== null || selected.state === 'preparing'"
            :title="selected.source_repair"
            @click="repairSource(selected)"
          >
            <AppIcon name="download" :size="14" />
            补齐训练源码
          </button>
          <button class="btn btn--sm" :disabled="busy !== null || !selected.python_executable || selected.state === 'preparing'" @click="diagnose(selected)">
            <AppIcon name="refresh" :size="13" :class="{ spin: busy === '环境诊断' }" />
            {{ busy === '环境诊断' ? '诊断中…' : '环境诊断' }}
          </button>
          <button
            class="btn btn--sm detail__wide"
            :disabled="busy !== null"
            title="打开一个配好该解释器的命令行窗口，便于手动安装依赖"
            @click="openTerminalDialog(selected)"
          >
            <AppIcon name="terminal" :size="14" />
            打开 Python 终端
          </button>
          <button
            class="btn btn--sm btn--danger detail__danger"
            :disabled="busy !== null || selected.state === 'preparing'"
            title="只删除 LP 的登记条目，不删除 engine/ 下的文件"
            @click="removeDialog = true"
          >删除条目</button>
        </div>
        <p class="field__hint">
          已排队或运行中的任务始终使用其提交时固定的安装实例；这里的操作只影响之后新建的配置。
        </p>
      </section>
      <section v-else class="panel">
        <EmptyState icon="cpu" title="未选择实例" />
      </section>

      <section class="panel panel--pad stack note">
        <span class="label-pill">接入方式</span>
        <ol>
          <li>点击“Git 拉取引擎”从仓库拉取，或手动复制完整引擎目录到 <code>engine/&lt;任意名称&gt;/</code></li>
          <li>刷新发现，确认识别出的引擎类型</li>
          <li>确认源码可信后，通过“安装引擎环境”调用该引擎自身的安装流程</li>
          <li>在终端查看安装输出；诊断通过后，按适配器已支持的训练能力使用</li>
        </ol>
      </section>
    </aside>

    <ModalDialog v-if="installDialog" title="安装引擎环境" @close="installDialog = false">
      <p class="muted">将调用 {{ engineName(selected?.engine_id ?? '') }} 的安装流程，不会统一猜测依赖。训练依赖只安装到当前引擎目录内的独立环境，不安装进应用运行时或系统 Python。</p>
      <p class="field__hint">{{ selectedAdapter?.installation_help ?? '由当前引擎适配器提供安装流程。' }}</p>
      <div v-if="selectedAdapter?.optional_extras?.length" class="install-extras">
        <span class="field__label">可选组件（默认不安装）</span>
        <label v-for="extra in selectedAdapter.optional_extras" :key="extra.name" class="install-extras__row">
          <input
            type="checkbox"
            :value="extra.name"
            :checked="installExtras.includes(extra.name)"
            @change="toggleExtra(extra.name, ($event.target as HTMLInputElement).checked)"
          />
          <span>{{ extra.label }}</span>
        </label>
        <p class="field__hint">勾选后以 <code>-e 源码[{{ installExtras.join(',') || '…' }}]</code> 安装；不勾选保持引擎原始依赖集。</p>
      </div>
      <p v-if="installPlan" class="field__hint">
        推荐：{{ installPlan.torch_source_reason }}<template v-if="installPlan.gpu_name">（{{ installPlan.gpu_name }}<template v-if="installPlan.driver_version">，驱动 {{ installPlan.driver_version }}</template><template v-if="installPlan.cuda_version">，驱动支持 CUDA {{ installPlan.cuda_version }}</template>）</template>
      </p>
      <p v-if="installPlan && !installPlan.supported" class="field__error">无法安装：{{ installPlan.blocked_reason }}</p>
      <label v-if="installPlan && !installPlan.engine_decides_cuda && installPlan.torch_source_options.length > 1" class="field">
        <span class="field__label">引擎安装来源 / CUDA 配置</span>
        <SelectInput v-model="torchSource" class="input">
          <option v-for="source in installPlan.torch_source_options" :key="source" :value="source">{{ sourceLabel(source) }}</option>
        </SelectInput>
        <span class="field__hint">{{ selectedAdapter?.installation_source_help }}</span>
      </label>
      <label v-else class="field">
        <span class="field__label">CUDA 配置</span>
        <p class="field__hint">{{ selectedAdapter?.installation_source_help ?? '该引擎由官方安装器自行选择 CUDA。' }}</p>
      </label>
      <label class="field">
        <span class="field__label">Python / uv 镜像源</span>
        <SelectInput v-model="mirrorPreset" class="input">
          <option v-for="preset in mirrorPresets" :key="preset.value" :value="preset.value">{{ preset.label }}</option>
        </SelectInput>
        <input
          v-if="mirrorPreset === 'custom'"
          v-model="customMirrorUrl"
          class="input"
          type="url"
          placeholder="https://example.com/simple"
        />
        <span class="field__hint">仅设置依赖包索引，会传递给引擎自己的 pip/uv 安装流程；Musubi 声明的 PyTorch CUDA 专用源仍由项目配置管理。</span>
      </label>
      <label class="field">
        <span class="field__label">PyTorch / CUDA wheel 镜像源 <span v-if="installPlan?.cuda_tag" class="field__hint">{{ installPlan.cuda_tag }}</span></span>
        <SelectInput v-model="torchMirrorPreset" class="input">
          <option v-for="preset in torchMirrorPresets" :key="preset.value" :value="preset.value">{{ preset.label }}</option>
        </SelectInput>
        <input
          v-if="torchMirrorPreset === 'custom'"
          v-model="customTorchMirrorUrl"
          class="input"
          type="url"
          :placeholder="installPlan?.cuda_tag ? `https://download.pytorch.org/whl/${installPlan.cuda_tag}` : 'https://download.pytorch.org/whl/cu130'"
        />
        <span class="field__hint">wheel 地址随所选 CUDA 生成。由官方安装器自选 CUDA 的引擎会把它作为附加 pip/uv 索引注入；若官方安装器显式指定 index 可能不生效。自定义时请填写实际提供对应 CUDA wheel 的地址，例如 /whl/{{ installPlan?.cuda_tag ?? 'cu130' }}。</span>
      </label>
      <label class="field">
        <span class="field__label">Python 下载源</span>
        <SelectInput v-model="pythonMirrorPreset" class="input">
          <option v-for="preset in pythonMirrorPresets" :key="preset.value" :value="preset.value">{{ preset.label }}</option>
        </SelectInput>
        <input
          v-if="pythonMirrorPreset === 'custom'"
          v-model="customPythonMirrorUrl"
          class="input"
          type="url"
          placeholder="https://mirror.example/github-release/astral-sh/python-build-standalone"
        />
        <span class="field__hint">安装器需要当前机器尚无的 Python 版本时，uv 会下载受管 Python（Kohya 需要 3.11）。GitHub 不可达时安装会报「Failed to download .../python-build-standalone/...」，此时请改选镜像；镜像需按 GitHub release 目录结构提供 <code>版本目录/文件</code>。</span>
      </label>
      <label class="row" style="gap: 10px; margin-top: 16px">
        <input v-model="installUseProxy" type="checkbox" :disabled="!proxyConfigured" />
        <span>启用代理下载<template v-if="!proxyConfigured">（未配置代理地址，请到「设置」填写）</template></span>
      </label>
      <label class="row" style="gap: 10px; margin-top: 16px">
        <input v-model="installConfirmed" type="checkbox" />
        <span>我信任这份引擎源码，并允许其安装流程联网下载依赖及修改所选环境。</span>
      </label>
      <template #footer>
        <button class="btn btn--ghost" @click="installDialog = false">取消</button>
        <button class="btn btn--primary" :disabled="!installConfirmed || busy !== null || (installPlan ? !installPlan.supported : false)" @click="installEnvironment">安装并打开终端</button>
      </template>
    </ModalDialog>

    <ModalDialog v-if="updateDialog" title="更新引擎源码？" :width="520" @close="updateDialog = false">
      <p class="muted mono selectable">{{ updateTarget?.label || updateTarget?.engine_id }} · {{ updateTarget?.source_path }}</p>
      <p>本次只执行 <code>git fetch</code> + <code>merge --ff-only</code>，不会暂存或改动 <code>.lp/python</code>、<code>.venv</code> 或缓存目录。</p>
      <div class="update-warning">
        <strong>更新后可能不再支持当前参数与配置。</strong>
        新版本可能改动可用参数、默认值、启动方式或输出格式；LP 当前适配器、参数校验和已保存的任务配置不一定同步兼容。更新后请重新检查参数，必要时重新选择引擎/重新检测；旧配置可能无法提交或运行。
      </div>
      <label class="row" style="gap: 10px; margin-top: 14px; align-items: flex-start">
        <input v-model="updateConfirmed" type="checkbox" style="margin-top: 3px" />
        <span>我已了解更新后可能需要重新适配参数与任务配置。</span>
      </label>
      <label class="row" style="gap: 10px; margin-top: 10px; align-items: flex-start">
        <input v-model="updateUseProxy" type="checkbox" :disabled="!proxyConfigured" style="margin-top: 3px" />
        <span>启用代理下载<template v-if="!proxyConfigured">（未配置代理地址，请到「设置」填写）</template></span>
      </label>
      <p v-if="updateUseProxy" class="field__hint">
        HTTP(S) 远程仓库会经过代理；SSH 地址需要在 Git/SSH 配置里设置 ProxyCommand。
      </p>
      <template #footer>
        <button class="btn btn--ghost" @click="updateDialog = false">取消</button>
        <button class="btn btn--warn" :disabled="!updateConfirmed || busy !== null" @click="confirmUpdateSource">
          <AppIcon name="refresh" :size="14" />
          继续更新
        </button>
      </template>
    </ModalDialog>

    <ModalDialog v-if="removeDialog" title="删除引擎条目？" :width="460" @close="removeDialog = false">
      <p>{{ selected?.label || selected?.engine_id || '当前条目' }}</p>
      <p class="muted mono selectable">{{ selected?.source_path }}</p>
      <p class="muted">只删除 LP 的登记条目（类型、解释器绑定与状态），<strong>不会删除 <code>engine/</code> 下的任何文件</strong>。若该实例已被任务引用，删除会被拒绝，以保留任务历史。</p>
      <template #footer>
        <button class="btn btn--ghost" @click="removeDialog = false">取消</button>
        <button class="btn btn--danger" @click="removeEngine">删除条目</button>
      </template>
    </ModalDialog>

    <ModalDialog v-if="terminalDialog" title="打开 Python 终端" :width="560" @close="terminalDialog = false">
      <p class="muted">会打开一个独立的命令行窗口：该解释器所在目录会加入 PATH，若它是虚拟环境还会一并激活。LP 不执行任何命令，你在窗口里手动安装依赖。</p>
      <label class="field">
        <span class="field__label">Python 解释器</span>
        <PathInput v-model="terminalPython" kind="file" dialog-title="选择 python.exe" :file-types="['Python executable (*.exe)']" />
        <span class="field__hint">默认使用所选安装实例的解释器，也可以改成别的 python.exe。</span>
      </label>
      <p class="field__hint">若该环境的 pip 不可用，可在窗口里执行 <code>uv pip install --python "&lt;解释器路径&gt;" &lt;包名&gt;</code>（LP 自带 uv），或先运行 <code>-m ensurepip --upgrade</code> 装回 pip。</p>
      <template #footer>
        <button class="btn btn--ghost" @click="terminalDialog = false">取消</button>
        <button class="btn btn--primary" :disabled="!terminalPython.trim()" @click="confirmTerminal">打开终端</button>
      </template>
    </ModalDialog>

    <ModalDialog v-if="cloneDialog" title="Git 拉取引擎" :width="560" @close="busy === null && (cloneDialog = false)">
      <p class="muted">从 Git 仓库克隆引擎源码到 <code>engine/&lt;目录名&gt;/</code>。拉取只下载源码，不会运行其中任何代码；拉取完成后请在列表确认引擎类型，再单独安装环境。</p>
      <div class="row" style="flex-wrap: wrap; gap: 6px">
        <button v-for="p in REPO_PRESETS" :key="p.url" type="button" class="chip" @click="applyPreset(p)">{{ p.label }}</button>
      </div>
      <label class="field">
        <span class="field__label">仓库地址（必填）</span>
        <input v-model="cloneUrl" class="input mono" spellcheck="false" placeholder="https://github.com/用户/仓库.git" />
        <span v-if="cloneUrl && !cloneUrlValid" class="field__error">请填写 https://、ssh://、file:/// 或 git@ 开头的地址</span>
      </label>
      <div class="row" style="gap: 12px; align-items: flex-start">
        <label class="field" style="flex: 1">
          <span class="field__label">目录名</span>
          <input v-model="cloneName" class="input mono" spellcheck="false" :placeholder="cloneDefaultName" maxlength="100" />
          <span v-if="cloneNameWillSuffix" class="field__hint">已存在同名目录，将自动创建 <code>engine/{{ cloneResolvedName }}</code></span>
        </label>
        <label class="field" style="flex: 1">
          <span class="field__label">分支 / 标签（可选）</span>
          <input v-model="cloneRef" class="input mono" spellcheck="false" placeholder="默认分支" maxlength="200" />
        </label>
      </div>
      <label class="field">
        <span class="field__label">提交（可选）</span>
        <input v-model="cloneCommit" class="input mono" spellcheck="false" placeholder="留空则检出所选分支的最新提交" maxlength="40" />
        <span v-if="!cloneCommitValid" class="field__error">提交号应为 7–40 位十六进制字符</span>
        <span v-else class="field__hint">默认检出 LP 已验证的提交（需存在于所选分支 / 标签）；清空则拉取该分支的最新提交。</span>
      </label>
      <p class="field__hint">需要本机已安装 Git 且在 PATH 中。私有仓库请使用已配置好凭据的 Git（拉取时不会弹出登录提示）；地址中的账号密码不会写入日志。</p>
      <label class="row" style="gap: 10px; margin-top: 12px">
        <input v-model="cloneUseProxy" type="checkbox" :disabled="!proxyConfigured" />
        <span>启用代理下载<template v-if="!proxyConfigured">（未配置代理地址，请到「设置」填写）</template></span>
      </label>
      <p v-if="cloneUseProxy && cloneUsesSsh" class="field__hint">
        SSH 地址（ssh:// 或 git@）不会经过 HTTP 代理；如需代理，请改用 HTTPS 地址，或在 Git/SSH 配置中设置 ProxyCommand。
      </p>
      <label class="row" style="gap: 10px; margin-top: 12px">
        <input v-model="cloneTrusted" type="checkbox" />
        <span>我信任这个仓库来源，并允许把它下载到 engine/ 目录。</span>
      </label>
      <template #footer>
        <button class="btn btn--ghost" @click="cloneDialog = false">取消</button>
        <button class="btn btn--primary" :disabled="!cloneUrlValid || !cloneCommitValid || !cloneTrusted || busy !== null" @click="cloneEngine">拉取并打开终端</button>
      </template>
    </ModalDialog>

    <ModalDialog v-if="pyDialog" title="选择 Python 解释器" @close="pyDialog = false">
      <p class="muted">
        选择该安装实例使用的 python.exe。检索顺序：<code>&lt;引擎&gt;/.lp/python</code>（embedded，可迁移）→ <code>.venv</code> / <code>venv</code> / <code>env</code> → 引擎根目录。复制来的环境不会被默认视为可用，后端会核实解释器、路径绑定和必要依赖。
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
          v-for="{ engine_id: id } in store.adapters"
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
  /* knob · label/path · revision · state · verification · optional modules (Triton).
     The column count must match the children exactly: an extra child without a
     column wraps onto an implicit second row. */
  grid-template-columns: auto minmax(0, 1fr) 90px auto auto auto;
  align-items: center;
  gap: 14px;
  padding: 12px 16px 12px 10px;
  border-radius: var(--radius-md);
  background: var(--panel-raised);
  text-align: left;
  transition: background 0.15s, box-shadow 0.15s;
}
.inst:hover {
  background: var(--ink-soft);
}
/* Optional install components: unticked by default, so the base install stays the
   engine's own dependency set. */
.install-extras {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--field);
}
.install-extras__row {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  font-size: var(--text-sm);
  color: var(--text-2);
  cursor: pointer;
}
.install-extras__row input {
  margin-top: 2px;
  accent-color: var(--light);
}
.update-warning {
  margin-top: 12px;
  padding: 10px 12px;
  border: 1px solid var(--warn);
  border-radius: var(--radius-sm);
  background: var(--warn-bg);
  color: var(--text-2);
  font-size: var(--text-sm);
  line-height: 1.65;
}
.update-warning strong {
  color: var(--warn);
}
/* Ready engines keep their remarks folded behind one summary line. The items live in
   their own flex list so the spacing never depends on how <details> lays out. */
.issues--folded {
  display: block;
}
.issues--folded > summary {
  cursor: pointer;
  font-size: var(--text-xs);
  color: var(--text-3);
  padding: 2px 0;
  list-style-position: inside;
}
.issues--folded[open] > summary {
  margin-bottom: 6px;
}
.issues__list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.inst.is-selected {
  box-shadow: inset 0 0 0 1.5px var(--light);
}
/* Destructive entry removal sits on its own line, away from the routine actions. */
.detail__danger {
  grid-column: 1 / -1;
  justify-self: start;
  margin-top: 4px;
}
/* Full-width rows inside the two-column action grid. */
.detail__wide {
  grid-column: 1 / -1;
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
/* Two even columns: the routine actions pair up cleanly instead of leaving a ragged
   half-row when one of them is absent. Destructive removal spans the full width. */
.actions {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
}
.actions .btn {
  justify-content: center;
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
