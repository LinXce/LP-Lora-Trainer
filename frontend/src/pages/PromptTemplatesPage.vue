<script setup lang="ts">
import { computed, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import AppIcon from '@/components/AppIcon.vue'
import EmptyState from '@/components/EmptyState.vue'
import ModalDialog from '@/components/ModalDialog.vue'
import { api } from '@/api'
import { toast, toastError } from '@/features/toast'
import type { PromptTemplate } from '@/types/api'

const router = useRouter()
const templates = ref<PromptTemplate[]>([])
const loading = ref(false)
const error = ref('')
const search = ref('')
const filtered = computed(() => {
  const text = search.value.trim().toLowerCase()
  return templates.value.filter(t => !text || t.name.toLowerCase().includes(text) || t.prompt.toLowerCase().includes(text))
})
const editor = ref(false)
const editingId = ref<string | null>(null)
const form = reactive({ name: '', prompt: '' })
const deleting = ref<PromptTemplate | null>(null)
const saving = ref(false)
const resultLabel = computed(() => search.value.trim() ? `匹配 ${filtered.value.length} / ${templates.value.length}` : `共 ${templates.value.length} 个`)
const PREVIEW_LINES = 4
const totalLines = computed(() => templates.value.reduce((sum, t) => sum + lineCount(t.prompt), 0))
const lastUpdated = computed(() => templates.value.reduce((latest, t) => (t.updated_at > latest ? t.updated_at : latest), ''))

function promptLines(prompt: string) {
  return prompt.split(/\r?\n/).filter(line => line.trim())
}
function lineCount(prompt: string) {
  return promptLines(prompt).length
}

function formatUpdatedAt(value: string) {
  if (!value) return '未记录更新时间'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '未记录更新时间'
  return new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }).format(date)
}

async function load() {
  loading.value = true; error.value = ''
  try { templates.value = await api.promptTemplates.list() }
  catch (err) { error.value = err instanceof Error ? err.message : String(err) }
  finally { loading.value = false }
}
function edit(item?: PromptTemplate) {
  editingId.value = item?.template_id ?? null
  form.name = item?.name ?? ''; form.prompt = item?.prompt ?? ''; editor.value = true
}
async function save() {
  if (saving.value || !form.name.trim() || !form.prompt.trim()) return
  saving.value = true
  try {
    const item = editingId.value ? await api.promptTemplates.update(editingId.value, { ...form }) : await api.promptTemplates.create({ ...form })
    templates.value = [item, ...templates.value.filter(t => t.template_id !== item.template_id)]
    editor.value = false; toast('提示词模板已保存', 'ok')
  } catch (err) { toastError(err, '保存提示词模板') }
  finally { saving.value = false }
}
async function remove() {
  if (saving.value || !deleting.value) return
  saving.value = true
  try {
    const id = deleting.value.template_id
    await api.promptTemplates.delete(id)
    templates.value = templates.value.filter(t => t.template_id !== id)
    deleting.value = null; toast('提示词模板已删除，历史任务不受影响', 'ok')
  } catch (err) { toastError(err, '删除提示词模板') }
  finally { saving.value = false }
}
void load()
</script>

<template>
  <div class="page">
    <section class="page__main panel">
      <header class="page__header">
        <div class="stack" style="gap: 4px; min-width: 0">
          <h1 class="title-lg">提示词模板</h1>
          <p class="muted truncate">管理可复用的采样提示词；也可在新建训练中直接输入一次性内容。</p>
        </div>
        <span class="spacer" />
        <button class="btn btn--icon" :disabled="loading" title="刷新" aria-label="刷新" @click="load"><AppIcon name="refresh" :size="15" /></button>
        <button class="btn btn--primary" @click="edit()"><AppIcon name="plus" :size="15" />新建模板</button>
      </header>
      <div class="prompt-toolbar">
        <label class="prompt-search">
          <AppIcon name="search" :size="15" />
          <input v-model="search" class="prompt-search__input" placeholder="搜索模板名称或提示词内容" aria-label="搜索提示词模板" />
          <button v-if="search" type="button" class="prompt-search__clear" aria-label="清除搜索" @click="search = ''"><AppIcon name="x" :size="13" /></button>
        </label>
        <span class="prompt-count num">{{ resultLabel }}</span>
      </div>
      <div class="page__body">
        <p v-if="error" class="field__error">{{ error }}</p>
        <p v-else-if="loading && !templates.length" class="prompt-loading"><span class="prompt-loading__dot" />正在读取模板…</p>
        <div v-else-if="filtered.length" class="prompt-grid">
          <article v-for="(item, i) in filtered" :key="item.template_id" class="prompt-card enter" :style="{ animationDelay: `${Math.min(i, 8) * 30}ms` }">
            <div class="prompt-card__head">
              <h3 class="prompt-card__name selectable" :title="item.name">{{ item.name }}</h3>
              <span class="chip prompt-card__count num">{{ lineCount(item.prompt) }} 条</span>
            </div>
            <ol class="prompt-lines mono" title="点击编辑" @click="edit(item)">
              <li v-for="(line, n) in promptLines(item.prompt).slice(0, PREVIEW_LINES)" :key="n">
                <span class="prompt-lines__no num">{{ n + 1 }}</span>
                <span class="prompt-lines__text">{{ line }}</span>
              </li>
              <li v-if="lineCount(item.prompt) > PREVIEW_LINES" class="prompt-lines__more">
                还有 {{ lineCount(item.prompt) - PREVIEW_LINES }} 条…
              </li>
            </ol>
            <div class="prompt-card__foot">
              <span class="prompt-card__date num">{{ formatUpdatedAt(item.updated_at) }}</span>
              <span class="spacer" />
              <button class="btn btn--ghost btn--sm btn--icon prompt-card__icon-btn" title="编辑" aria-label="编辑" @click="edit(item)"><AppIcon name="edit" :size="13" /></button>
              <button class="btn btn--ghost btn--sm btn--icon prompt-card__icon-btn prompt-card__delete" title="删除" aria-label="删除" @click="deleting = item"><AppIcon name="trash" :size="13" /></button>
              <button class="btn btn--primary btn--sm" @click="router.push({ name: 'train', query: { promptTemplate: item.template_id } })">用于训练<AppIcon name="chevron" :size="13" /></button>
            </div>
          </article>
        </div>
        <div v-else class="prompt-empty">
          <EmptyState icon="file" :title="search ? '没有匹配的模板' : '还没有提示词模板'" :text="search ? '尝试更换关键词，或清除搜索条件。' : '新建模板，或在新建训练的预览图参数中将当前提示词保存为模板。'" />
          <button v-if="search" class="btn btn--ghost btn--sm" @click="search = ''">清除搜索条件</button>
          <button v-else class="btn btn--primary btn--sm" @click="edit()"><AppIcon name="plus" :size="14" />新建第一个模板</button>
        </div>
      </div>
    </section>

    <aside class="page__side">
      <section class="panel panel--light panel--pad prompt-stats enter">
        <span class="eyebrow prompt-stats__eyebrow">提示词库</span>
        <div class="prompt-stats__grid">
          <div><strong class="num">{{ templates.length }}</strong><span>模板</span></div>
          <div><strong class="num">{{ totalLines }}</strong><span>提示词</span></div>
        </div>
        <span class="prompt-stats__updated">最近更新 · {{ lastUpdated ? formatUpdatedAt(lastUpdated) : '—' }}</span>
      </section>
      <section class="panel panel--pad stack prompt-guide enter" style="animation-delay: 60ms">
        <span class="label-pill">使用说明</span>
        <ul class="prompt-guide__list">
          <li><AppIcon name="info" :size="13" /><span>每行一个采样提示词，每行对应一次采样。</span></li>
          <li><AppIcon name="layers" :size="13" /><span>纯文本模板可以跨引擎复用。</span></li>
          <li><AppIcon name="alert" :size="13" /><span>行内参数请遵守目标引擎格式，AI Toolkit 不解析 Kohya / Musubi 的行内选项。</span></li>
          <li><AppIcon name="file" :size="13" /><span>修改或删除模板不会影响已提交的任务。</span></li>
        </ul>
      </section>
    </aside>

    <ModalDialog v-if="editor" :title="editingId ? '编辑提示词模板' : '新建提示词模板'" :width="680" @close="!saving && (editor = false)">
      <div class="prompt-editor">
        <label class="field"><span class="field__label">模板名称（必填）</span><input v-model="form.name" class="input" maxlength="200" :disabled="saving" placeholder="例如：人物肖像采样" /></label>
        <label class="field"><span class="field__label">提示词内容（必填）<span class="prompt-editor__counter">{{ form.prompt.length.toLocaleString() }} / 100,000</span></span><textarea v-model="form.prompt" class="input area mono prompt-editor__textarea" rows="10" maxlength="100000" spellcheck="false" :disabled="saving" placeholder="每行一个采样提示词" /></label>
        <div class="prompt-editor__hint"><AppIcon name="info" :size="14" /><span>每行代表一次采样。保存模板不会修改已经使用此模板的任务或暂存参数。</span></div>
      </div>
      <template #footer>
        <button class="btn btn--ghost" :disabled="saving" @click="editor = false">取消</button>
        <button class="btn btn--primary" :disabled="saving || !form.name.trim() || !form.prompt.trim()" @click="save">{{ saving ? '保存中…' : '保存' }}</button>
      </template>
    </ModalDialog>
    <ModalDialog v-if="deleting" title="删除提示词模板？" @close="!saving && (deleting = null)">
      <p>将删除「{{ deleting.name }}」。已提交任务及已复制到表单的提示词不会被删除。</p>
      <template #footer>
        <button class="btn btn--ghost" :disabled="saving" @click="deleting = null">取消</button>
        <button class="btn btn--danger" :disabled="saving" @click="remove">{{ saving ? '删除中…' : '确认删除' }}</button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
/* Toolbar sits between the header and the scrolling grid, so it stays visible. */
.prompt-toolbar { display: flex; align-items: center; gap: 12px; padding: 0 26px 16px; }
.prompt-search { display: flex; align-items: center; gap: 10px; min-width: 0; flex: 1; height: 36px; padding: 0 8px 0 14px; border: 1px solid var(--line); border-radius: var(--radius-pill); background: var(--field); color: var(--text-3); transition: border-color .15s, background .15s; }
.prompt-search:focus-within { border-color: var(--line-strong); background: var(--field-focus); color: var(--text-2); }
.prompt-search__input { min-width: 0; flex: 1; border: 0; outline: 0; background: transparent; color: var(--text); font-size: 13px; user-select: text; }
.prompt-search__input::placeholder { color: var(--text-3); }
.prompt-search__clear { display: inline-flex; align-items: center; justify-content: center; width: 22px; height: 22px; border-radius: 50%; color: var(--text-3); }
.prompt-search__clear:hover { background: rgba(255, 255, 255, .08); color: var(--text); }
.prompt-count { flex: none; color: var(--text-3); font-size: 11px; white-space: nowrap; }

.prompt-loading { display: flex; align-items: center; gap: 8px; color: var(--text-3); }
.prompt-loading__dot { width: 7px; height: 7px; border-radius: 50%; background: var(--light-strong); animation: pulse 1.1s ease-in-out infinite; }

.prompt-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr)); gap: var(--gap); }
.prompt-card { display: flex; flex-direction: column; gap: 12px; min-width: 0; padding: 16px 16px 12px; border-radius: var(--radius-md); background: var(--panel-raised); transition: background .18s, transform .2s var(--ease); }
.prompt-card:hover { background: var(--panel-strong); transform: translateY(-2px); }
.prompt-card__head { display: flex; align-items: center; gap: 10px; min-width: 0; }
.prompt-card__name { flex: 1; min-width: 0; margin: 0; font-size: 14px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.prompt-card__count { flex: none; height: 22px; padding: 0 9px; font-size: 11px; }

/* Numbered preview of the first few prompts; click opens the editor. */
.prompt-lines { display: flex; flex-direction: column; gap: 4px; min-height: 104px; margin: 0; padding: 10px 12px; list-style: none; border-radius: var(--radius-sm); background: rgba(0, 0, 0, .16); font-size: 11.5px; line-height: 1.5; cursor: pointer; transition: background .15s; }
.prompt-lines:hover { background: rgba(0, 0, 0, .22); }
.prompt-lines li { display: flex; gap: 10px; min-width: 0; }
.prompt-lines__no { flex: none; width: 14px; text-align: right; color: var(--text-3); opacity: .7; }
.prompt-lines__text { min-width: 0; color: var(--text-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.prompt-lines__more { padding-left: 24px; color: var(--text-3); font-family: var(--font-sans); font-size: 11px; }

.prompt-card__foot { display: flex; align-items: center; gap: 6px; margin-top: auto; }
.prompt-card__date { color: var(--text-3); font-size: 11px; }
.prompt-card__icon-btn { width: 28px; padding: 0; border-color: transparent; color: var(--text-3); }
.prompt-card__icon-btn:hover:not(:disabled) { color: var(--text); border-color: var(--line-strong); }
.prompt-card__delete:hover:not(:disabled) { border-color: transparent; color: var(--danger); background: var(--danger-bg); }

.prompt-empty { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 36px 16px 48px; }
.prompt-empty :deep(.empty-state) { width: 100%; }

/* Right column */
.prompt-stats { display: flex; flex-direction: column; gap: 14px; }
.prompt-stats__eyebrow { color: var(--ink-soft); }
.prompt-stats__grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.prompt-stats__grid div { display: flex; flex-direction: column; gap: 2px; }
.prompt-stats__grid strong { font-size: 28px; font-weight: 600; line-height: 1.1; letter-spacing: -.01em; }
.prompt-stats__grid span { font-size: 11px; color: var(--ink-soft); }
.prompt-stats__updated { font-size: 11px; color: var(--ink-soft); }
.prompt-guide { gap: 12px; }
.prompt-guide__list { display: flex; flex-direction: column; gap: 10px; margin: 0; padding: 0; list-style: none; font-size: 12px; line-height: 1.55; color: var(--text-2); }
.prompt-guide__list li { display: flex; gap: 9px; }
.prompt-guide__list .icon { flex: none; margin-top: 3px; color: var(--text-3); }

.prompt-editor { display: flex; flex-direction: column; gap: 18px; }
.prompt-editor__counter { color: var(--text-3); font-size: 11px; font-variant-numeric: tabular-nums; }
.prompt-editor__textarea { min-height: 220px; resize: vertical; }
.prompt-editor__hint { display: flex; align-items: flex-start; gap: 8px; color: var(--text-3); font-size: 11px; }
.prompt-editor__hint .icon { flex: none; margin-top: 1px; }
@media (prefers-reduced-motion: reduce) {
  .prompt-card, .prompt-loading__dot { animation: none; transition: none; }
}
</style>
