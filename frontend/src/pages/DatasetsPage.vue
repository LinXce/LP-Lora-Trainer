<script setup lang="ts">
/**
 * Datasets: referenced local directories (never copied). Scans are explicit;
 * images are paged and only thumbnails are requested.
 */
import { computed, nextTick, ref, watch } from 'vue'
import AppIcon from '@/components/AppIcon.vue'
import EmptyState from '@/components/EmptyState.vue'
import ImageViewer, { type ViewerItem } from '@/components/ImageViewer.vue'
import ModalDialog from '@/components/ModalDialog.vue'
import PathInput from '@/components/PathInput.vue'
import SegTabs from '@/components/SegTabs.vue'
import { api } from '@/api'
import { toast, toastError } from '@/features/toast'
import { fmtNum, fmtRelative } from '@/features/format'
import type { Dataset, DatasetImage, DatasetIssueCount } from '@/types/api'

const PAGE = 48

const issueLabel: Record<DatasetIssueCount['kind'], string> = {
  corrupt: '损坏图片',
  missing_caption: '缺失 caption',
  odd_size: '异常尺寸',
  duplicate: '疑似重复',
}

const datasets = ref<Dataset[]>([])
const loaded = ref(false)
async function loadDatasets() {
  try {
    datasets.value = await api.datasets.list()
  } catch (err) {
    toastError(err, '读取数据集')
  } finally {
    loaded.value = true
  }
}
void loadDatasets()

const selectedId = ref<string | null>(null)
/* Deleting forgets LP's index for a dataset; the folder on disk is never touched. */
const removeOpen = ref(false)
const removing = ref(false)
async function removeDataset() {
  const id = selectedId.value
  if (removing.value || !id) return
  removing.value = true
  try {
    await api.datasets.remove(id)
    removeOpen.value = false
    selectedId.value = null
    images.value = []
    total.value = 0
    activeImageId.value = null
    await loadDatasets()
    toast('已删除数据集索引，磁盘文件未被改动', 'ok')
  } catch (err) {
    toastError(err, '删除数据集索引')
  } finally {
    removing.value = false
  }
}
watch(datasets, (list) => {
  if (!list.find((d) => d.dataset_id === selectedId.value)) selectedId.value = list[0]?.dataset_id ?? null
})
const selected = computed(() => datasets.value.find((d) => d.dataset_id === selectedId.value) ?? null)

/* Images (paged) */
type IssueFilter = 'all' | DatasetIssueCount['kind']
const issueFilter = ref<IssueFilter>('all')
const images = ref<DatasetImage[]>([])
const total = ref(0)
const offset = ref(0)
const loadingImages = ref(false)
let imageRequest = 0
const gallery = ref<HTMLElement | null>(null)

/* A new page (or dataset/filter) starts at the top of the gallery. */
async function scrollGalleryTop() {
  await nextTick()
  if (gallery.value) gallery.value.scrollTop = 0
}

async function loadImages() {
  const seq = ++imageRequest
  const id = selectedId.value
  if (!id) {
    images.value = []
    total.value = 0
    return
  }
  loadingImages.value = true
  try {
    const page = await api.datasets.images(id, offset.value, PAGE, issueFilter.value === 'all' ? undefined : issueFilter.value)
    if (selectedId.value !== id || seq !== imageRequest) return
    images.value = page.items
    total.value = page.total
  } catch (err) {
    if (seq === imageRequest) toastError(err, '读取图片')
  } finally {
    if (seq === imageRequest) loadingImages.value = false
  }
}
watch([selectedId, issueFilter], () => {
  offset.value = 0
  activeImageId.value = null
  void loadImages()
  void scrollGalleryTop()
})
watch(offset, () => {
  void loadImages()
  void scrollGalleryTop()
})

const pageCount = computed(() => Math.max(1, Math.ceil(total.value / PAGE)))
const pageNo = computed(() => Math.floor(offset.value / PAGE) + 1)

const issueTabs = computed(() => [
  { value: 'all' as IssueFilter, label: '全部', count: selected.value?.image_count ?? undefined },
  ...(selected.value?.issues ?? []).map((i) => ({ value: i.kind as IssueFilter, label: issueLabel[i.kind], count: i.count })),
])

/* Caption editor */
const activeImageId = ref<string | null>(null)

/* Viewer walks the images of the current page; selecting still drives the caption panel. */
const viewerItems = computed<ViewerItem[]>(() => images.value.map((im) => ({
  src: im.thumbnail_url ? im.thumbnail_url.replace(/\/thumbnail$/, '/full') : null,
  title: im.file_name,
  subtitle: `${im.width}×${im.height}`,
})))
const viewerIndex = ref<number | null>(null)
watch(viewerIndex, (i) => {
  if (i !== null && images.value[i]) activeImageId.value = images.value[i].image_id
})
watch(images, () => (viewerIndex.value = null))
function selectImage(id: string) {
  // Clicking a picture means "work on this caption"; fullscreen is the loupe button,
  // otherwise the viewer covered the caption panel.
  activeImageId.value = id
  viewerIndex.value = null
}
function openViewer(id: string) {
  activeImageId.value = id
  const i = images.value.findIndex((im) => im.image_id === id)
  viewerIndex.value = i >= 0 && images.value[i].thumbnail_url ? i : null
}
const activeImage = computed(() => images.value.find((i) => i.image_id === activeImageId.value) ?? null)
const captionDraft = ref('')
watch(activeImage, (im) => (captionDraft.value = im?.caption ?? ''))
/* An image change starts a fresh caption, so the tag picks go with it. */
watch(activeImage, () => { pickedTags.value = [] })
const captionDirty = computed(() => activeImage.value !== null && captionDraft.value !== (activeImage.value.caption ?? ''))
const savingCaption = ref(false)

async function saveCaption() {
  const ds = selectedId.value
  const im = activeImage.value
  if (!ds || !im) return
  savingCaption.value = true
  try {
    const text = captionDraft.value
    await api.datasets.saveCaption(ds, im.image_id, text)
    im.caption = text
    await loadDatasets()
    await loadImages()
    toast('caption 已保存；已有文件已备份', 'ok')
  } catch (err) {
    toastError(err, '保存 caption')
  } finally {
    savingCaption.value = false
  }
}

/* Scan / add */
const scanning = ref(false)
async function scan() {
  if (!selectedId.value) return
  scanning.value = true
  try {
    await api.datasets.scan(selectedId.value)
    await loadDatasets()
    await loadImages()
    toast('扫描完成', 'ok')
  } catch (err) {
    toastError(err, '扫描')
  } finally {
    scanning.value = false
  }
}

const addOpen = ref(false)
const addPath = ref('')
const addName = ref('')
watch(addPath, (p) => {
  if (!addName.value) addName.value = p.split(/[\\/]/).filter(Boolean).pop() ?? ''
})
async function addDataset() {
  try {
    const d = await api.datasets.add(addPath.value.trim(), addName.value.trim())
    datasets.value.push(d)
    selectedId.value = d.dataset_id
    addOpen.value = false
    addPath.value = ''
    addName.value = ''
    toast('已添加数据集引用（未复制源文件）', 'ok')
  } catch (err) {
    toastError(err, '添加数据集')
  }
}

function tagsOf(caption: string | null): string[] {
  return (caption ?? '').split(',').map((s) => s.trim()).filter(Boolean)
}
/* Chips toggle on click, but a drag that selected text keeps the selection instead of
   switching the chip — that is what makes the two behaviours coexist. */
const pickedTags = ref<string[]>([])
function toggleTag(tag: string) {
  if (!window.getSelection()?.isCollapsed) return
  pickedTags.value = pickedTags.value.includes(tag)
    ? pickedTags.value.filter((t) => t !== tag)
    : [...pickedTags.value, tag]
}
/* Copies what is picked, or the whole caption when nothing is picked. */
async function copyTags() {
  const list = pickedTags.value.length ? pickedTags.value : tagsOf(captionDraft.value)
  const text = list.join(', ')
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
    toast(`已复制 ${list.length} 个标签`, 'ok')
  } catch {
    const field = document.createElement('textarea')
    field.value = text
    document.body.append(field)
    field.select()
    document.execCommand('copy')
    field.remove()
    toast(`已复制 ${list.length} 个标签`, 'ok')
  }
}
</script>

<template>
  <div class="ds">
    <!-- Dataset list -->
    <aside class="ds__list panel">
      <div class="list__head">
        <h1 class="title-md">数据集</h1>
        <span class="spacer" />
        <button class="btn btn--sm btn--primary" @click="addOpen = true">
          <AppIcon name="plus" :size="14" />
          添加
        </button>
      </div>
      <div class="list__body scroll">
        <button
          v-for="d in datasets"
          :key="d.dataset_id"
          class="ds-item"
          :class="{ 'is-on': d.dataset_id === selectedId }"
          @click="selectedId = d.dataset_id"
        >
          <span class="ds-item__name truncate">{{ d.name }}</span>
          <span class="ds-item__path mono truncate">{{ d.path }}</span>
          <span class="ds-item__meta">
            <span class="num">{{ d.image_count === null ? '未扫描' : `${fmtNum(d.image_count)} 张` }}</span>
            <span v-if="d.issues.length" class="ds-item__warn num">
              {{ d.issues.reduce((n, i) => n + i.count, 0) }} 项问题
            </span>
          </span>
        </button>
        <p v-if="loaded && !datasets.length" class="muted list__empty">尚未添加数据集</p>
      </div>
    </aside>

    <!-- Gallery -->
    <section class="ds__main panel">
      <template v-if="selected">
        <header class="page__header">
          <div class="stack" style="gap: 4px; min-width: 0">
            <h2 class="title-lg truncate">{{ selected.name }}</h2>
            <p class="muted mono truncate selectable">{{ selected.path }}</p>
          </div>
          <span class="spacer" />
          <span class="muted">上次扫描 {{ fmtRelative(selected.scanned_at) }}</span>
          <button class="btn" :disabled="scanning" @click="scan">
            <AppIcon name="refresh" :size="15" />
            扫描
          </button>
          <button class="btn" :disabled="scanning" title="只删除 LP 的索引，不删除磁盘上的数据集文件" @click="removeOpen = true">
            <AppIcon name="trash" :size="15" />
            删除索引
          </button>
        </header>
        <div class="gallery__bar">
          <SegTabs v-model="issueFilter" :options="issueTabs" />
        </div>

        <div ref="gallery" class="page__body" :class="{ 'is-loading': loadingImages }">
          <EmptyState
            v-if="selected.image_count === null"
            icon="search"
            title="数据集尚未扫描"
            text="扫描会建立分页索引并检查损坏图片、缺失 caption、异常尺寸与疑似重复项。不会修改源文件。"
          >
            <button class="btn btn--primary" :disabled="scanning" @click="scan">开始扫描</button>
          </EmptyState>
          <EmptyState v-else-if="!images.length && !loadingImages" icon="check" title="没有符合条件的图片" />
          <div v-else class="grid">
            <button
              v-for="im in images"
              :key="im.image_id"
              class="thumb"
              :class="{ 'is-on': im.image_id === activeImageId, 'has-issue': im.issues.length }"
              :title="`${im.file_name}（点击查看 caption）`"
              @click="selectImage(im.image_id)"
            >
              <img v-if="im.thumbnail_url" :src="im.thumbnail_url" alt="" loading="lazy" decoding="async" />
              <span v-else class="thumb__ph" :style="{ aspectRatio: `${im.width} / ${im.height}` }">
                <AppIcon name="image" :size="20" />
              </span>
              <span
                class="thumb__zoom"
                role="button"
                tabindex="0"
                :aria-label="`全屏查看 ${im.file_name}`"
                :title="`全屏查看 ${im.file_name}`"
                @click.stop="openViewer(im.image_id)"
                @keydown.enter.stop.prevent="openViewer(im.image_id)"
                @keydown.space.stop.prevent="openViewer(im.image_id)"
              >
                <AppIcon name="search" :size="13" />
              </span>
              <span class="thumb__meta">
                <span class="truncate">{{ im.file_name }}</span>
                <span class="num">{{ im.width }}×{{ im.height }}</span>
              </span>
              <span v-if="im.issues.length" class="thumb__flag" :title="im.issues.map((k) => issueLabel[k]).join('、')">!</span>
            </button>
          </div>
          <ImageViewer v-model:index="viewerIndex" :items="viewerItems" />
        </div>

        <footer v-if="total > PAGE" class="pager">
          <button class="btn btn--ghost btn--sm" :disabled="pageNo <= 1" @click="offset -= PAGE">上一页</button>
          <span class="num muted">{{ pageNo }} / {{ pageCount }}</span>
          <button class="btn btn--ghost btn--sm" :disabled="pageNo >= pageCount" @click="offset += PAGE">下一页</button>
        </footer>
      </template>
      <EmptyState
        v-else
        icon="images"
        title="添加一个本地数据集目录"
        text="数据集以引用方式登记，不会复制原图片。"
      >
        <button class="btn btn--primary" @click="addOpen = true">添加数据集</button>
      </EmptyState>
    </section>

    <!-- Caption editor -->
    <aside class="ds__side">
      <section class="panel panel--pad stack" style="flex: 1; min-height: 0">
        <span class="label-pill" style="align-self: flex-start">Caption</span>
        <template v-if="activeImage">
          <div class="row">
            <strong class="truncate selectable">{{ activeImage.file_name }}</strong>
            <span class="spacer" />
            <span class="muted num">{{ activeImage.width }}×{{ activeImage.height }}</span>
          </div>
          <div v-if="activeImage.issues.length" class="row" style="flex-wrap: wrap">
            <span v-for="k in activeImage.issues" :key="k" class="chip chip--warn">{{ issueLabel[k] }}</span>
          </div>
          <textarea v-model="captionDraft" class="input mono caption" placeholder="此图片没有 caption" spellcheck="false" />
          <div class="tags">
            <span
              v-for="t in tagsOf(captionDraft)"
              :key="t"
              class="chip tag"
              :class="{ 'is-picked': pickedTags.includes(t) }"
              role="button"
              tabindex="0"
              :aria-pressed="pickedTags.includes(t)"
              :title="`点击选中「${t}」；拖动可选择文字`"
              @click="toggleTag(t)"
              @keydown.enter.prevent="toggleTag(t)"
              @keydown.space.prevent="toggleTag(t)"
            >{{ t }}</span>
          </div>
          <div class="row">
            <button class="btn btn--ghost btn--sm" :disabled="!captionDirty" @click="captionDraft = activeImage.caption ?? ''">还原</button>
            <button
              class="btn btn--ghost btn--sm"
              :disabled="!tagsOf(captionDraft).length"
              :title="pickedTags.length ? `复制选中的 ${pickedTags.length} 个标签` : '未选中标签时复制全部'"
              @click="copyTags"
            >复制标签</button>
            <span class="spacer" />
            <button class="btn btn--primary btn--sm" :disabled="!captionDirty || savingCaption" @click="saveCaption">保存</button>
          </div>
          <p class="field__hint">保存前会备份原 caption 文件，可回退。</p>
        </template>
        <p v-else class="muted">选择一张图片查看和编辑 caption。</p>
      </section>

      <section v-if="selected && selected.image_count !== null" class="panel panel--pad stack summary">
        <div class="summary__row">
          <span class="muted">图片</span>
          <strong class="num">{{ fmtNum(selected.image_count) }}</strong>
        </div>
        <div class="summary__row">
          <span class="muted">有 caption</span>
          <strong class="num">{{ fmtNum(selected.caption_count) }}</strong>
        </div>
        <div v-for="i in selected.issues" :key="i.kind" class="summary__row">
          <span class="muted">{{ issueLabel[i.kind] }}</span>
          <strong class="num warn">{{ i.count }}</strong>
        </div>
      </section>
    </aside>

    <ModalDialog v-if="addOpen" title="添加数据集" @close="addOpen = false">
      <p class="muted">引用本地目录中的图片与 caption 文件，不复制源文件。添加后需手动扫描建立索引。</p>
      <label class="field">
        <span class="field__label">数据集目录</span>
        <PathInput v-model="addPath" kind="directory" dialog-title="选择数据集目录" />
      </label>
      <label class="field">
        <span class="field__label">名称</span>
        <input v-model="addName" class="input" placeholder="用于在界面中显示" />
      </label>
      <template #footer>
        <button class="btn btn--ghost" @click="addOpen = false">取消</button>
        <button class="btn btn--primary" :disabled="!addPath.trim() || !addName.trim()" @click="addDataset">添加</button>
      </template>
    </ModalDialog>

    <ModalDialog v-if="removeOpen" title="删除数据集索引？" :width="520" @close="!removing && (removeOpen = false)">
      <p>{{ selected?.name }}</p>
      <p class="muted mono selectable">{{ selected?.path }}</p>
      <p class="muted">只删除 LP 的索引（图片列表、扫描结果、caption 副本与缩略图记录），不会修改或删除磁盘上的数据集文件；需要时重新扫描即可恢复。</p>
      <template #footer>
        <button class="btn btn--ghost" :disabled="removing" @click="removeOpen = false">取消</button>
        <button class="btn btn--primary" :disabled="removing" @click="removeDataset">{{ removing ? '删除中…' : '删除索引' }}</button>
      </template>
    </ModalDialog>
  </div>
</template>

<style scoped>
.ds {
  display: grid;
  grid-template-columns: 250px minmax(0, 1fr) 300px;
  gap: var(--gap);
  height: 100%;
}
.ds__list {
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.list__head {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 20px 16px 12px 40px;
}
.list__body {
  flex: 1;
  padding: 0 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.list__empty {
  padding: 12px 30px;
}
.ds-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 12px 14px;
  border-radius: var(--radius-md);
  text-align: left;
  transition: background 0.15s;
}
.ds-item:hover {
  background: rgba(255, 255, 255, 0.05);
}
.ds-item.is-on {
  background: var(--light);
  color: var(--ink);
}
.ds-item__name {
  font-weight: 600;
}
.ds-item__path {
  font-size: 10.5px;
  opacity: 0.6;
}
.ds-item__meta {
  display: flex;
  gap: 10px;
  font-size: 11px;
  opacity: 0.8;
}
.ds-item__warn {
  color: var(--warn);
}
.ds-item.is-on .ds-item__warn {
  color: #8a6418;
}

.ds__main {
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.gallery__bar {
  padding: 0 26px 14px;
}
.page__body.is-loading {
  opacity: 0.55;
  transition: opacity 0.2s;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(128px, 1fr));
  gap: 10px;
}
.thumb {
  position: relative;
  display: flex;
  flex-direction: column;
  border-radius: var(--radius-md);
  background: var(--panel-raised);
  overflow: hidden;
  text-align: left;
  transition: transform 0.15s, box-shadow 0.15s;
}
.thumb:hover {
  transform: translateY(-2px);
}
.thumb.is-on {
  box-shadow: 0 0 0 2px var(--light);
}
.thumb img,
.thumb__ph {
  width: 100%;
  aspect-ratio: 3 / 4;
  object-fit: cover;
}
.thumb__ph {
  display: grid;
  place-items: center;
  max-height: 180px;
  background: linear-gradient(160deg, #5a5a5a, #4a4a4a);
  color: var(--light);
}
.thumb__meta {
  display: flex;
  justify-content: space-between;
  gap: 6px;
  padding: 6px 10px 8px;
  font-size: 10.5px;
  color: var(--text-3);
}
/* Fullscreen is its own control: a plain click now only selects the caption. */
.thumb__zoom {
  position: absolute;
  top: 6px;
  right: 6px;
  display: grid;
  place-items: center;
  width: 24px;
  height: 24px;
  border-radius: 7px;
  background: color-mix(in srgb, var(--ink) 74%, transparent);
  color: #fff;
  opacity: 0.86;
  transition: opacity 0.15s, background 0.15s;
}
.thumb__zoom:hover,
.thumb__zoom:focus-visible {
  opacity: 1;
  background: var(--ink);
}
.thumb__zoom:focus-visible {
  outline: 2px solid var(--info);
  outline-offset: 1px;
}
.thumb__flag {
  position: absolute;
  top: 6px;
  left: 6px;
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--warn);
  color: var(--ink);
  font-size: 11px;
  font-weight: 800;
}
.pager {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  padding: 10px;
  border-top: 1px solid var(--line);
}

.ds__side {
  display: flex;
  flex-direction: column;
  gap: var(--gap);
  min-height: 0;
}
.caption {
  min-height: 140px;
  font-size: 12px;
}
.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  max-height: 200px;
  overflow: auto;
}
.tags .chip {
  height: 24px;
  font-size: 11px;
  cursor: pointer;
  /* Selectable text, not a button: the drag-select case is honoured in toggleTag. */
  user-select: text;
}
.tags .chip.is-picked {
  background: #fff;
  border-color: #fff;
  color: var(--ink);
}
.chip--warn {
  border-color: var(--warn);
  color: var(--warn);
}
.summary {
  gap: 8px;
}
.summary__row {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
}
.summary .warn {
  color: var(--warn);
}

@media (max-width: 1180px) {
  .ds {
    grid-template-columns: 210px minmax(0, 1fr) 260px;
  }
}
</style>
