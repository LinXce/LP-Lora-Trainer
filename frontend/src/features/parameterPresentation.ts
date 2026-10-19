import type { ParamGroup, ParamSpec } from '@/types/api'

/** Multiple tabs expose a single draft.params[key], never a second value. */
export function paramGroups(p: ParamSpec): ParamGroup[] {
  return [...new Set([p.group, ...(p.groups ?? [])])]
}

/* Ordering is declared by the adapter: `display_order` first, then declaration order.
   The shell knows no engine-specific parameter names — an earlier whitelist here made
   the UI carry engine knowledge and had to be mirrored in every adapter anyway. */
function paramOrder(p: ParamSpec): number {
  return p.display_order ?? 100
}

/** Section order inside a tab. Save is first: it is the setting users adjust most,
 *  and the engine-side save/log grouping is split so the two never compete. */
function sectionOrder(section: string): number {
  if (/保存/.test(section)) return 0
  if (/模型|编码器/.test(section)) return 10
  // Memory knobs sit right after saving: they are adjusted while watching checkpoints.
  if (/精度|显存|内存|加速/.test(section)) return 50
  if (/数据集/.test(section)) return 100
  if (/训练/.test(section)) return 200
  if (/网络/.test(section)) return 300
  if (/优化/.test(section)) return 400
  if (/时间步|损失/.test(section)) return 600
  if (/采样/.test(section)) return 700
  if (/日志|上传/.test(section)) return 800
  if (/元数据/.test(section)) return 900
  return 1000
}

export function parameterSections(params: ParamSpec[], tab: ParamGroup): [string, ParamSpec[]][] {
  const sections = new Map<string, ParamSpec[]>()
  for (const p of params) {
    if (!paramGroups(p).includes(tab)) continue
    const section = p.section || '其他参数'
    if (!sections.has(section)) sections.set(section, [])
    sections.get(section)!.push(p)
  }
  return [...sections.entries()]
    .map(([section, fields]) => [section, [...fields].sort((a, b) => paramOrder(a) - paramOrder(b))] as [string, ParamSpec[]])
    .sort((a, b) => sectionOrder(a[0]) - sectionOrder(b[0]) || paramOrder(a[1][0]) - paramOrder(b[1][0]))
}

const typeLabels: Record<ParamSpec['type'], string> = {
  int: '整数', float: '小数', string: '文本/路径', bool: '开关', enum: '选项',
}

function formatDefault(p: ParamSpec): string {
  if (typeof p.default === 'boolean') return p.default ? '开启' : '关闭'
  if (p.default === null) return '未设置（留空不指定）'
  if (p.default === '') return '留空'
  return p.options?.find(o => o.value === String(p.default))?.label || String(p.default)
}

export function parameterMeta(p: ParamSpec, architecture: string): string {
  const required = p.required_for?.includes(architecture)
  const parts = [p.unsupported_reason ? '自动设置 / 不可编辑' : required ? '必填' : '可选', p.list_arg ? '列表（空格或换行分隔）' : typeLabels[p.type]]
  parts.push(`界面默认：${formatDefault(p)}`)
  if (p.min !== undefined || p.max !== undefined) parts.push(`范围：${p.min ?? '不限'}–${p.max ?? '不限'}`)
  if (p.step !== undefined) parts.push(`输入步长：${p.step}`)
  if (p.native === false) parts.push('数据集 / 流程参数')
  else parts.push(`引擎键：${p.block ? p.block + '.' : ''}${p.native_key || p.key}`)
  if (required && p.default !== null && p.default !== '') parts.push('已有默认值，无需重复填写')
  return parts.join(' · ')
}
