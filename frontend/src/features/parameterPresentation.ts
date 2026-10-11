import type { ParamGroup, ParamSpec } from '@/types/api'

/** Multiple tabs expose a single draft.params[key], never a second value. */
export function paramGroups(p: ParamSpec): ParamGroup[] {
  return [...new Set([p.group, ...(p.groups ?? [])])]
}

const modelFileKeys = new Set([
  'vae', 'vae_dir', 'vae_path', 'h3_video_vae', 'h3_audio_vae',
  'hv_text_encoder1', 'hv_text_encoder2', 'hv15_text_encoder', 'hv15_byt5',
  'wan_t5', 'flux2_text_encoder', 'kontext_text_encoder1', 'kontext_text_encoder2',
  'qwen_text_encoder', 'zimage_text_encoder', 'ideogram_text_encoder', 'h3_text_encoder',
  'krea2_text_encoder', 'krea2_tokenizer', 'kandinsky_text_encoder_qwen', 'kandinsky_text_encoder_clip',
  'fpack_text_encoder1', 'fpack_text_encoder2', 'hv15_image_encoder', 'wan_clip', 'fpack_image_encoder',
])

function paramOrder(p: ParamSpec): number {
  if (p.display_order !== undefined && p.display_order !== 100) return p.display_order
  if (modelFileKeys.has(p.key)) return p.key.includes('vae') ? 0 : 10
  return 100
}

function sectionOrder(section: string): number {
  if (/模型|编码器/.test(section)) return 0
  if (/数据集/.test(section)) return 100
  if (/优化/.test(section)) return 400
  if (/网络/.test(section)) return 300
  if (/训练/.test(section)) return 200
  if (/精度|显存|内存|加速/.test(section)) return 500
  if (/时间步|损失/.test(section)) return 600
  if (/保存|采样/.test(section)) return 700
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
    .sort((a, b) => {
      const modelFirst = Number(!a[1].some(p => modelFileKeys.has(p.key))) - Number(!b[1].some(p => modelFileKeys.has(p.key)))
      return modelFirst || sectionOrder(a[0]) - sectionOrder(b[0]) || paramOrder(a[1][0]) - paramOrder(b[1][0])
    })
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
