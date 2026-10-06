import type { InstallationState, TaskState, VerificationState } from '@/types/api'

export type Tone = 'neutral' | 'ok' | 'warn' | 'danger' | 'info' | 'live'

export const taskStateMeta: Record<TaskState, { label: string; tone: Tone }> = {
  draft: { label: '草稿', tone: 'neutral' },
  validating: { label: '校验中', tone: 'info' },
  queued: { label: '排队', tone: 'neutral' },
  preparing: { label: '准备中', tone: 'info' },
  running: { label: '运行中', tone: 'live' },
  stopping: { label: '停止中', tone: 'warn' },
  stopped: { label: '已停止', tone: 'neutral' },
  succeeded: { label: '成功', tone: 'ok' },
  failed: { label: '失败', tone: 'danger' },
  connection_lost: { label: '连接丢失', tone: 'warn' },
}

export const ACTIVE_TASK_STATES: TaskState[] = ['validating', 'queued', 'preparing', 'running', 'stopping']

export const installStateMeta: Record<InstallationState, { label: string; tone: Tone }> = {
  discovered: { label: '已发现 · 环境未就绪', tone: 'warn' },
  preparing: { label: '环境准备中', tone: 'info' },
  ready: { label: '就绪', tone: 'ok' },
  failed: { label: '安装失败', tone: 'danger' },
  missing: { label: '目录缺失', tone: 'danger' },
}

export const verificationMeta: Record<VerificationState, { label: string; tone: Tone }> = {
  verified: { label: '已验证', tone: 'ok' },
  experimental: { label: '实验性', tone: 'info' },
  unverified: { label: '未经验证', tone: 'neutral' },
}

export const engineNames: Record<string, string> = {
  kohya: 'Kohya',
  ai_toolkit: 'AI Toolkit',
}

export function engineName(id: string | null): string {
  if (!id) return '未识别'
  return engineNames[id] ?? id
}

export const UNKNOWN = '未知'

export function fmtNum(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return UNKNOWN
  return v.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits })
}

export function fmtLoss(v: number | null | undefined): string {
  if (v === null || v === undefined) return UNKNOWN
  return v.toFixed(4)
}

export function fmtDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return UNKNOWN
  const s = Math.max(0, Math.round(seconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) return `${h} 小时 ${m} 分`
  if (m > 0) return `${m} 分 ${sec} 秒`
  return `${sec} 秒`
}

export function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined) return UNKNOWN
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i++
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`
}

export function fmtTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}

export function fmtRelative(iso: string | null | undefined): string {
  if (!iso) return '—'
  const diff = (Date.now() - new Date(iso).getTime()) / 1000
  if (diff < 60) return '刚刚'
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`
  return `${Math.floor(diff / 86400)} 天前`
}

export function progressRatio(step: number | null, total: number | null): number | null {
  if (step === null || !total) return null
  return Math.min(1, Math.max(0, step / total))
}

export function shortCommit(c: string | null): string | null {
  return c ? c.slice(0, 7) : null
}
