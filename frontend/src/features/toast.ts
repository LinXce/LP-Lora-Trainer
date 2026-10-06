import { reactive } from 'vue'
import { ApiError } from '@/api/http'

export interface Toast {
  id: number
  tone: 'info' | 'ok' | 'danger'
  text: string
}

let seq = 0
export const toasts = reactive<Toast[]>([])

export function toast(text: string, tone: Toast['tone'] = 'info', ms = 3800): void {
  const id = ++seq
  toasts.push({ id, tone, text })
  if (toasts.length > 4) toasts.shift()
  setTimeout(() => dismissToast(id), ms)
}

export function dismissToast(id: number): void {
  const i = toasts.findIndex((t) => t.id === id)
  if (i >= 0) toasts.splice(i, 1)
}

/** Report a failed operation honestly — never swallow it into a fake success. */
export function toastError(err: unknown, prefix?: string): void {
  const msg = err instanceof ApiError || err instanceof Error ? err.message : String(err)
  toast(prefix ? `${prefix}：${msg}` : msg, 'danger', 5200)
}
