/**
 * Allowlisted pywebview bridge: native path/save dialogs and window state only.
 * Returned paths are *suggestions* — the backend still validates them.
 */

type DialogKind = 'directory' | 'file'
export type WindowAction = 'minimize' | 'toggle_maximize' | 'close'
export type ResizeEdge = 'n' | 's' | 'e' | 'w' | 'ne' | 'nw' | 'se' | 'sw'
export interface NativeWindowState {
  maximized: boolean
  resizable: boolean
}

interface BridgeApi {
  get_window_state?(): Promise<NativeWindowState>
  window_action?(action: WindowAction): Promise<NativeWindowState | null>
  begin_window_resize?(edge: ResizeEdge): Promise<void>
  pick_path(kind: DialogKind, title: string, file_types?: string[]): Promise<string | null>
  save_text_file?(suggested_name: string, text: string): Promise<string | null>
  open_in_explorer?(path: string): Promise<void>
  open_terminal?(python: string): Promise<void>
}

declare global {
  interface Window {
    pywebview?: { api: BridgeApi }
  }
}

export function hasNativeBridge(): boolean {
  return typeof window.pywebview?.api?.pick_path === 'function'
}

/** pywebview injects its API asynchronously; resolve once ready or after a short timeout. */
export function bridgeReady(timeoutMs = 1500): Promise<boolean> {
  if (hasNativeBridge()) return Promise.resolve(true)
  return new Promise((resolve) => {
    let timer: ReturnType<typeof setTimeout>
    const done = () => {
      clearTimeout(timer)
      window.removeEventListener('pywebviewready', done)
      resolve(hasNativeBridge())
    }
    window.addEventListener('pywebviewready', done, { once: true })
    timer = setTimeout(done, timeoutMs)
  })
}

export async function pickPath(
  kind: DialogKind,
  title: string,
  fileTypes?: string[],
): Promise<string | null> {
  if (!hasNativeBridge()) {
    throw new Error('\u684c\u9762\u7a97\u53e3\u539f\u751f\u9009\u62e9\u5668\u5c1a\u672a\u5c31\u7eea')
  }
  return window.pywebview!.api.pick_path(kind, title, fileTypes)
}

export async function revealPath(path: string): Promise<boolean> {
  const fn = window.pywebview?.api.open_in_explorer
  if (!fn) return false
  await fn(path)
  return true
}

/** Open a console window with one interpreter ready, for manual installs. */
export async function openTerminal(python: string): Promise<boolean> {
  const fn = window.pywebview?.api.open_terminal
  if (!fn) return false
  await fn(python)
  return true
}

export function hasWindowControls(): boolean {
  const api = window.pywebview?.api
  return typeof api?.get_window_state === 'function' && typeof api?.window_action === 'function'
}

export async function getWindowState(): Promise<NativeWindowState> {
  const api = window.pywebview?.api
  if (!api?.get_window_state) throw new Error('桌面窗口接口不可用')
  return api.get_window_state()
}

export async function windowAction(action: WindowAction): Promise<NativeWindowState | null> {
  const api = window.pywebview?.api
  if (!api?.window_action) throw new Error('桌面窗口接口不可用')
  return api.window_action(action)
}

export async function beginWindowResize(edge: ResizeEdge): Promise<void> {
  const api = window.pywebview?.api
  if (!api?.begin_window_resize) throw new Error('窗口缩放接口不可用')
  return api.begin_window_resize(edge)
}

/** Native Save As in desktop mode; browser download otherwise. Cancel is not an error. */
export async function saveTextFile(filename: string, text: string): Promise<boolean> {
  await bridgeReady()
  if (hasNativeBridge()) {
    const api = window.pywebview!.api
    if (!api.save_text_file) throw new Error('桌面保存接口不可用，请重启更新后的窗口')
    return (await api.save_text_file(filename, text)) !== null
  }
  const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  try { link.click() } finally {
    link.remove()
    window.setTimeout(() => URL.revokeObjectURL(url), 30000)
  }
  return true
}
