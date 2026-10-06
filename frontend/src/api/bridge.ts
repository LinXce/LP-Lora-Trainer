/**
 * Allowlisted pywebview bridge: native path dialogs and window state only.
 * Returned paths are *suggestions* — the backend still validates them.
 */

type DialogKind = 'directory' | 'file'

interface BridgeApi {
  pick_path(kind: DialogKind, title: string, file_types?: string[]): Promise<string | null>
  open_in_explorer?(path: string): Promise<void>
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
    const done = () => resolve(hasNativeBridge())
    window.addEventListener('pywebviewready', done, { once: true })
    setTimeout(done, timeoutMs)
  })
}

export async function pickPath(
  kind: DialogKind,
  title: string,
  fileTypes?: string[],
): Promise<string | null> {
  if (!hasNativeBridge()) return null
  return window.pywebview!.api.pick_path(kind, title, fileTypes)
}

export async function revealPath(path: string): Promise<boolean> {
  const fn = window.pywebview?.api.open_in_explorer
  if (!fn) return false
  await fn(path)
  return true
}
