/**
 * Thin HTTP client for the local backend. All business operations go through
 * here; the frontend never executes commands or touches files directly.
 */

export const API_BASE = '/api/v1'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code: string | null = null,
  ) {
    super(message)
    this.name = 'ApiError'
  }

  /** Backend unreachable or not yet started (network failure or proxy 5xx without body). */
  get isUnreachable(): boolean {
    return this.status === 0
  }
}

/** A hung local backend must never freeze the UI: every request is bounded. */
const REQUEST_TIMEOUT_MS = 20_000

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'
  body?: unknown
  query?: Record<string, string | number | boolean | null | undefined>
  signal?: AbortSignal
}

let handshake: Promise<void> | null = null

export function initializeSession(): Promise<void> {
  if (handshake) return handshake
  handshake = (async () => {
    let res: Response
    try {
      res = await fetch(`${API_BASE}/session`, { method: 'POST', credentials: 'same-origin' })
    } catch {
      throw new ApiError('无法连接本地后端服务', 0)
    }
    if (!res.ok) throw new ApiError('无法建立本机会话', res.status >= 500 ? 0 : res.status)
  })().finally(() => { handshake = null })
  return handshake
}

export async function request<T>(path: string, opts: RequestOptions = {}, retry = true): Promise<T> {
  const url = new URL(API_BASE + path, window.location.href)
  for (const [k, v] of Object.entries(opts.query ?? {})) {
    if (v !== null && v !== undefined) url.searchParams.set(k, String(v))
  }

  const headers: Record<string, string> = { Accept: 'application/json' }
  if (opts.body !== undefined) headers['Content-Type'] = 'application/json'

  // Merge the caller's cancellation with our own timeout: whichever fires
  // first aborts the internal controller. Cancelling through the caller's
  // signal keeps its original AbortError; only our timer becomes a timeout.
  const controller = new AbortController()
  const caller = opts.signal
  let timedOut: boolean = false
  const timer = window.setTimeout(() => {
    timedOut = true
    controller.abort()
  }, REQUEST_TIMEOUT_MS)
  const forwardAbort = () => controller.abort()
  if (caller) {
    if (caller.aborted) controller.abort()
    else caller.addEventListener('abort', forwardAbort)
  }

  let res: Response
  try {
    res = await fetch(url, {
      method: opts.method ?? 'GET',
      credentials: 'same-origin',
      headers,
      body: opts.body === undefined ? undefined : JSON.stringify(opts.body),
      signal: controller.signal,
    })
  } catch (err) {
    if ((err as Error).name === 'AbortError') {
      if (timedOut && !caller?.aborted) throw new ApiError('请求超时，本地后端未在 20 秒内响应', 0)
      throw err
    }
    throw new ApiError('无法连接本地后端服务', 0)
  } finally {
    window.clearTimeout(timer)
    caller?.removeEventListener('abort', forwardAbort)
  }

  if (res.status === 401 && retry) {
    await initializeSession()
    return request<T>(path, opts, false)
  }

  if (res.status === 204) return undefined as T

  const text = await res.text()
  const data: unknown = text ? safeJson(text) : null

  if (!res.ok) {
    const detail = (data as { detail?: unknown; code?: string } | null) ?? null
    const message =
      typeof detail?.detail === 'string' ? detail.detail : Array.isArray(detail?.detail) ? detail.detail.map((d: { msg?: string }) => d.msg ?? '参数无效').join('；') : `请求失败（HTTP ${res.status}）`
    // Dev proxy returns 5xx with empty body when the backend is down.
    const status = res.status >= 500 && !text ? 0 : res.status
    throw new ApiError(message, status, detail?.code ?? null)
  }
  if (text && data === null) throw new ApiError('后端返回了无效响应', 502)
  return data as T
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}
