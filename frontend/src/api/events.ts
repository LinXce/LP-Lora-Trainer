/**
 * Server-sent events with reconnect. Callers must fetch a fresh snapshot in
 * `onOpen` before applying increments, so a reconnect never leaves stale state.
 */
import { API_BASE, initializeSession } from './http'
import type { ServerEvent } from '@/types/api'

export type StreamStatus = 'connecting' | 'open' | 'retrying' | 'closed'

interface StreamOptions {
  onEvent: (ev: ServerEvent) => void
  onOpen?: () => void
  onStatus?: (status: StreamStatus) => void
}

export function openEventStream(opts: StreamOptions): () => void {
  let source: EventSource | null = null
  let retryTimer: number | undefined
  let attempt = 0
  let closed = false

  const connect = async () => {
    if (closed) return
    opts.onStatus?.(attempt === 0 ? 'connecting' : 'retrying')
    try {
      await initializeSession()
    } catch {
      if (closed) return
      attempt += 1
      opts.onStatus?.('retrying')
      retryTimer = window.setTimeout(() => void connect(), Math.min(15000, 500 * 2 ** Math.min(attempt, 5)))
      return
    }
    if (closed) return
    source = new EventSource(`${API_BASE}/events`)

    source.onopen = () => {
      attempt = 0
      opts.onStatus?.('open')
      opts.onOpen?.()
    }
    source.onmessage = (msg) => {
      try {
        opts.onEvent(JSON.parse(msg.data) as ServerEvent)
      } catch {
        // Ignore malformed frames; raw logs remain on disk.
      }
    }
    source.onerror = () => {
      source?.close()
      source = null
      if (closed) return
      attempt += 1
      opts.onStatus?.('retrying')
      const delay = Math.min(15000, 500 * 2 ** Math.min(attempt, 5))
      retryTimer = window.setTimeout(() => void connect(), delay)
    }
  }

  void connect()

  return () => {
    closed = true
    window.clearTimeout(retryTimer)
    source?.close()
    opts.onStatus?.('closed')
  }
}
