/**
 * Staged "new training" form, persisted by the backend.
 *
 * The form is remembered between visits — base-model type, engine instance and
 * every parameter — and because the draft lives in the app's data root, the
 * desktop window, a browser tab and a second window all see the same staged
 * form. Parameters are remembered *per base-model type* so switching base model
 * (and back) does not throw away the other one's values.
 *
 * A draft is convenience data only: the backend bounds it, and real submission
 * still goes through full validation. An absent, unreadable or older entry
 * simply yields an empty form.
 */
import { api } from '@/api'
import type { TrainingDraft } from '@/types/api'

/** Parameter values as staged for one base-model type. */
export type DraftParams = Record<string, string | number | boolean | null>

/** Drafts staged by an earlier build in local storage; migrated once. */
const LEGACY_STORAGE_KEY = 'lp.training.draft'
const LEGACY_VERSION = 1
const MAX_TEXT = 2048
const MAX_ARCHITECTURES = 64

export interface StoredDraft {
  draft: TrainingDraft
  /** Parameter values remembered per base-model type. */
  paramsByArchitecture: Record<string, DraftParams>
  updatedAt: string | null
}

export function emptyDraft(): TrainingDraft {
  return { name: '', installation_id: '', architecture: '', base_model_path: '', dataset_id: '', output_dir: '', params: {} }
}

function asText(value: unknown): string {
  return typeof value === 'string' ? value.slice(0, MAX_TEXT) : ''
}

function asParams(value: unknown): DraftParams {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const out: DraftParams = {}
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    if (!key || key.length > 200) continue
    if (typeof raw === 'string') out[key] = raw.slice(0, MAX_TEXT)
    else if (typeof raw === 'number') { if (Number.isFinite(raw)) out[key] = raw }
    else if (typeof raw === 'boolean') out[key] = raw
    else if (raw === null) out[key] = null
  }
  return out
}

function asArchitectureParams(value: unknown): Record<string, DraftParams> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const out: Record<string, DraftParams> = {}
  for (const [architecture, params] of Object.entries(value as Record<string, unknown>).slice(0, MAX_ARCHITECTURES)) {
    if (architecture) out[architecture.slice(0, 64)] = asParams(params)
  }
  return out
}

function asDraft(value: unknown): TrainingDraft | null {
  if (!value || typeof value !== 'object') return null
  const stored = value as Record<string, unknown>
  return {
    name: asText(stored.name),
    installation_id: asText(stored.installation_id),
    architecture: asText(stored.architecture),
    base_model_path: asText(stored.base_model_path),
    dataset_id: asText(stored.dataset_id),
    output_dir: asText(stored.output_dir),
    params: asParams(stored.params),
  }
}

/** Read a draft staged by an earlier frontend-only build, if one is present. */
function readLegacyDraft(): StoredDraft | null {
  try {
    const raw = window.localStorage.getItem(LEGACY_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as { version?: unknown; draft?: unknown; paramsByArchitecture?: unknown }
    if (!parsed || typeof parsed !== 'object' || parsed.version !== LEGACY_VERSION) return null
    const draft = asDraft(parsed.draft)
    if (!draft) return null
    return { draft, paramsByArchitecture: asArchitectureParams(parsed.paramsByArchitecture), updatedAt: null }
  } catch {
    return null
  }
}

function clearLegacyDraft(): void {
  try {
    window.localStorage.removeItem(LEGACY_STORAGE_KEY)
  } catch {
    // Ignore: an unreadable store is already treated as empty.
  }
}

/** The shared staged form; null when nothing usable is staged. */
export async function loadDraft(): Promise<StoredDraft | null> {
  const response = await api.training.draft()
  const remote = asDraft(response?.draft)
  const stored: StoredDraft | null = remote
    ? { draft: remote, paramsByArchitecture: asArchitectureParams(response.params_by_architecture), updatedAt: response.updated_at ?? null }
    : null
  const legacy = readLegacyDraft()
  if (!legacy) return stored
  // One-time migration: keep whichever side already holds something.
  if (stored && hasContent(stored.draft)) {
    clearLegacyDraft()
    return stored
  }
  try {
    await saveDraft(legacy.draft, legacy.paramsByArchitecture)
  } catch {
    return legacy // the shared store is unavailable; do not lose what was staged
  }
  clearLegacyDraft()
  return legacy
}

export async function saveDraft(draft: TrainingDraft, paramsByArchitecture: Record<string, DraftParams>): Promise<void> {
  await api.training.saveDraft({ ...draft, params_by_architecture: paramsByArchitecture })
}

export async function clearDraft(): Promise<void> {
  clearLegacyDraft()
  await api.training.clearDraft()
}

/**
 * True when the user actually entered something worth mentioning.
 * Engine defaults alone must not make an untouched form look restored.
 */
export function hasContent(draft: TrainingDraft): boolean {
  return Boolean(draft.name.trim() || draft.base_model_path.trim() || draft.output_dir.trim())
}
