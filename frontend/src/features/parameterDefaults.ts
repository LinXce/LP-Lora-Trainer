import type { ParamSpec, TrainingDraft } from '@/types/api'

export type ParamValues = TrainingDraft['params']

/**
 * The shell's only engine-specific model keys: adapters whose line trains from a model
 * directory carry the base model in one of these parameters (MOSS-style engines) instead
 * of the generic base-model file. They select which model is trained, so they are context
 * for a parameter initialization, never a knob to reset.
 */
export const BASE_MODEL_PARAM_KEYS = [
  'WEIGHTS', 'WEIGHT', 'moss_model_dir', 'pretrained_model_name_or_path', 'model_dir',
] as const

/** Parameters of one engine that apply to the selected base-model type. */
export function paramsForArchitecture(params: ParamSpec[], architecture: string): ParamSpec[] {
  return params.filter((p) => !p.architectures?.length || p.architectures.includes(architecture))
}

/** The base model currently carried by parameters (the negative key when none is). */
export function baseModelFromParams(params: ParamValues): string {
  for (const key of BASE_MODEL_PARAM_KEYS) {
    const raw = params[key]
    if (typeof raw === 'string' && raw.trim()) return raw.trim().toLowerCase()
  }
  return '-'
}

/**
 * Bucket key for the parameter memory.
 *
 * Only the generic base-model file (and the fact that an optional base-model type
 * owns its model directory) identifies the bucket. Model-like parameter values
 * (WEIGHTS, moss_model_dir, pretrained_model_name_or_path, ...) must never feed
 * back into this key: changing one would rebuild the parameter set, clear the
 * parameter, and immediately change the key again — an infinite watch loop.
 */
export function parameterBaseModelKey(baseModelOptional: boolean, baseModelPath: string): string {
  if (baseModelOptional) return '-'
  const selected = baseModelPath.trim()
  return selected ? selected.toLowerCase() : '-'
}

/**
 * The engine defaults for one base-model type: exactly what a freshly selected engine
 * starts from. Values in `keep` win over the defaults, so callers can hold on to data
 * that is context (the base model) rather than a tuning knob.
 */
export function defaultParams(params: ParamSpec[], architecture: string, keep: ParamValues = {}): ParamValues {
  const out: ParamValues = {}
  for (const p of paramsForArchitecture(params, architecture)) {
    out[p.key] = Object.prototype.hasOwnProperty.call(keep, p.key) ? keep[p.key] : p.default
  }
  return out
}

/**
 * Snapshot of the engine defaults the form last saw. Engine upgrades can change
 * defaults without touching the staged draft; comparing a value with this snapshot
 * lets the form update untouched defaults while preserving user edits.
 */
const DEFAULTS_PREFIX = 'lp:param-defaults:'

/**
 * One-time defaults from before the adapter started reading the engine config.
 * Only fields whose old default actually changed need an entry; the first
 * capabilities load after this migration replaces those exact values.
 */
export const LEGACY_ENGINE_DEFAULTS: Record<string, ParamValues> = {
  anima: {
    network_dim: 16,
    network_alpha: 1,
    learning_rate: 0.000002,
    lr_scheduler: 'constant',
    lr_warmup_steps: 0,
    max_train_steps: 1600,
    max_train_epochs: null,
    save_every_n_epochs: null,
    checkpointing_epochs: null,
    use_vae_cache: false,
    use_text_cache: false,
    caption_dropout_rate: 0,
    use_shuffled_caption_variants: false,
    cache_llm_adapter_outputs: false,
    attn_mode: '',
    blocks_to_swap: null,
    vae_chunk_size: null,
    sample_ratio: null,
    path_pattern: '',
    persistent_data_loader_workers: null,
    dataloader_pin_memory: null,
    network_train_unet_only: null,
    compile_dynamic_seq: false,
  },
  moss_image: { moss_cache_text_encoder: 'fp8' },
}

function defaultsStorageKey(installationId: string, architecture: string): string {
  return `${DEFAULTS_PREFIX}${encodeURIComponent(installationId)}:${encodeURIComponent(architecture)}`
}

/** The defaults snapshot for one engine instance + base-model type, if readable. */
export function readDefaultsSnapshot(installationId: string, architecture: string): ParamValues | null {
  if (!installationId || !architecture || typeof window === 'undefined') return null
  try {
    const raw = window.localStorage.getItem(defaultsStorageKey(installationId, architecture))
    if (!raw) return null
    const parsed = JSON.parse(raw) as unknown
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null
    return parsed as ParamValues
  } catch {
    return null
  }
}

export function writeDefaultsSnapshot(
  installationId: string,
  architecture: string,
  defaults: ParamValues,
): void {
  if (!installationId || !architecture || typeof window === 'undefined') return
  try {
    window.localStorage.setItem(
      defaultsStorageKey(installationId, architecture),
      JSON.stringify(defaults),
    )
  } catch {
    // A full/disabled storage must not break the form.
  }
}

/** Default values keyed by parameter key for the visible parameter set. */
export function defaultsFromSpecs(specs: ParamSpec[]): ParamValues {
  const out: ParamValues = {}
  for (const p of specs) out[p.key] = p.default
  return out
}

/**
 * Keys whose current value still equals the previous engine default. Those are
 * safe to move to the new default; anything the user changed is left alone.
 */
export function defaultsToApply(
  current: ParamValues,
  previousDefaults: ParamValues | null,
  specs: ParamSpec[],
): ParamValues {
  if (!previousDefaults) return {}
  const out: ParamValues = {}
  for (const p of specs) {
    if (!Object.prototype.hasOwnProperty.call(previousDefaults, p.key)) continue
    if (Object.is(current[p.key], previousDefaults[p.key])) out[p.key] = p.default
  }
  return out
}
