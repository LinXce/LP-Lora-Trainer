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
