import type { TrainingDraft, TaskConfiguration, TrainingExportSource } from '@/types/api'

/** Portable form export. IDs are hints; engine/architecture identify its meaning. */
export function serializeTrainingDraft(draft: TrainingDraft, engineId: string | null, datasetPath?: string): string {
  return JSON.stringify({
    format: 'lp-training-parameters', version: 1,
    engine_id: engineId, dataset_path: datasetPath ?? null,
    ...draft, params: { ...draft.params },
  }, null, 2) + '\n'
}

/** Freeze the exact form/history target when its export dialog opens. */
export function snapshotTrainingExport(draft: TrainingDraft, engineId: string | null, datasetPath?: string | null): TrainingExportSource {
  return {
    name: draft.name, output_name: draft.output_name, installation_id: draft.installation_id, architecture: draft.architecture,
    base_model_path: draft.base_model_path, dataset_id: draft.dataset_id, output_dir: draft.output_dir,
    params: { ...draft.params }, engine_id: engineId, dataset_path: datasetPath ?? null,
  }
}

export function exportFilename(name: string, suffix = '.params.json'): string {
  const leaf = name.trim().replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_').replace(/[. ]+$/, '').slice(0, 100)
  const safe = !leaf || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(leaf) ? 'training' : leaf
  return safe + suffix
}

export function matchImportedDataset(source: Record<string, unknown>, datasets: { dataset_id: string; path: string }[]): string | null {
  const byId = datasets.find(d => d.dataset_id === source.dataset_id)
  if (byId) return byId.dataset_id
  const normalize = (value: string) => value.trim().replaceAll('\\', '/').replace(/\/+$/, '').toLowerCase()
  if (typeof source.dataset_path !== 'string' || !source.dataset_path.trim()) return null
  return datasets.find(d => normalize(d.path) === normalize(source.dataset_path as string))?.dataset_id ?? null
}

/** Rebind portable history to current registrations; never substitute another engine type. */
export function restoreTaskConfiguration(config: TaskConfiguration,
  engines: { installation_id: string; engine_id: string | null; is_default: boolean }[],
  datasets: { dataset_id: string; path: string }[],
): { draft: TrainingDraft; missingEngine: boolean; missingDataset: boolean } {
  const compatible = engines.filter(e => e.engine_id === config.engine_id)
  const engine = compatible.find(e => e.installation_id === config.installation_id) ??
    compatible.find(e => e.is_default) ?? compatible[0]
  const datasetId = matchImportedDataset(config as unknown as Record<string, unknown>, datasets)
  return {
    draft: { name: config.name, output_name: config.output_name || config.name, installation_id: engine?.installation_id ?? '', architecture: config.architecture,
      base_model_path: config.base_model_path, dataset_id: datasetId ?? '', output_dir: config.output_dir,
      params: { ...config.params } },
    missingEngine: !engine, missingDataset: !datasetId,
  }
}
