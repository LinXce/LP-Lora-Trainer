/**
 * Sample images carry their step/epoch and creation time in the file name:
 * `{name}_{000015|e000003}_{prompt:02d}_{YYYYMMDDHHMMSS}[_{seed}][_{n:03d}].png`
 * (musubi-tuner and sd-scripts share this scheme). The backend's `step` field
 * is empty for samples, so grouping and ordering are derived here.
 */
import type { Artifact } from '@/types/api'

const SAMPLE_NAME = /_(e?)(\d{6})_(\d{2})_(\d{14})(?:_\d+)*$/

export interface SampleInfo {
  /** Stable group key: one group per step (or epoch) of one training run. */
  group: string
  label: string
  /** Sort key within a run (step or epoch number; unknown sorts last). */
  order: number
  /** Index of the prompt within the sample batch. */
  prompt: number
  /** Creation time in ms parsed from the file name; 0 when unknown. */
  time: number
}

function stem(name: string) {
  const dot = name.lastIndexOf('.')
  return dot > 0 ? name.slice(0, dot) : name
}

function parseStamp(s: string): number {
  const t = new Date(+s.slice(0, 4), +s.slice(4, 6) - 1, +s.slice(6, 8), +s.slice(8, 10), +s.slice(10, 12), +s.slice(12, 14)).getTime()
  return Number.isNaN(t) ? 0 : t
}

export function sampleInfo(a: Artifact): SampleInfo {
  const m = SAMPLE_NAME.exec(stem(a.file_name))
  if (m) {
    const n = Number(m[2])
    return {
      group: `${m[1] ? 'e' : 's'}${n}`,
      label: m[1] ? `epoch ${n}` : `step ${n}`,
      order: n,
      prompt: Number(m[3]),
      time: parseStamp(m[4]),
    }
  }
  // `created_at` is re-stamped whenever the backend re-registers a file, so it
  // says nothing about when the image was made: unnamed-by-scheme files rank last.
  if (a.step !== null) return { group: `s${a.step}`, label: `step ${a.step}`, order: a.step, prompt: 0, time: 0 }
  return { group: 'other', label: '其他', order: Number.POSITIVE_INFINITY, prompt: 0, time: 0 }
}

/** Newest first: by creation time, then by step/epoch, then by prompt. */
export function newestSamples(artifacts: Artifact[]): Artifact[] {
  return artifacts
    .filter((a) => (a.kind === 'sample' || a.kind === 'video') && a.complete)
    .map((a) => ({ a, i: sampleInfo(a) }))
    .sort((x, y) => y.i.time - x.i.time || y.i.order - x.i.order || y.i.prompt - x.i.prompt || y.a.file_name.localeCompare(x.a.file_name))
    .map(({ a }) => a)
}

export interface SampleGroup {
  key: string
  label: string
  order: number
  items: Artifact[]
}

/** Samples grouped per step/epoch, oldest group first, prompts in order inside a group. */
export function groupSamples(artifacts: Artifact[]): SampleGroup[] {
  const groups = new Map<string, SampleGroup & { prompts: Map<string, number> }>()
  for (const a of artifacts) {
    if (a.kind !== 'sample' && a.kind !== 'video') continue
    const info = sampleInfo(a)
    let g = groups.get(info.group)
    if (!g) groups.set(info.group, (g = { key: info.group, label: info.label, order: info.order, items: [], prompts: new Map() }))
    g.items.push(a)
    g.prompts.set(a.artifact_id, info.prompt)
  }
  return [...groups.values()]
    .map(({ prompts, ...g }) => ({
      ...g,
      items: g.items.sort((x, y) => (prompts.get(x.artifact_id)! - prompts.get(y.artifact_id)!) || x.file_name.localeCompare(y.file_name)),
    }))
    .sort((x, y) => (x.order === y.order ? 0 : x.order < y.order ? -1 : 1))
}
