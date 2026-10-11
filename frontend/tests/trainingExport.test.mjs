import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/features/trainingExport.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
const exports = {}
runInNewContext(compiled, { exports })
const { serializeTrainingDraft, exportFilename, matchImportedDataset, restoreTaskConfiguration, snapshotTrainingExport } = exports
const plain = value => JSON.parse(JSON.stringify(value))

test('portable parameter export preserves raw values and imports cleanly as JSON', () => {
  const draft = { name: ' 测试 ', base_model_path: 'D:/models/base.safetensors', params: { resolution: '512,\n768', enabled: false, rank: 0 } }
  const text = serializeTrainingDraft(draft, 'musubi_tuner', 'D:/data/set')
  const result = JSON.parse(text)
  assert.equal(result.format, 'lp-training-parameters')
  assert.equal(result.version, 1)
  assert.equal(result.engine_id, 'musubi_tuner')
  assert.equal(result.dataset_path, 'D:/data/set')
  assert.deepEqual(plain(result.params), plain(draft.params))
  assert.equal(result.params.resolution, '512,\n768')
  assert.ok(text.endsWith('\n'))
})

test('export filename is safe for Windows reserved names and invalid characters', () => {
  assert.equal(exportFilename('CON'), 'training.params.json')
  assert.equal(exportFilename('my:run. '), 'my_run.params.json')
  assert.ok(exportFilename('x'.repeat(200)).length <= 113)
})

test('dataset import matches stable id then normalized path and otherwise clears selection', () => {
  const datasets = [{ dataset_id: 'a', path: 'D:\\Data\\Set' }, { dataset_id: 'b', path: '/data/other' }]
  assert.equal(matchImportedDataset({ dataset_id: 'b', dataset_path: 'D:/Data/Set' }, datasets), 'b')
  assert.equal(matchImportedDataset({ dataset_path: 'd:/data/set/' }, datasets), 'a')
  assert.equal(matchImportedDataset({ dataset_path: 'E:/missing' }, datasets), null)
})

const historyConfig = { name: 'history', installation_id: 'old', architecture: 'krea2', engine_id: 'musubi_tuner',
  dataset_id: 'old-data', dataset_path: 'H:/data/images', base_model_path: 'H:/models/base.safetensors',
  output_dir: 'H:/output', params: { rank: 32, seed: 0, enabled: false, resolution: '512,1024' } }
test('history reuse copies every parameter and matches a renamed same-type engine and dataset', () => {
  const restored = restoreTaskConfiguration(historyConfig,
    [{ installation_id: 'kohya', engine_id: 'kohya', is_default: true },
     { installation_id: 'new', engine_id: 'musubi_tuner', is_default: true }],
    [{ dataset_id: 'new-data', path: 'h:\\data\\images\\' }])
  assert.equal(restored.draft.installation_id, 'new')
  assert.equal(restored.draft.dataset_id, 'new-data')
  assert.deepEqual(plain(restored.draft.params), historyConfig.params)
  assert.equal(restored.draft.output_dir, 'H:/output')
  assert.equal(restored.missingEngine, false)
  assert.equal(restored.missingDataset, false)
  restored.draft.params.rank = 1
  assert.equal(historyConfig.params.rank, 32)
})
test('missing historical registrations preserve paths and parameters without choosing the wrong engine', () => {
  const restored = restoreTaskConfiguration(historyConfig,
    [{ installation_id: 'old', engine_id: 'kohya', is_default: true }], [])
  assert.equal(restored.draft.installation_id, '')
  assert.equal(restored.draft.dataset_id, '')
  assert.equal(restored.missingEngine, true)
  assert.equal(restored.missingDataset, true)
  assert.deepEqual(plain(restored.draft.params), historyConfig.params)
})
test('history prefers the exact installation rather than a newer default', () => {
  const restored = restoreTaskConfiguration(historyConfig,
    [{ installation_id: 'new', engine_id: 'musubi_tuner', is_default: true },
     { installation_id: 'old', engine_id: 'musubi_tuner', is_default: false }], [])
  assert.equal(restored.draft.installation_id, 'old')
})


test('export dialog snapshots its target and strips history-only wire metadata', () => {
  const original = { ...historyConfig, params: { ...historyConfig.params }, format: 'lp-training-parameters', version: 1 }
  const snapshot = snapshotTrainingExport(original, original.engine_id, original.dataset_path)
  original.name = 'changed'
  original.params.rank = 1
  assert.equal(snapshot.name, 'history')
  assert.equal(snapshot.params.rank, 32)
  assert.equal(snapshot.params.seed, 0)
  assert.equal(snapshot.params.enabled, false)
  assert.equal('format' in snapshot, false)
  assert.equal('version' in snapshot, false)
  assert.equal(snapshotTrainingExport(original, null).dataset_path, null)
  assert.equal(snapshotTrainingExport(original, null).engine_id, null)
})
