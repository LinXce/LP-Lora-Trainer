import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'

// Test the actual TS helpers with existing dependencies; no test framework added.
const source = readFileSync(new URL('../src/features/parameterDefaults.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText
const exports = {}
const storageMemory = new Map()
const testWindow = {
  localStorage: {
    getItem: key => storageMemory.has(key) ? storageMemory.get(key) : null,
    setItem: (key, value) => storageMemory.set(key, value),
  },
}
runInNewContext(compiled, { exports, window: testWindow })
const { BASE_MODEL_PARAM_KEYS, baseModelFromParams, defaultParams, parameterBaseModelKey, paramsForArchitecture, defaultsFromSpecs, defaultsToApply, LEGACY_ENGINE_DEFAULTS, readDefaultsSnapshot, writeDefaultsSnapshot } = exports
const plain = value => JSON.parse(JSON.stringify(value))
const param = (key, overrides = {}) => ({ key, label: key, type: 'string', group: 'basic', section: '训练', default: '', ...overrides })

test('only parameters of the selected base-model type are initialized', () => {
  const specs = [
    param('rank', { default: 16 }),
    param('vae', { default: '', architectures: ['wan'] }),
    param('llm', { default: 'x', architectures: ['hidream'] }),
  ]
  assert.deepEqual(plain(paramsForArchitecture(specs, 'wan').map(p => p.key)), ['rank', 'vae'])
  assert.deepEqual(plain(defaultParams(specs, 'wan')), { rank: 16, vae: '' })
})

test('defaults replace edited knobs while the base model stays as context', () => {
  const specs = [
    param('WEIGHTS', { default: '' }),
    param('rank', { default: 16 }),
    param('lr', { default: 0.0002 }),
  ]
  const edited = { WEIGHTS: 'D:\\models\\moss', rank: 128, lr: 0.5, unknown_key: 'drop me' }
  const keep = {}
  for (const key of BASE_MODEL_PARAM_KEYS) if (key in edited) keep[key] = edited[key]
  assert.deepEqual(plain(defaultParams(specs, 'moss_image', keep)), {
    WEIGHTS: 'D:\\models\\moss', rank: 16, lr: 0.0002,
  })
})

test('a base model carried by a parameter identifies the staged bucket', () => {
  assert.equal(baseModelFromParams({ WEIGHTS: ' D:/Models/MOSS-Image ' }), 'd:/models/moss-image')
  assert.equal(baseModelFromParams({ moss_model_dir: 'C:\\moss' }), 'c:\\moss')
  assert.equal(baseModelFromParams({ rank: 16 }), '-')
})

test('model-like parameter values are not part of the bucket key', () => {
  // The key only follows the generic base-model file (or stays '-' for optional
  // base-model types). WEIGHTS / moss_model_dir / pretrained_model_name_or_path
  // are values inside the parameter set; feeding them back into the key caused an
  // infinite watch loop when the user picked a model directory.
  assert.equal(parameterBaseModelKey(true, ''), '-')
  assert.equal(parameterBaseModelKey(true, 'D:/models/MOSS-Image-Base'), '-')
  assert.equal(parameterBaseModelKey(false, ' D:/Base.safetensors '), 'd:/base.safetensors')
  assert.equal(parameterBaseModelKey(false, ''), '-')
})

test('only untouched defaults follow an engine default change', () => {
  const specs = [
    param('network_dim', { default: 32 }),
    param('seed', { default: 0 }),
    param('extra', { default: 'new' }),
  ]
  assert.deepEqual(plain(defaultsFromSpecs(specs)), {
    network_dim: 32, seed: 0, extra: 'new',
  })
  assert.deepEqual(plain(defaultsToApply(
    { network_dim: 16, seed: 1234, extra: 'mine' },
    { network_dim: 16, seed: 0, extra: 'old' },
    specs,
  )), { network_dim: 32 })
  assert.deepEqual(plain(defaultsToApply({}, null, specs)), {})
  assert.equal(LEGACY_ENGINE_DEFAULTS.anima.network_dim, 16)
  assert.equal(LEGACY_ENGINE_DEFAULTS.anima.lr_scheduler, 'constant')
})

test('defaults snapshots round-trip through local storage', () => {
  writeDefaultsSnapshot('engine-1', 'anima', { network_dim: 32, lr_scheduler: 'cosine' })
  assert.deepEqual(plain(readDefaultsSnapshot('engine-1', 'anima')), {
    network_dim: 32, lr_scheduler: 'cosine',
  })
  assert.equal(readDefaultsSnapshot('engine-2', 'anima'), null)
})
