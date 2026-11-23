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
runInNewContext(compiled, { exports })
const { BASE_MODEL_PARAM_KEYS, baseModelFromParams, defaultParams, parameterBaseModelKey, paramsForArchitecture } = exports
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

test('restored draft wins even when an explicit value equals a former default', () => {
  const specs = [param('rank', { default: 4 }), param('compile', { default: true }),
    param('seed', { default: 42 }), param('optional', { default: 10 }), param('new', { default: 8 })]
  assert.deepEqual(plain(defaultParams(specs, 'wan', { rank: 16, compile: false, seed: 0, optional: null })),
    { rank: 16, compile: false, seed: 0, optional: null, new: 8 })
})

test('Kohaku Base and SFT under one weights root have independent drafts', () => {
  assert.notEqual(baseModelFromParams({ WEIGHTS: 'D:/models', VARIANT: 'base' }),
    baseModelFromParams({ WEIGHTS: 'D:/models', VARIANT: 'sft' }))
})
