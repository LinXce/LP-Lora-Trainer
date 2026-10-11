import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/features/format.ts', import.meta.url), 'utf8')
const exports = {}
runInNewContext(ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText, { exports })
const { configureAdapters, engineName, isEngineId, knownEngineIds, candidateFallback } = exports
const plain = value => JSON.parse(JSON.stringify(value))

test('a new adapter is recognized from backend metadata, not built-in names', () => {
  configureAdapters([{ engine_id: 'fourth_trainer', display_name: 'Fourth Trainer' }])
  assert.equal(isEngineId('fourth_trainer'), true)
  assert.equal(engineName('fourth_trainer'), 'Fourth Trainer')
  assert.equal(isEngineId('kohya'), false)
  assert.deepEqual(plain(knownEngineIds(['fourth_trainer', 'unknown', 'fourth_trainer'])), ['fourth_trainer'])
  assert.equal(candidateFallback({ engine_id: null, candidate_engines: ['fourth_trainer'] }), 'fourth_trainer')
})

test('refresh replaces stale catalog entries and preserves ambiguous candidates', () => {
  configureAdapters([{ engine_id: 'other', display_name: '' }, { engine_id: 'next', display_name: 'Next' }])
  assert.equal(isEngineId('fourth_trainer'), false)
  assert.equal(engineName('other'), 'other')
  assert.equal(candidateFallback({ candidate_engines: ['other', 'next'] }), null)
  assert.equal(candidateFallback({ engine_id: 'next', candidate_engines: ['other'] }), 'next')
  configureAdapters([])
  assert.deepEqual(plain(knownEngineIds(['other'])), [])
})
