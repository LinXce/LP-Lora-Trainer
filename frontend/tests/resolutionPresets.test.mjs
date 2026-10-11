import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/features/resolutionPresets.ts', import.meta.url), 'utf8')
const exports = {}
runInNewContext(ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText, { exports })
const { hasResolution, toggleResolution } = exports

test('manual square resolutions use the adapter syntax', () => {
  for (const text of ['512x512', '512X512', '512*512', '512 512', '768x768,512x512', '768x768\n512x512']) {
    assert.equal(hasResolution(text, 512), true, text)
  }
  for (const text of ['', null, 512, '512', '512x768', 'broken']) {
    assert.equal(hasResolution(text, 512), false)
  }
})

test('switches append to the shared text and preserve custom and invalid input', () => {
  const value = toggleResolution('960x544\ninvalid', 1024)
  assert.equal(value, '960x544\ninvalid\n1024x1024')
  assert.equal(hasResolution(value, 1024), true)
  assert.equal(toggleResolution(value, 1024), '960x544\ninvalid')
})

test('turning off a preset removes equivalent duplicates only', () => {
  assert.equal(toggleResolution('512x512,512*512\n512 512\n512x768\n2048X2048', 512), '512x768\n2048X2048')
  assert.equal(toggleResolution('512x512', 512), '')
  assert.equal(toggleResolution('', 512), '512x512')
})

test('all eight presets can be edited and toggled independently', () => {
  const presets = [512, 640, 768, 896, 1024, 1280, 1536, 2048]
  let value = ''
  for (const size of presets) value = toggleResolution(value, size)
  for (const size of presets) assert.equal(hasResolution(value, size), true)
  value = toggleResolution(value, 896)
  assert.equal(hasResolution(value, 896), false)
  assert.equal(hasResolution(value, 1024), true)
})
