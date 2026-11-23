import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/features/format.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } })
const exports = {}
runInNewContext(compiled.outputText, { exports })
const { fmtTrainingCount, fmtTrainingRate, fmtDuration, progressRatio } = exports

test('progress preserves a reported zero, totals without a current count, and old records without totals', () => {
  assert.equal(fmtTrainingCount(0, 1500), '0 / 1,500')
  assert.equal(fmtTrainingCount(null, 12), '未知 / 12')
  assert.equal(fmtTrainingCount(3, undefined), '3')
  assert.equal(fmtTrainingCount(undefined, null), '未知')
  assert.equal(progressRatio(0, 1500), 0)
  assert.equal(progressRatio(1500, 1500), 1)
  for (const [step, total] of [[undefined, 1], [1501, 1500], [-1, 1], [0, 0], [Infinity, 10]]) {
    assert.equal(progressRatio(step, total), null)
  }
})

test('reported throughput uses a readable unit and invalid telemetry has no units or ETA', () => {
  assert.equal(fmtTrainingRate(1 / 2.12), '2.12 s/it')
  assert.equal(fmtTrainingRate(2.5), '2.50 it/s')
  assert.equal(fmtTrainingRate(0), '0.00 it/s')
  for (const value of [null, undefined, NaN, Infinity, -1]) {
    assert.equal(fmtTrainingRate(value), '未知')
    assert.equal(fmtDuration(value), '未知')
  }
  assert.equal(fmtTrainingRate(Number.MIN_VALUE), '未知')
  assert.equal(fmtDuration(0), '0 秒')
})
