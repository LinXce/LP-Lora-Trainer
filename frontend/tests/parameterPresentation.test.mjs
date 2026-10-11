import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'

// Test the actual TS helpers with existing dependencies; no test framework added.
const source = readFileSync(new URL('../src/features/parameterPresentation.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText
const exports = {}
runInNewContext(compiled, { exports })
const { paramGroups, parameterSections, parameterMeta } = exports
const plain = value => JSON.parse(JSON.stringify(value))
const param = (key, overrides = {}) => ({ key, label: key, type: 'string', group: 'basic', section: '训练', default: '', ...overrides })

test('additional tabs are deduplicated and reference the same field', () => {
  const vae = param('vae', { group: 'native', groups: ['basic', 'native', 'basic'] })
  assert.deepEqual(plain(paramGroups(vae)), ['native', 'basic'])
  const basic = parameterSections([vae], 'basic')[0][1][0]
  const native = parameterSections([vae], 'native')[0][1][0]
  assert.strictEqual(basic, native)
  assert.strictEqual(basic, vae)
  assert.equal(parameterSections([vae], 'advanced').length, 0)
})

test('model files first, related sections together, stable ordering', () => {
  const fields = [
    param('log', { section: '日志' }),
    param('lr', { section: '优化器' }),
    param('rank', { section: '网络' }),
    param('steps'), param('epochs'),
    param('wan_t5', { section: '模型与编码器', display_order: 10 }),
    param('vae', { section: '模型与编码器', display_order: 0 }),
    param('resolution', { section: '数据集' }),
  ]
  const sections = parameterSections(fields, 'basic')
  assert.deepEqual(plain(sections.map(([name]) => name)), ['模型与编码器', '数据集', '训练', '网络', '优化器', '日志'])
  assert.deepEqual(plain(sections[0][1].map(p => p.key)), ['vae', 'wan_t5'])
  assert.deepEqual(plain(sections[2][1].map(p => p.key)), ['steps', 'epochs'])
  assert.equal(fields[0].key, 'log')
})

test('required status follows the selected architecture', () => {
  const vae = param('vae', { required_for: ['wan'] })
  assert.match(parameterMeta(vae, 'wan'), /^必填/)
  assert.match(parameterMeta(vae, 'hidream_o1'), /^可选/)
  assert.match(parameterMeta(param('task', { default: 't2v', required_for: ['wan'] }), 'wan'), /已有默认值/)
  assert.match(parameterMeta(param('preset', { unsupported_reason: 'automatic' }), 'wan'), /^自动设置/)
})

test('default descriptions preserve null, zero, false and enum labels', () => {
  assert.match(parameterMeta(param('empty', { default: null }), ''), /未设置（留空不指定）/)
  assert.match(parameterMeta(param('seed', { type: 'int', default: 0, min: 0, step: 1 }), ''), /界面默认：0/)
  assert.match(parameterMeta(param('flag', { type: 'bool', default: false }), ''), /界面默认：关闭/)
  assert.match(parameterMeta(param('mode', { type: 'enum', default: 'auto', options: [{ value: 'auto', label: '自动' }] }), ''), /界面默认：自动/)
})

test('native block, ranges and list input descriptions are visible', () => {
  const text = parameterMeta(param('vae_path', { native_key: 'vae', block: 'model', min: 0, max: 8, step: 1, list_arg: true }), '')
  assert.match(text, /引擎键：model.vae/)
  assert.match(text, /范围：0–8/)
  assert.match(text, /输入步长：1/)
  assert.match(text, /空格或换行分隔/)
  const workflow = parameterMeta(param('cache', { native: false }), '')
  assert.match(workflow, /数据集 \/ 流程参数/)
  assert.doesNotMatch(workflow, /引擎键/)
})
