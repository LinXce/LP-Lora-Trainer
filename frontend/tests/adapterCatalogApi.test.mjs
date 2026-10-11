import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'

const source = readFileSync(new URL('../src/api/adapterCatalog.ts', import.meta.url), 'utf8')
const output = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText

class ApiError extends Error {
  constructor(message, status, code = null) {
    super(message)
    this.status = status
    this.code = code
  }
}

function client(request) {
  const exports = {}
  runInNewContext(output, {
    exports,
    require: name => {
      assert.equal(name, './http')
      return { ApiError, request }
    },
  })
  return exports.loadAdapterCatalog
}

test('catalog API accepts additional adapters without a frontend whitelist', async () => {
  const catalog = { adapters: [{ engine_id: 'fourth_trainer', display_name: 'Fourth Trainer' }], errors: [] }
  const load = client(async path => {
    assert.equal(path, '/adapters')
    return catalog
  })
  assert.equal(await load(), catalog)
})

test('missing catalog on a resident old backend gives an actionable compatibility error', async () => {
  const load = client(async () => { throw new ApiError('Not Found', 404) })
  await assert.rejects(load, err => {
    assert.equal(err.status, 404)
    assert.equal(err.code, 'incompatible_backend')
    assert.match(err.message, /后台版本不兼容/)
    assert.match(err.message, /安装和训练/)
    assert.match(err.message, /退出应用/)
    assert.match(err.message, /EXE/)
    assert.doesNotMatch(err.message, /Not Found/)
    return true
  })
})

test('network, authorization, server and unexpected errors are not hidden', async () => {
  for (const err of [
    new ApiError('offline', 0),
    new ApiError('unauthorized', 401),
    new ApiError('server error', 500),
    new Error('unexpected failure'),
  ]) {
    const load = client(async () => { throw err })
    await assert.rejects(load, actual => actual === err)
  }
})
