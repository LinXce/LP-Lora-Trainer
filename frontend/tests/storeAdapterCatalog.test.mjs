import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { runInNewContext } from 'node:vm'
import ts from 'typescript'

function transpile(path) {
  return ts.transpileModule(readFileSync(new URL(path, import.meta.url), 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText
}

const storeSource = transpile('../src/features/store.ts')
const formatSource = transpile('../src/features/format.ts')
class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.status = status
  }
}

function harness() {
  const format = {}
  runInNewContext(formatSource, { exports: format })
  const api = {
    adapters: { list: async () => ({ adapters: [], errors: [] }) },
    engines: { list: async () => [] },
    tasks: { list: async () => [] },
    system: { status: async () => ({}) },
  }
  const timers = new Map()
  let nextTimer = 0
  let stream
  const exports = {}
  const imports = {
    vue: { reactive: value => value },
    '@/api': { api },
    '@/api/demo': { isDemoMode: () => false },
    '@/api/http': { ApiError },
    '@/api/events': {
      openEventStream: handlers => {
        stream = handlers
        return () => { stream = undefined }
      },
    },
    './format': format,
  }
  runInNewContext(storeSource, {
    exports,
    Error,
    require: name => {
      assert.ok(name in imports, `Unexpected import: ${name}`)
      return imports[name]
    },
    document: { hidden: false },
    window: {
      setTimeout: fn => {
        const id = ++nextTimer
        timers.set(id, fn)
        return id
      },
      clearTimeout: id => timers.delete(id),
    },
  })
  return {
    ...exports,
    api,
    async start() {
      exports.startLiveUpdates()
      await new Promise(resolve => setImmediate(resolve))
    },
    emit(event) {
      assert.ok(stream, 'Live updates must be started first')
      stream.onEvent(event)
      const scheduled = [...timers.entries()]
      timers.clear()
      for (const [, flush] of scheduled) flush()
    },
  }
}

const catalog = {
  adapters: [{ engine_id: 'fourth_trainer', display_name: 'Fourth Trainer' }],
  errors: [],
}
const installation = {
  installation_id: 'copied-source',
  engine_id: 'fourth_trainer',
  candidate_engines: ['fourth_trainer'],
  state: 'discovered',
}

test('successful task refresh cannot hide an adapter catalog failure', async () => {
  const h = harness()
  const error = new ApiError('incompatible backend: missing adapter catalog', 404)
  h.api.adapters.list = async () => { throw error }
  assert.equal(await h.refreshEngines(), false)
  await h.refreshTasks()
  assert.equal(h.store.loadError, null)
  assert.equal(h.store.engineLoadError, error.message)
  assert.equal(h.store.enginesLoaded, true)
  assert.equal(h.store.adaptersLoaded, false)
})

test('engine events without a loaded catalog do not create unclassified records', async () => {
  const h = harness()
  h.api.adapters.list = async () => { throw new ApiError('missing catalog', 404) }
  await h.start()
  h.emit({ kind: 'engine.updated', installation })
  assert.equal(h.store.engines.length, 0)
  h.emit({ kind: 'task.updated', task: { task_id: 'task-1', state: 'running' } })
  assert.equal(h.store.tasks[0].task_id, 'task-1')
  assert.equal(h.store.engineLoadError, 'missing catalog')
  h.stopLiveUpdates()
})

test('catalog recovery supports a new adapter and preserves type across stale events', async () => {
  const h = harness()
  h.api.adapters.list = async () => { throw new ApiError('missing catalog', 404) }
  await h.start()
  h.emit({ kind: 'engine.updated', installation })
  h.api.adapters.list = async () => catalog
  h.api.engines.list = async () => [installation]
  assert.equal(await h.refreshEngines(), true)
  assert.equal(h.store.engineLoadError, null)
  assert.equal(h.store.adaptersLoaded, true)
  assert.equal(h.store.engines[0].engine_id, 'fourth_trainer')
  h.emit({
    kind: 'engine.updated',
    installation: { ...installation, engine_id: null, candidate_engines: [], state: 'ready' },
  })
  assert.equal(h.store.engines[0].engine_id, 'fourth_trainer')
  assert.equal(h.store.engines[0].state, 'ready')
  h.emit({
    kind: 'engine.updated',
    installation: { ...installation, installation_id: 'second-copy', engine_id: null },
  })
  assert.equal(h.store.engines[0].installation_id, 'second-copy')
  assert.equal(h.store.engines[0].engine_id, 'fourth_trainer')
  h.stopLiveUpdates()
})

test('failed refresh retains the last valid catalog and engines until recovery', async () => {
  const h = harness()
  h.api.adapters.list = async () => catalog
  h.api.engines.list = async () => [installation]
  assert.equal(await h.refreshEngines(), true)
  const previousAdapters = h.store.adapters
  const previousEngines = h.store.engines
  h.api.engines.list = async () => { throw new ApiError('server failure', 500) }
  assert.equal(await h.refreshEngines(), false)
  await h.refreshTasks()
  assert.equal(h.store.adapters, previousAdapters)
  assert.equal(h.store.engines, previousEngines)
  assert.equal(h.store.adaptersLoaded, true)
  assert.equal(h.store.engineLoadError, 'server failure')
  h.api.engines.list = async () => [installation]
  assert.equal(await h.refreshEngines(), true)
  assert.equal(h.store.engineLoadError, null)
})

test('task deletion events remove history in other clients without affecting the remaining task', async () => {
  const h = harness()
  h.api.tasks.list = async () => [{ task_id: 'old', state: 'succeeded' }, { task_id: 'live', state: 'running' }]
  await h.start()
  h.emit({ kind: 'task.updated', task: { task_id: 'old', state: 'succeeded', name: 'updated snapshot' } })
  h.emit({ kind: 'task.deleted', task_id: 'old' })
  assert.equal(h.store.tasks.length, 1)
  assert.equal(h.store.tasks[0].task_id, 'live')
  h.emit({ kind: 'task.deleted', task_id: 'absent' })
  assert.equal(h.store.tasks.length, 1)
  h.stopLiveUpdates()
})

test('in-flight snapshots and buffered updates cannot resurrect deleted training history', async () => {
  const h = harness()
  h.api.tasks.list = async () => [{ task_id: 'old', state: 'succeeded' }]
  await h.start()
  h.removeTask('old')
  h.emit({ kind: 'task.updated', task: { task_id: 'old', state: 'succeeded' } })
  assert.equal(h.store.tasks.length, 0)
  await h.refreshTasks()
  assert.equal(h.store.tasks.length, 0)
  h.stopLiveUpdates()
})
