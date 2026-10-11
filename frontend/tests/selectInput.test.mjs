import test, { after } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { runInThisContext } from 'node:vm'
import { JSDOM } from 'jsdom'
import { parse, compileScript } from '@vue/compiler-sfc'
import ts from 'typescript'

// Mount the real SFC: compiler-only assertions miss Vue's Boolean prop casting.
const dom = new JSDOM('<!doctype html><html><body></body></html>', { pretendToBeVisual: true })
for (const key of ['window', 'document', 'Node', 'Element', 'HTMLElement', 'SVGElement', 'MutationObserver', 'Event', 'FocusEvent', 'KeyboardEvent']) {
  globalThis[key] = dom.window[key]
}
globalThis.requestAnimationFrame = dom.window.requestAnimationFrame.bind(dom.window)
globalThis.cancelAnimationFrame = dom.window.cancelAnimationFrame.bind(dom.window)
globalThis.getComputedStyle = dom.window.getComputedStyle.bind(dom.window)
dom.window.HTMLElement.prototype.scrollIntoView = function () {}
const require = createRequire(import.meta.url)
const { createApp, h, ref, nextTick } = require('vue')
const source = readFileSync(new URL('../src/components/SelectInput.vue', import.meta.url), 'utf8')
const { descriptor } = parse(source)
const script = compileScript(descriptor, { id: 'select-regression', inlineTemplate: true })
const compiled = ts.transpileModule(
  script.content.replace("import AppIcon from './AppIcon.vue'", 'const AppIcon = { render: () => null }'),
  { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } },
).outputText
const module = { exports: {} }
runInThisContext(`(function(require, module, exports) { ${compiled}\n })`)(require, module, module.exports)
const SelectInput = module.exports.default
const settle = async () => { await nextTick(); await Promise.resolve(); await nextTick() }
function mountSelect(props, options, parent = document.body) {
  const host = document.createElement('div')
  parent.append(host)
  const app = createApp({ render: () => h(SelectInput, props(), {
    default: () => options().map(o => h('option', { value: o.value, disabled: o.disabled }, o.label ?? String(o.value))),
  }) })
  app.mount(host)
  return {
    host,
    trigger: () => host.querySelector('[role="combobox"]'),
    popup: () => document.getElementById(host.querySelector('[role="combobox"]').getAttribute('aria-controls')),
    destroy: () => { app.unmount(); host.remove() },
  }
}
const captionOptions = () => ['.txt', '.caption', '.cap'].map(value => ({ value }))
after(() => dom.window.close())

test('value + change enum selects retain their default and commit the clicked option', async () => {
  const value = ref('.txt')
  const events = []
  const view = mountSelect(() => ({ value: value.value, onChange: event => {
    events.push([event.target.tagName, event.target.value])
    value.value = event.target.value
  } }), captionOptions)
  try {
    await settle()
    assert.equal(view.trigger().textContent, '.txt')
    view.trigger().click(); await settle()
    view.popup().querySelectorAll('[role="option"]')[1].click(); await settle()
    assert.equal(value.value, '.caption')
    assert.equal(view.trigger().textContent, '.caption')
    assert.deepEqual(events, [['SELECT', '.caption']])
    assert.equal(view.trigger().getAttribute('aria-expanded'), 'false')
    assert.equal(document.activeElement, view.trigger())
    value.value = '.cap'; await settle()
    assert.equal(view.trigger().textContent, '.cap')
    assert.equal(view.host.querySelector('select').value, '.cap')
  } finally { view.destroy() }
})

test('v-model dataset selects still commit values through update:modelValue', async () => {
  const model = ref('dataset-one')
  const view = mountSelect(() => ({ modelValue: model.value, 'onUpdate:modelValue': value => { model.value = value } }),
    () => [{ value: '', label: 'Choose dataset', disabled: true }, { value: 'dataset-one' }, { value: 'dataset-two' }])
  try {
    await settle()
    assert.equal(view.trigger().textContent, 'dataset-one')
    view.trigger().click(); await settle()
    view.popup().querySelectorAll('[role="option"]')[2].click(); await settle()
    assert.equal(model.value, 'dataset-two')
    assert.equal(view.trigger().textContent, 'dataset-two')
  } finally { view.destroy() }
})

test('unbound controls show a placeholder rather than selecting the false option', async () => {
  const view = mountSelect(() => ({ placeholder: 'Choose an option' }), () => [{ value: false }, { value: true }])
  try {
    await settle()
    assert.equal(view.trigger().textContent, 'Choose an option')
    assert.equal(view.host.querySelector('select').value, '')
  } finally { view.destroy() }
})

test('explicit false and zero values are not mistaken for absent props', async () => {
  for (const [prop, value] of [['value', false], ['value', 0], ['modelValue', false], ['modelValue', 0]]) {
    const view = mountSelect(() => ({ [prop]: value }), () => [{ value: false, label: 'False' }, { value: 0, label: 'Zero' }])
    try {
      await settle()
      assert.equal(view.trigger().textContent, value === false ? 'False' : 'Zero', `${prop}=${value}`)
    } finally { view.destroy() }
  }
})

test('late-loaded options synchronize and keyboard selection skips disabled options', async () => {
  const value = ref('.txt')
  const options = ref([])
  const view = mountSelect(() => ({ value: value.value, onChange: e => { value.value = e.target.value } }), () => options.value)
  try {
    await settle()
    options.value = [{ value: '.txt' }, { value: '.caption', disabled: true }, { value: '.cap' }]
    await settle()
    assert.equal(view.trigger().textContent, '.txt')
    view.trigger().click(); await settle()
    view.trigger().dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }))
    view.trigger().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
    await settle()
    assert.equal(value.value, '.cap')
    assert.equal(view.trigger().textContent, '.cap')
  } finally { view.destroy() }
})


test('CUDA and mirror lists stay inside the owning modal, including without native popovers', async () => {
  const dialog = document.createElement('dialog')
  dialog.setAttribute('open', '')
  document.body.append(dialog)
  const value = ref('cu124')
  const view = mountSelect(() => ({ modelValue: value.value, 'onUpdate:modelValue': v => { value.value = v } }),
    () => ['cu124', 'cu128', 'cu130', 'cu132', 'existing'].map(value => ({ value })), dialog)
  try {
    await settle()
    view.trigger().click(); await settle()
    assert.equal(view.popup().closest('dialog'), dialog)
    assert.equal(view.popup().querySelectorAll('[role="option"]').length, 5)
    view.popup().querySelectorAll('[role="option"]')[1].click(); await settle()
    assert.equal(value.value, 'cu128')
    assert.equal(view.trigger().textContent, 'cu128')
  } finally { view.destroy(); dialog.remove() }
})

test('Escape closes an open list without bubbling to the dialog; a second Escape is delegated', async () => {
  const dialog = document.createElement('dialog')
  document.body.append(dialog)
  const events = []
  dialog.addEventListener('keydown', e => { if (e.key === 'Escape') events.push(e.defaultPrevented) })
  const view = mountSelect(() => ({ value: '.txt' }), captionOptions, dialog)
  try {
    await settle()
    view.trigger().click(); await settle()
    view.trigger().dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }))
    await settle()
    assert.equal(view.trigger().getAttribute('aria-expanded'), 'false')
    assert.deepEqual(events, [])
    view.trigger().dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true }))
    assert.deepEqual(events, [false])
  } finally { view.destroy(); dialog.remove() }
})

test('disabled controls cannot open and disabling an open control closes it', async () => {
  const disabled = ref(true)
  const view = mountSelect(() => ({ value: '.txt', disabled: disabled.value }), captionOptions)
  try {
    await settle()
    view.trigger().click(); await settle()
    assert.equal(view.trigger().getAttribute('aria-expanded'), 'false')
    disabled.value = false; await settle()
    view.trigger().click(); await settle()
    assert.equal(view.trigger().getAttribute('aria-expanded'), 'true')
    disabled.value = true; await settle()
    assert.equal(view.trigger().getAttribute('aria-expanded'), 'false')
  } finally { view.destroy() }
})


test('external selection changes update keyboard focus in an already open list', async () => {
  const value = ref('.txt')
  const view = mountSelect(() => ({ modelValue: value.value, 'onUpdate:modelValue': v => { value.value = v } }), captionOptions)
  try {
    await settle(); view.trigger().click(); await settle()
    value.value = '.caption'; await settle()
    const active = view.trigger().getAttribute('aria-activedescendant')
    assert.equal(document.getElementById(active).textContent, '.caption')
    view.trigger().dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true }))
    view.trigger().dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
    await settle()
    assert.equal(value.value, '.cap')
  } finally { view.destroy() }
})

test('outside pointer events close the list without changing its value', async () => {
  const value = ref('.txt')
  const view = mountSelect(() => ({ value: value.value }), captionOptions)
  try {
    await settle(); view.trigger().click(); await settle()
    document.body.dispatchEvent(new Event('pointerdown', { bubbles: true }))
    await settle()
    assert.equal(view.trigger().getAttribute('aria-expanded'), 'false')
    assert.equal(view.trigger().textContent, '.txt')
  } finally { view.destroy() }
})
