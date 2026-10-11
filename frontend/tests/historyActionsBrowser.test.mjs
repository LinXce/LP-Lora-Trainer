import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { resolve, sep, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawn } from 'node:child_process'
import { createServer } from 'vite'

const browser = [process.env.LP_TEST_BROWSER,
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
].find(path => path && existsSync(path))

test('real browser: task context menu, config export/reuse, history deletion and preserved results', { skip: !browser, timeout: 60000 }, async () => {
  const root = fileURLToPath(new URL('..', import.meta.url))
  const server = await createServer({ root, server: { host: '127.0.0.1', port: 0, strictPort: false, open: false } })
  const profile = await mkdtemp(join(tmpdir(), 'lp-history-browser-'))
  let child
  try {
    await server.listen()
    const address = server.httpServer.address()
    const url = `http://127.0.0.1:${address.port}/tests/fixtures/historyActions.html`
    const output = await new Promise((done, reject) => {
      child = spawn(browser, ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
        '--disable-extensions', `--user-data-dir=${profile}`, '--window-size=1000,900', '--virtual-time-budget=12000', '--dump-dom', url], { windowsHide: true })
      let stdout = '', stderr = ''
      child.stdout.on('data', chunk => { stdout += chunk })
      child.stderr.on('data', chunk => { stderr += chunk })
      child.on('error', reject)
      child.on('close', code => code === 0 ? done(stdout) : reject(new Error(`Browser exited ${code}: ${stderr}`)))
    })
    const result = output.match(/<pre id="result"[^>]*data-status="([^"]+)"[^>]*>([\s\S]*?)<\/pre>/)
    assert.ok(result, 'Browser did not return the history actions fixture')
    assert.equal(result[1], 'passed', result[2])
    console.log('Browser history scenarios:', result[2])
  } finally {
    if (child && child.exitCode === null) child.kill()
    await server.close()
    // Resolve and verify the temporary directory before recursive cleanup.
    assert.ok(resolve(profile).startsWith(resolve(tmpdir()) + sep))
    await rm(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 })
  }
})
