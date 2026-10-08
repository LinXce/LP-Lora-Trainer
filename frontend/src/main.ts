import { createApp, watch } from 'vue'
import App from './App.vue'
import { router } from './router'
import { store } from './features/store'
import './styles/tokens.css'
import './styles/base.css'

/** Upper bound for the boot screen: a slow or unreachable backend must not hide the app (and its error state). */
const BOOT_TIMEOUT_MS = 10000

function dismissBoot() {
  const boot = document.getElementById('boot')
  if (!boot || boot.classList.contains('is-done')) return
  boot.classList.add('is-done')
  boot.addEventListener('transitionend', () => boot.remove(), { once: true })
  window.setTimeout(() => boot.remove(), 600)
}

const app = createApp(App).use(router)
router.isReady().then(() => {
  app.mount('#app')
  // Fully started = first task and engine snapshots answered (successfully or not).
  const loaded = () => store.tasksLoaded && store.enginesLoaded
  if (loaded()) return dismissBoot()
  const stop = watch(loaded, (ready) => {
    if (!ready) return
    dismissBoot()
    stop()
  })
  window.setTimeout(dismissBoot, BOOT_TIMEOUT_MS)
})
