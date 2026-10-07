import { createRouter, createWebHashHistory, type RouteRecordRaw } from 'vue-router'

export interface NavMeta {
  title: string
  icon: string
  /** Pinned to the bottom of the rail (like the lone dot above the avatar in the reference). */
  bottom?: boolean
}

declare module 'vue-router' {
  interface RouteMeta extends Partial<NavMeta> {}
}

export const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'overview',
    component: () => import('@/pages/OverviewPage.vue'),
    meta: { title: '概览', icon: 'grid' },
  },
  {
    path: '/engines',
    name: 'engines',
    component: () => import('@/pages/EnginesPage.vue'),
    meta: { title: '引擎管理', icon: 'cpu' },
  },
  {
    path: '/datasets',
    name: 'datasets',
    component: () => import('@/pages/DatasetsPage.vue'),
    meta: { title: '数据集', icon: 'images' },
  },
  {
    path: '/train',
    name: 'train',
    component: () => import('@/pages/NewTrainingPage.vue'),
    meta: { title: '新建训练', icon: 'plus' },
  },
  {
    path: '/tasks/:taskId?',
    name: 'tasks',
    component: () => import('@/pages/TasksPage.vue'),
    meta: { title: '训练任务', icon: 'activity' },
  },
  {
    path: '/results/:taskId?',
    name: 'results',
    component: () => import('@/pages/ResultsPage.vue'),
    meta: { title: '训练结果', icon: 'layers' },
  },
  {
    path: '/terminal',
    name: 'terminal',
    component: () => import('@/pages/TerminalPage.vue'),
    meta: { title: '终端', icon: 'terminal' },
  },
  {
    path: '/settings',
    name: 'settings',
    component: () => import('@/pages/SettingsPage.vue'),
    meta: { title: '设置', icon: 'settings', bottom: true },
  },
  { path: '/:pathMatch(.*)*', redirect: '/' },
]

// Hash history: the static bundle is loaded from the local server without SPA fallback.
export const router = createRouter({
  history: createWebHashHistory(),
  routes,
})

router.afterEach((to) => {
  document.title = to.meta.title ? `${to.meta.title} · LP LoRA Trainer` : 'LP LoRA Trainer'
})
