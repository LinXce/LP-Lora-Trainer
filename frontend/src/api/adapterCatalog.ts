/** The catalog is required: never replace it with hard-coded engine names. */
import { ApiError, request } from './http'
import type { AdapterCatalog } from '@/types/api'

export async function loadAdapterCatalog(): Promise<AdapterCatalog> {
  try {
    return await request<AdapterCatalog>('/adapters')
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      throw new ApiError(
        '当前后台版本不兼容，缺少适配器目录接口，无法确认引擎类型。请先等待或停止安装和训练，通过“退出应用”结束后台，再重新打开 EXE；若旧后台仍驻留，请先确认没有训练或安装任务，再正常退出旧应用并重新启动。',
        404,
        'incompatible_backend',
      )
    }
    throw err
  }
}
