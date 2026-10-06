# 前端

Vue 3 + TypeScript + Vite，构建后由本地 Python 服务托管，在 pywebview 中加载。不依赖 CDN、在线字体或常驻 Node 服务。

## 命令

```bash
npm install
npm run dev        # http://127.0.0.1:5173 ，/api 代理到 LP_API_URL（默认 http://127.0.0.1:8765）
npm run typecheck
npm run build      # 输出 dist/，对应 AppPaths.frontend_dist
```

开发构建可在地址后加 `?demo`（如 `http://127.0.0.1:5173/?demo#/tasks`）使用示例数据预览界面；
该模式全程显示横幅、拒绝所有写操作，生产构建中不可用。

## 界面结构

单窗口、单 WebView，hash 路由（静态资源无需服务端回退）。布局参照“设备框”设计：

- 左侧：星形徽标 + 胶囊导航条；底部圆形同时是后端连接状态指示。
- 当前页在工作区左缘以圆形“缺口”标记。
- 工作区：主面板 + 右侧栏；概览页为 L 形主卡片嵌套引擎卡、任务卡片堆叠、扇形采样预览。

页面：概览、引擎管理、数据集、新建训练、训练任务、训练结果、设置。

## 目录职责

- `src/api/`：`http.ts` 请求与错误；`events.ts` SSE 与重连；`bridge.ts` pywebview 受限桥接（路径对话框、定位文件）；`index.ts` 端点；`demo.ts` 开发用示例数据。
- `src/components/`：外壳、图标、状态标签、进度条、路径输入、弹窗、Loss 图表等基础组件。
- `src/features/`：全局状态（快照 + 增量事件，约 2 Hz 合并刷新，窗口隐藏时降频）、格式化、任务指标/日志订阅、提示。
- `src/pages/`：页面入口。
- `src/router/`、`src/styles/`（CSS 变量主题）、`src/types/`（与 `app/schemas` 枚举保持一致的接口类型）。

## 约定的后端接口（`/api/v1`，待后端实现）

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/system/status` | 后端版本、监管进程、GPU |
| GET | `/events` | SSE：`task.updated` / `task.metrics` / `task.log` / `engine.updated` / `system.status` |
| GET / POST | `/engines`、`/engines/rescan` | 安装实例列表、重新发现 |
| POST / PUT | `/engines/{id}/default`、`/diagnose`、`/python`、`/engine-type` | 设为默认、诊断、绑定解释器、确认类型 |
| GET | `/engines/{id}/capabilities` | 架构与参数描述（basic/advanced/native） |
| GET / POST | `/datasets`、`/datasets/{id}/scan` | 数据集引用与扫描 |
| GET / PUT | `/datasets/{id}/images`、`…/images/{img}/caption` | 分页图片、caption 编辑 |
| POST | `/training/validate`、`/training/submit` | 后端校验 + 原生配置预览、提交 |
| GET / POST | `/tasks`、`/tasks/{id}/metrics`、`/tasks/{id}/log?tail=`、`/tasks/{id}/stop` | 任务、指标、日志尾部、停止（`force`） |
| GET / POST | `/artifacts?task_id=`、`/artifacts/{id}/publish` | 产物与显式发布 |
| GET / PUT | `/settings` | 路径与监控设置 |

请求携带 `X-LP-Session`（由桌面启动入口注入）。桥接对象需提供 `pick_path(kind, title, file_types)`，可选 `open_in_explorer(path)`。

前端只展示后端返回的状态：无法解析的指标显示“未知”，后端不可达时显示明确提示，不伪造成功或进度。
