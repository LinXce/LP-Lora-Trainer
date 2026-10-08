# 前端

Vue 3 + TypeScript + Vite，构建后由本地 Python 服务托管，在 pywebview 中加载。不依赖 CDN、在线字体或常驻 Node 服务。

## 命令

```bash
npm install
npm run dev        # http://127.0.0.1:5173 ，/api 代理到 LP_API_URL（默认 http://127.0.0.1:5900）
npm run typecheck
npm run build      # 输出 dist/，对应 AppPaths.frontend_dist
```

开发构建可在地址后加 `?demo`（如 `http://127.0.0.1:5173/?demo#/tasks`）使用示例数据预览界面；
该模式全程显示横幅、拒绝所有写操作，生产构建中不可用。

## 界面结构

单窗口、单 WebView，hash 路由（静态资源无需服务端回退）。布局参照“设备框”设计：

- 左侧：LP 徽标 + 胶囊导航条；底部圆形同时是后端连接状态指示。
- 当前页在工作区左缘以圆形“缺口”标记。
- 工作区：主面板 + 右侧栏；概览页为 L 形主卡片嵌套引擎卡、任务卡片堆叠、扇形采样预览。

页面：概览、引擎管理、数据集、新建训练、训练任务、训练结果、设置。

## 目录职责

- `src/api/`：`http.ts` 请求与错误；`events.ts` SSE 与重连；`bridge.ts` pywebview 受限桥接（路径对话框、定位文件）；`index.ts` 端点；`demo.ts` 开发用示例数据。
- `src/components/`：外壳、图标、状态标签、进度条、路径输入、弹窗、Loss 图表等基础组件。
- `src/features/`：全局状态（快照 + 增量事件，约 2 Hz 合并刷新，窗口隐藏时降频）、格式化、任务指标/日志订阅、提示。
- `src/pages/`：页面入口。
- `src/router/`、`src/styles/`（CSS 变量主题）、`src/types/`（与 `app/schemas` 枚举保持一致的接口类型）。

## 已接入的本机后端接口（`/api/v1`）

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| POST | `/session` | 建立 HttpOnly Cookie 本机会话 |
| GET | `/system/status` | 后端版本、监管进程、GPU |
| GET | `/events` | SSE：`task.updated` / `task.metrics` / `task.log` / `engine.updated` / `system.status` |
| GET / POST | `/engines`、`/engines/rescan` | 安装实例列表、重新发现 |
| POST / PUT | `/engines/{id}/default`、`/diagnose`、`/python`、`/engine-type` | 设为默认、诊断、绑定解释器、确认类型 |
| GET | `/engines/{id}/capabilities` | 架构与参数描述（basic/advanced/native） |
| GET / POST | `/datasets`、`/datasets/{id}/scan` | 数据集引用与扫描 |
| GET / PUT | `/datasets/{id}/images`、`…/images/{img}/caption` | 分页图片、caption 编辑 |
| POST | `/training/validate`、`/training/submit` | 后端校验 + 原生配置预览、提交 |
| GET / POST | `/tasks`、`/tasks/{id}/metrics`、`/tasks/{id}/log?tail=`、`/tasks/{id}/stop` | 任务、指标、日志尾部、停止（`force`） |
| POST | `/tasks/{id}/acknowledge-exit` | 人工确认失联训练进程已退出，解除队列阻塞 |
| GET / POST | `/artifacts?task_id=`、`/artifacts/{id}/publish` | 产物与显式发布 |
| GET / PUT | `/settings` | 路径与监控设置 |

前端通过同源 `POST /session` 建立 HttpOnly、SameSite=Strict 的 Cookie 会话；HTTP 和 SSE 使用同一会话，401 后重新建立会话。后端检查 Host、Origin 和跨站来源，不开放远程业务访问。`X-LP-Session` 保留兼容路径，但桌面启动不把密钥暴露到 URL。

SSE 重连时重取系统、任务和引擎快照，同时刷新当前任务日志尾部及指标，补上断线期间的历史。可见窗口约 2 Hz 合并事件，隐藏窗口降频；图表和日志缓存有上限。

pywebview 桥接提供 `pick_path(kind, title, file_types)` 和 `open_in_explorer(path)`。浏览器开发模式未提供桥接时可手动填写路径。

前端只展示后端返回的状态：无法解析的指标显示“未知”，后端不可达时显示明确提示，不伪造成功或进度。


## 当前范围与设计约束

保留既有页面布局、CSS、导航和交互视觉风格，仅补全接口、状态与功能逻辑；左上角原菱形/星形图案及站点图标已替换为 LP Logo。

Kohya 已接入训练闭环；AI Toolkit 当前仅支持管理和诊断，不展示伪造的训练成功。没有做额外的前端技术选型或性能原型验证。

开发时先启动后端 `.\.venv\Scripts\python.exe -m app.api.server --dev`，再在 `frontend/` 执行 `npm run dev`。桌面运行前执行 `npm run build`，由后端托管 `dist/`；`dist/` 不提交版本控制。
