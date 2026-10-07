# 桌面层

- `launcher.py`：桌面单实例锁，启动或连接本地后端，等待服务就绪，再打开主窗口。
- `window.py`：创建无系统标题栏的 pywebview 窗口；Windows 使用 WebView2。
- `bridge.py`：原生路径选择、资源管理器定位以及受限窗口控制与状态事件。
- `window_chrome.py`：Windows 原生边缘缩放、最大化工作区及状态读取，不接收任意窗口句柄或系统消息。
- `frontend/src/components/WindowTitleBar.vue`：与既有深灰主题一致的 LP 标题栏及最小化、最大化/还原、关闭按钮。

## 窗口交互

- 标题栏空白区域支持拖动，双击切换最大化/还原，按钮区域不参与拖动。
- Windows 窗口非最大化时，四边和四角支持调整大小；最小尺寸仍为 1000 × 700。
- 最大化使用当前显示器的工作区，避免覆盖任务栏；原生最大化/还原事件同步按钮图标，无定时轮询。
- 标题栏只在 pywebview 窗口桥接就绪后出现，普通浏览器访问不显示桌面控制按钮。
- 关闭按钮只销毁窗口。重新运行 `LP-Lora-Trainer.exe` 会连接仍在运行的后端；训练不中断。

窗口层不管理训练子进程，不调用引擎适配器，不提供任意命令执行接口。标题栏复用现有字体、主题颜色和 LP 图形，不增加 UI 框架、图像资源或 Python 依赖，也不更改业务页面样式。

无 GUI 单元测试：`python_runtime\python.exe -m unittest discover -s tests -p test_window_chrome.py -v`。
