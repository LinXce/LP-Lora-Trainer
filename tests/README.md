# 测试目录

测试全部平铺在本目录下（`test_*.py`），无 GPU、网络或真实引擎依赖；部分用例仅在 Windows 或便携运行时下运行，会自动跳过。

- `local_client.py`：基于真实本机 HTTP（uvicorn + 标准库）的测试客户端，覆盖套接字、会话 cookie、中间件与应用生命周期。
- `engine_fixtures.py`：最小引擎解释器与目录夹具；不下载模型、数据集或上游仓库。

`unit/`、`integration/`、`fixtures/` 目前为空占位目录（仅 `.gitkeep`），未使用。

运行：

```powershell
.\python_runtime\python.exe -m unittest discover -s tests -v
```

桌面退出验收需要 Windows、WebView2 和已构建的前端。在项目根目录运行以下命令，会依次打开四个隔离测试窗口。每个窗口使用独立临时数据、随机端口和 CPU 测试引擎，分别执行取消和确认操作。

```powershell
.\python_runtime\python.exe -B tests/windows_close_acceptance.py
```

同名训练验收会先排入两个同名、同权重前缀的任务。第二个任务结束后，以及删除第一条任务记录后，分别按任务 ID 查找权重，校验两个原始文件的 SHA-256。目录归属用例同时覆盖升级前已排队的任务。

```powershell
.\python_runtime\python.exe -B -m unittest tests.test_same_name_runs tests.test_output_ownership tests.test_bound_output_layout
```
