# 测试目录

测试全部平铺在本目录下（`test_*.py`），无 GPU、网络或真实引擎依赖；部分用例仅在 Windows 或便携运行时下运行，会自动跳过。

- `local_client.py`：基于真实本机 HTTP（uvicorn + 标准库）的测试客户端，覆盖套接字、会话 cookie、中间件与应用生命周期。
- `engine_fixtures.py`：最小引擎解释器与目录夹具；不下载模型、数据集或上游仓库。

`unit/`、`integration/`、`fixtures/` 目前为空占位目录（仅 `.gitkeep`），未使用。

运行：

```powershell
.\python_runtime\python.exe -m unittest discover -s tests -v
```
