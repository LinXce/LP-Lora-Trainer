"""Close a local LP surface using one authenticated backend session."""
import http.cookiejar
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request


class ShutdownClient:
    def __init__(self, url):
        self.url = url
        self.http = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        )
        # Keep this cookie for the entire decision. A replacement API on the
        # same port must reject our POST instead of inheriting this consent.
        with self.http.open(self._request("/session", "POST"), timeout=3):
            pass

    def _request(self, path, method):
        return urllib.request.Request(self.url + "/api/v1" + path, method=method,
                                      headers={"Origin": self.url})

    def call(self, path="/status", method="GET"):
        try:
            with self.http.open(self._request("/system/shutdown" + path, method), timeout=5) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.load(exc).get("detail")
            except (OSError, ValueError, AttributeError):
                detail = None
            raise ValueError(detail if isinstance(detail, str) else f"HTTP {exc.code}") from exc
        if not isinstance(data, dict):
            raise ValueError("退出接口返回了无效状态")
        return data


def close_surface(url, process, ask, progress, cancelled=lambda: False):
    """Return True only when the window alone may detach or owned shutdown finished.

    ask(reason, title, body, primary_label) returns a boolean. No native
    resource operation here depends on a PID looked up by port.
    """
    def tell(message):
        ask("notice", "保留当前窗口", message, "返回")
        return False

    try:
        client = ShutdownClient(url)
        state = client.call()
    except (OSError, ValueError) as exc:
        if process is not None:
            if process.poll() is not None:
                return ask("foreign", "仅关闭本窗口", "原后端已退出，无法核实训练状态。关闭本窗口会保留可能仍在运行的任务。", "仅关闭本窗口")
            return tell(f"暂时无法核实后端状态，训练和窗口已保留。{exc}")
        state = None

    if process is None or process.poll() is not None:
        # The originally owned process exited. Another instance may now be
        # serving this port; only the current window may be closed.
        if state and state.get("status") == "idle" and state.get("reason") == "idle":
            return True
        return ask("foreign", "仅关闭本窗口", "训练由其他实例管理，关闭本窗口后任务继续。", "仅关闭本窗口")

    if not isinstance(state, dict) or state.get("backend_pid") != process.pid:
        return tell("无法确认该后端属于本窗口，已保留当前工作。")

    def post(snapshot, force=False):
        query = {"graceful": "true", "expected_reason": snapshot["reason"]}
        if snapshot.get("task_id"):
            query["expected_task_id"] = snapshot["task_id"]
        if force:
            query["force"] = "true"
        return client.call("?" + urllib.parse.urlencode(query), "POST")

    def cancel():
        client.call("/cancel", "POST")
        return False

    try:
        reason = state.get("reason")
        if state.get("status") in ("draining", "blocked", "ready_to_exit", "shutting_down"):
            # Resume observing an existing attempt after a temporary API failure.
            pass
        elif reason == "unverified":
            return tell("有任务的训练进程状态未知，请在任务页核实。")
        elif reason == "install":
            if not ask("install", "环境安装仍在运行", "停止当前安装后退出，稍后可重新发起安装。", "停止安装并退出"):
                return False
            # This existing endpoint stops only installation; it refuses live training.
            client.call("?force=true", "POST")
            state = {"status": "shutting_down"}
        elif reason in ("training", "queued", "idle"):
            if reason == "training":
                names = "、".join(t.get("name") or t["task_id"] for t in state.get("tasks", [])
                                 if t.get("state") in ("running", "preparing", "stopping"))
                if not ask("training", "训练仍在运行", f"请求停止当前训练{('「' + names + '」') if names else ''}，等待引擎收尾后退出。能否续训取决于实际保存点。", "终止训练并退出"):
                    return False
            elif reason == "queued" and not ask("queued", "队列中有待训练任务", "保留排队任务，重新启动后继续处理。", "保留队列并退出"):
                return False
            state = post(state)
        else:
            return tell("无法核实退出条件，已保留当前工作。")

        saved_message = ""
        while True:
            phase = state.get("status")
            if phase == "shutting_down":
                progress((saved_message + " " if saved_message else "") + "后端正在退出…", False)
                try:
                    process.wait(timeout=30)
                    return True
                except subprocess.TimeoutExpired:
                    return tell("后端尚未退出，已保留窗口。可稍后重试。")
            if state.get("backend_pid", process.pid) != process.pid:
                return tell("后端实例已改变，已保留当前工作。")
            if cancelled():
                return cancel()
            message = state.get("message") or "正在等待训练收尾…"
            if phase == "ready_to_exit":
                saved = state.get("save_result") or {}
                saved_message = saved.get("message") or message
                progress(saved_message, False)
                state = post(state)
            elif phase == "blocked":
                if state.get("force_allowed") is True and state.get("task_id"):
                    if ask("force", "训练尚未退出", message + " 强制终止可能丢失未保存的进度。", "强制终止并退出"):
                        state = post(state, force=True)
                        continue
                    return cancel()
                cancel()
                return tell(message)
            elif phase == "draining":
                progress(message, True)
                time.sleep(0.5)
                state = client.call()
            else:
                return tell("后端没有确认退出，窗口已保留。")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # In particular a 409 means the snapshot the user approved changed.
        # Preserve the fence on a broken connection; a new gesture re-reads it.
        return tell(f"退出条件已变化或后端未响应，请核实后重试。{exc}")
