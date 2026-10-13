"""Plain-language explanation for a finished engine session.

Pure text analysis over the session's ``error``/``diagnostic_error`` fields and
the tail of its log. Every rule matches tool output that *any* engine can
produce (uv, pip, git, CUDA, the OS) and never an engine name, so the main
program keeps no engine-specific branches.
"""
from __future__ import annotations

MAX_HINTS = 4
MAX_EVIDENCE = 2
MAX_LINE_CHARS = 200

# Failure kinds that a different package source can actually fix. Anything else
# (a broken engine script, a missing entry point) must not be retried.
RETRYABLE = frozenset({"python_download", "network_unreachable", "network_timeout", "index_missing"})

MIRROR_FALLBACK = "mirror_fallback"

# Ordered most specific first: a managed-Python download failure is also a
# network failure, and the precise explanation must win.
RULES: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    (
        "python_download",
        "无法下载受管 Python",
        "uv 需要下载受管解释器（python-build-standalone），但下载失败。",
        "在安装对话框把「Python 下载源」改为可用镜像（南京大学 / 清华大学 / 自定义）后重试；"
        "或手动把所需版本放入 runtime/uv/python/。",
        ("python-build-standalone",),
    ),
    (
        "network_unreachable",
        "无法连接依赖源",
        "连接被拒绝或域名无法解析，通常是网络、代理或防火墙问题。",
        "检查网络与代理设置，选择可访问的镜像源后重试。",
        (
            "os error 10061", "os error 11001", "connection refused", "dns error",
            "failed to create underlying connection", "no route to host", "积极拒绝",
        ),
    ),
    (
        "network_timeout",
        "下载依赖超时",
        "依赖源响应超时，多为网络不稳定或镜像速度过慢。",
        "重试一次，或改用其它镜像源。",
        ("timed out", "timeout", "request failed after"),
    ),
    (
        "index_missing",
        "依赖源缺少所需包或版本",
        "所选依赖源没有匹配的发行版，常见于镜像未同步或 CUDA wheel 版本不全。",
        "改用官方源或其它镜像；PyTorch 相关请确认所选 CUDA 的 wheel 地址确实存在。",
        (
            "no matching distribution", "could not find a version that satisfies",
            "404 not found", "not found for url",
        ),
    ),
    (
        "cuda_driver",
        "显卡驱动与 CUDA 版本不匹配",
        "安装的 CUDA 运行时高于驱动支持上限，或缺少对应算力的 kernel。",
        "升级显卡驱动，或在对话框中改选不高于驱动支持的 CUDA 版本后重试。",
        (
            "cuda driver version is insufficient", "no kernel image is available",
            "insufficient for cuda", "cuda driver version",
        ),
    ),
    (
        "disk_space",
        "磁盘空间不足",
        "解压或写入依赖时空间不足。",
        "清理引擎所在磁盘空间后重试（依赖与模型通常需要数 GB）。",
        ("no space left on device", "not enough space", "disk full", "磁盘空间"),
    ),
    (
        "permission",
        "文件被占用或权限不足",
        "目标文件无法写入，常由杀毒软件、其它进程占用或权限限制导致。",
        "关闭占用该目录的程序与实时防护，或以管理员身份重新运行后再试。",
        (
            "access is denied", "permission denied", "os error 5", "winerror 5",
            "拒绝访问", "being used by another process",
        ),
    ),
    (
        "python_env",
        "Python 环境创建失败",
        "uv 未能创建引擎独立环境。",
        "查看上方 uv 报错详情；若提示缺少解释器请配置「Python 下载源」，"
        "或删除引擎目录下的 venv / .venv 后重试。",
        ("uv venv failed with exit code", "failed to create engine environment", "uv completed but did not create"),
    ),
    (
        "git_repository",
        "Git 仓库无法拉取",
        "仓库不存在、无访问权限，或地址无法解析。",
        "确认仓库地址与分支名正确；私有仓库需可用的凭据；网络受限时改用可达的镜像地址。",
        (
            "repository not found", "could not read username", "authentication failed",
            "could not resolve host", "terminal prompts disabled",
        ),
    ),
    (
        "missing_pkg_resources",
        "引擎安装器需要 pkg_resources",
        "引擎的安装脚本导入了 pkg_resources，但 setuptools 82 起已移除该模块。",
        "为引擎环境安装仍提供 pkg_resources 的 setuptools（如 setuptools<82）后重试；"
        "长期方案是引擎改用 importlib.metadata。",
        ("no module named 'pkg_resources'", 'no module named "pkg_resources"'),
    ),
    (
        "installer_exit",
        "引擎官方安装器执行失败",
        "引擎自己的安装脚本以非零状态退出。",
        "查看该阶段输出的最后几行；若失败发生在下载依赖，请更换镜像源后重试。",
        ("official installer failed", "installer failed", "exit code"),
    ),
)

_STOPPED = {
    "cancelled": (
        "安装已停止",
        "安装被手动停止，已完成的依赖会被保留。",
        "重新点击「安装引擎环境」即可继续，已安装的依赖会自动跳过。",
    ),
    "interrupted": (
        "安装被中断",
        "应用或安装进程在运行期间退出。",
        "重新运行安装；若反复中断，请查看完整日志末尾。",
    ),
}


def retry_reason(log_text: str, error: str | None = None) -> str | None:
    """Name the cause when a failure is worth retrying on the official source.

    Returns the matched hint title, or ``None`` for failures that another
    package source cannot fix.
    """
    for hint in analyze("failed", error, None, log_text):
        if hint["id"] in RETRYABLE:
            return hint["title"]
    return None


def mirror_fallback_notice(reason: str, outcome: str = "retrying") -> dict:
    """Warning for the mirror-to-official fallback, shown next to the log.

    ``outcome`` is ``retrying``, ``succeeded`` or ``failed``.
    """
    detail = {
        "retrying": f"镜像源下载失败（{reason}），正在改用官方源重试。",
        "succeeded": f"镜像源下载失败（{reason}），已改用官方源完成安装。",
        "failed": f"镜像源下载失败（{reason}），改用官方源重试后仍然失败。",
    }[outcome]
    return dict(
        id=MIRROR_FALLBACK,
        title="镜像源不可用，已回退到官方源",
        detail=detail,
        suggestion="如需继续使用镜像，请确认该镜像地址可访问；也可以改用其它镜像或保留官方源。",
        evidence=[],
    )


def _clean(line: str) -> str:
    return " ".join(line.split())[:MAX_LINE_CHARS]


def _evidence(lines: list[str], markers: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for line in lines:
        lowered = line.lower()
        if any(marker in lowered for marker in markers):
            cleaned = _clean(line)
            if cleaned and cleaned not in found:
                found.append(cleaned)
            if len(found) >= MAX_EVIDENCE:
                break
    return found


def analyze(
    state: str,
    error: str | None = None,
    diagnostic_error: str | None = None,
    log_text: str = "",
    kind: str | None = None,
) -> list[dict]:
    """Return zero or more failure explanations, most relevant first.

    Only finished sessions produce hints: a session that is still running has
    no failure to explain yet.
    """
    if state in {"succeeded", "queued", "running", "stopping", "verifying"}:
        return []
    if state in _STOPPED:
        title, detail, suggestion = _STOPPED[state]
        return [dict(id=state, title=title, detail=detail, suggestion=suggestion, evidence=[])]

    parts = [part for part in (error, diagnostic_error, log_text) if part]
    haystack = "\n".join(parts)
    lowered = haystack.lower()
    lines = [line.strip() for line in haystack.splitlines() if line.strip()]

    hints: list[dict] = []
    if diagnostic_error:
        hints.append(dict(
            id="diagnostic",
            title="环境诊断未通过",
            detail=_clean(diagnostic_error),
            suggestion="引擎入口或关键依赖缺失；请重新安装环境，并确认引擎源码目录完整。",
            evidence=[],
        ))
    for rule_id, title, detail, suggestion, markers in RULES:
        if any(marker in lowered for marker in markers):
            hints.append(dict(
                id=rule_id,
                title=title,
                detail=detail,
                suggestion=suggestion,
                evidence=_evidence(lines, markers),
            ))
    if not hints:
        hints.append(dict(
            id="unknown",
            title="拉取失败，原因未识别" if kind == "clone" else "运行失败，原因未识别",
            detail=_clean(error) if error else "日志中没有匹配到已知的失败特征。",
            suggestion="展开完整日志查看最后几行，或用「复制日志」反馈问题。",
            evidence=[],
        ))
    return hints[:MAX_HINTS]
