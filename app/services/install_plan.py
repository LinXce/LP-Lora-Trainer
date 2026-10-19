"""Pure install recommendation: match the machine and the clone to a CUDA source and mirrors.

No side effects. Used by ``EngineService.install_plan`` to prefill the one-click
install dialog. The engine's own installer still performs the actual install.
"""

import locale
import os
import platform
from pathlib import Path

from app.services import hardware

# PyPI / uv package index presets. ``url=None`` means "engine default".
PYPI_MIRRORS: tuple[tuple[str, str, str | None], ...] = (
    ("official", "官方 PyPI 源", None),
    ("tuna", "清华大学镜像", "https://pypi.tuna.tsinghua.edu.cn/simple"),
    ("aliyun", "阿里云镜像", "https://mirrors.aliyun.com/pypi/simple"),
    ("ustc", "中科大镜像", "https://mirrors.ustc.edu.cn/pypi/simple"),
    ("custom", "自定义镜像 URL", None),
)

# PyTorch/CUDA wheel index presets; ``{cu}`` is filled from the selected CUDA tag.
# Aliyun's pytorch-wheels path 404s for the tags engines pin today, so it is not
# offered; every URL here was checked against a real wheel listing.
TORCH_MIRRORS: tuple[tuple[str, str, str | None], ...] = (
    ("project_default", "跟随引擎项目默认", None),
    ("sjtu", "上海交大 PyTorch wheel", "https://mirror.sjtu.edu.cn/pytorch-wheels/{cu}"),
    ("official", "官方 PyTorch wheel 源", "https://download.pytorch.org/whl/{cu}"),
    ("custom", "自定义 PyTorch / CUDA 源", None),
)


# uv managed-Python download presets (astral-sh/python-build-standalone
# releases). uv appends "/<release-tag>/<asset>" to the base URL, so a mirror
# must proxy that GitHub release layout. ``None`` keeps uv's GitHub default,
# which is unreachable on some networks.
PYTHON_MIRRORS: tuple[tuple[str, str, str | None], ...] = (
    ("github", "GitHub 官方", None),
    ("nju", "南京大学镜像", "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone"),
    # The TUNA directory lagged behind the current python-build-standalone release
    # (HTTP 404 for 20261003), which cost one failed install attempt per run.
)


def _options(presets) -> list[dict]:
    return [dict(value=value, label=label, url=url) for value, label, url in presets]


def _prefers_cn_mirror() -> bool:
    """Whether this machine is Chinese, including Windows' own locale names.

    ``locale.getlocale()`` reports ``('Chinese (Simplified)_China', '936')`` on a
    Chinese Windows — it never starts with ``zh``, so a POSIX-only check leaves
    every mirror unpreselected there.
    """
    candidates = (os.environ.get("LC_ALL"), os.environ.get("LC_MESSAGES"), os.environ.get("LANG"))
    language = next((value for value in candidates if value), "")
    if not language:
        try:
            name, encoding = locale.getlocale()
        except (ValueError, TypeError):
            name, encoding = None, None
        language = f"{name or ''} {encoding or ''}"
    language = language.lower()
    return any(marker in language for marker in ("zh", "chinese", "936", "950"))


def _default_pypi_preset() -> str:
    return "tuna" if _prefers_cn_mirror() else "official"


def build(adapter, record: dict, hw: dict, requested_source: str | None = None) -> dict:
    """Return the install recommendation for one engine installation.

    ``adapter`` is None only when the engine type could not be resolved (no
    source match, or several candidates); a uniquely detected type is usable
    without a separate manual confirmation.
    """
    source_path = Path(record["source_path"])
    options = list(adapter.installation_sources(source_path)) if adapter else []
    blockers = list(adapter.install_blockers(hw)) if adapter else []
    known_type = adapter is not None
    if not hw.get("cuda_usable"):
        blockers.append("未检测到可用的 NVIDIA 驱动/CUDA 环境")
    engine_decides = not options

    driver = hw.get("cuda_version")
    driver_rank = hardware.driver_rank(driver)
    if adapter is None:
        torch_source = "existing"
        reason = "请先确认引擎类型后再选择安装来源"
    elif engine_decides:
        torch_source = "existing"
        reason = f"由 {adapter.display_name or record['engine_id']} 官方安装器自选 CUDA；本工具不指定 CUDA 配置"
    else:
        ranked = [(o, hardware.cuda_rank(o)) for o in options]
        ranked = [(o, rank) for o, rank in ranked if rank]
        fit = [(o, rank) for o, rank in ranked if driver_rank and rank <= driver_rank]
        if fit:
            torch_source = max(fit, key=lambda item: item[1])[0]
            reason = f"驱动支持 CUDA {driver}；克隆声明 {', '.join(options)} → 推荐 {torch_source}"
        else:
            torch_source = options[0]
            if driver_rank:
                blockers.append(f"驱动仅支持 CUDA {driver}，克隆声明的 {', '.join(options)} 均更高；请升级驱动或改用其它来源")
                reason = f"驱动支持 CUDA {driver}，但克隆声明的配置都更高"
            else:
                reason = "未读取到驱动 CUDA 版本；请确认显卡驱动，或手动选择安装来源"
        if requested_source in options and hardware.cuda_rank(requested_source):
            torch_source = requested_source  # honour an explicit dialog choice

    # CUDA tag used to build the wheel-mirror URLs. The engine's own requirements
    # decide which wheel index it installs from; only fall back to the driver
    # when the engine declares nothing, or mirrors would 404 on the wrong tag.
    engine_tag = None
    if adapter is not None:
        try:
            engine_tag = adapter.torch_index_tag(source_path)
        except (OSError, UnicodeError, ValueError, TypeError):
            engine_tag = None
    if engine_tag:
        reason += f"；引擎声明 torch wheel 标签 {engine_tag}（驱动为 {driver or '未知'}），镜像按该标签生成"
    rank = None if engine_decides else hardware.cuda_rank(torch_source)
    tag = engine_tag or hardware.cuda_extra(rank) or hardware.cuda_extra(driver_rank)

    torch_mirror_presets = []
    for value, label, template in TORCH_MIRRORS:
        url = template.format(cu=tag) if template and tag else None
        torch_mirror_presets.append(dict(value=value, label=label, url=url))

    blocked_reason = None
    if not known_type:
        blocked_reason = "无法唯一识别引擎类型，请先确认"
    elif blockers:
        blocked_reason = "；".join(blockers)

    return dict(
        installation_id=record["installation_id"],
        engine_id=record["engine_id"],
        supported=known_type and not blockers,
        blocked_reason=blocked_reason,
        engine_type_confirmed=bool(record.get("engine_type_confirmed")),
        engine_decides_cuda=engine_decides,
        torch_source=torch_source,
        torch_source_options=options + ["existing"],
        torch_source_reason=reason,
        cuda_version=driver,
        cuda_tag=tag,
        cuda_tag_source="engine" if engine_tag else ("engine_default" if engine_decides else "driver"),
        os=hw.get("os") or platform.system(),
        gpu_name=hw.get("gpu_name"),
        driver_version=hw.get("driver_version"),
        compute_cap=hw.get("compute_cap"),
        mirror_preset=_default_pypi_preset(),
        mirror_presets=_options(PYPI_MIRRORS),
        # A mirror only helps when its tag matches the engine's wheel index; a
        # Chinese-locale machine gets one preselected so no URL has to be typed.
        torch_mirror_preset="sjtu" if engine_tag and _prefers_cn_mirror() else "project_default",
        torch_mirror_presets=torch_mirror_presets,
        python_mirror_preset="github",
        python_mirror_presets=_options(PYTHON_MIRRORS),
    )
