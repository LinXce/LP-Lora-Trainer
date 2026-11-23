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
    choices = list(dict.fromkeys([*options, "existing"]))
    blockers = list(adapter.install_blockers(hw)) if adapter else []
    known_type = adapter is not None
    engine_decides = not options
    driver = hw.get("cuda_version")
    driver_rank = hardware.driver_rank(driver)
    fit = [(rank, name) for name in options if (rank := hardware.cuda_rank(name))
           and driver_rank and rank <= driver_rank]
    recommendation = max(fit)[1] if fit else choices[0]
    torch_source = requested_source if requested_source is not None else recommendation
    # Older clients sent cu124 unconditionally, even to the native manager
    # which always selects its own stack and has no configurable CUDA sources.
    if engine_decides and adapter and adapter.environment_strategy == "native" and hardware.cuda_rank(torch_source):
        torch_source = "existing"
    engine_tag = None
    if adapter is not None:
        try:
            engine_tag = adapter.torch_index_tag(source_path)
        except (OSError, UnicodeError, ValueError, TypeError):
            engine_tag = None
    rank = hardware.cuda_rank(torch_source) if torch_source in options else None
    tag = hardware.cuda_extra(rank)
    tag_origin = "engine_default"
    os_name = hw.get("os") or platform.system()
    if torch_source not in choices:
        reason = f"此引擎版本未声明安装来源 {torch_source}；可用来源：{', '.join(choices)}"
        blockers.append(reason)
    elif rank:
        if tag == engine_tag:
            tag_origin = "engine"
        elif requested_source is not None:
            tag_origin = "selection"
        else:
            tag_origin = "driver" if driver_rank else "engine"
        selection = "所选" if requested_source is not None else "推荐"
        reason = f"{selection} CUDA 来源为 {tag}，PyTorch 镜像按此标签生成"
        if driver_rank:
            reason += f"；本机驱动支持 CUDA {driver}"
            if rank > driver_rank:
                blockers.append(f"{tag} 超出本机驱动支持的 CUDA {driver}，请选择兼容来源或升级驱动")
        else:
            reason += "；未读取到 NVIDIA 驱动的 CUDA 版本，请核实兼容性"
    elif torch_source == "rocm-windows":
        reason = "所选来源使用引擎的 Windows AMD ROCm 依赖组"
        if os_name.lower() != "windows":
            blockers.append("rocm-windows 依赖组需要 Windows")
    else:
        reason = "使用引擎项目默认依赖，由官方安装流程处理 PyTorch"
        if engine_tag and torch_source == "existing":
            reason += f"；项目声明的默认 CUDA wheel 标签为 {engine_tag}"
            pinned_rank = hardware.cuda_rank(engine_tag)
            if pinned_rank and driver_rank and pinned_rank > driver_rank:
                blockers.append(f"项目默认 {engine_tag} 超出本机驱动支持的 CUDA {driver}")
    torch_mirror_presets = [
        dict(value=value, label=label, url=template.format(cu=tag) if template else None)
        for value, label, template in TORCH_MIRRORS if tag or value == "project_default"
    ]

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
        torch_source_options=choices,
        torch_source_reason=reason,
        cuda_version=driver,
        cuda_tag=tag,
        cuda_tag_source=tag_origin,
        os=os_name,
        gpu_name=hw.get("gpu_name"),
        driver_version=hw.get("driver_version"),
        compute_cap=hw.get("compute_cap"),
        mirror_preset=_default_pypi_preset(),
        mirror_presets=_options(PYPI_MIRRORS),
        # A mirror only helps when its tag matches the engine's wheel index; a
        # Chinese-locale machine gets one preselected so no URL has to be typed.
        torch_mirror_preset="sjtu" if tag and tag == engine_tag and _prefers_cn_mirror() else "project_default",
        torch_mirror_presets=torch_mirror_presets,
        python_mirror_preset="github",
        python_mirror_presets=_options(PYTHON_MIRRORS),
    )
