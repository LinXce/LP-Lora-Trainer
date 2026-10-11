"""Prompt text rendering shared by adapters. No engine imports or path reads."""
import re
from pathlib import Path

PROMPT_FILENAME = "sample_prompts.txt"
PROMPT_TEXT_LIMIT = 100000


def looks_like_prompt_path(value):
    """Recognize legacy file references, not natural-language prompt sentences."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\n" in text or "\r" in text or " --" in text:
        return False
    return bool(re.match(r"^[a-zA-Z]:[/\\]", text) or text.startswith("\\\\")
                or re.fullmatch(r"[^\n<>|?*]+\.(?:txt|json)", text, re.IGNORECASE))


def prompt_error(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return "采样提示词必须是文本"
    if len(value) > PROMPT_TEXT_LIMIT or "\x00" in value:
        return "采样提示词不能包含空字符且不能超过 100000 字符"
    if looks_like_prompt_path(value):
        return "采样提示词现在使用文本，请粘贴原提示词文件内容，而不是填写文件路径"
    return None


def prompt_text(config):
    value = config.get("params", {}).get("sample_prompts", "")
    error = prompt_error(value)
    if error:
        raise ValueError(error)
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def prompt_file(config, directory):
    return str(Path(directory) / PROMPT_FILENAME) if prompt_text(config) else ""


def write_prompts(config, directory):
    from app.services.common import atomic_text
    text = prompt_text(config)
    if text:
        atomic_text(Path(directory) / PROMPT_FILENAME, text + "\n")


def exported_prompt_files(config):
    text = prompt_text(config)
    return [{"filename": PROMPT_FILENAME, "text": text + "\n"}] if text else []
