"""Supplementary engine arguments for every training base model.

The text box is the escape hatch for flags the adapters do not model yet. LP
parses it into engine-config keys instead of raw argv, because both engines read
their settings from a config file and would otherwise resolve a repeated flag
silently (argparse keeps the last one, which is never what a reader expects).

Grammar, deliberately small and predictable:

* ``--flag value`` sets one key; numbers and booleans are typed automatically.
* ``--flag`` alone switches the key on (``true``).
* Quotes group a value so it may contain spaces; a quoted value stays text.
* Backslashes stay literal so Windows paths survive unchanged.

Everything else (multi-value arguments, sub-commands) is out of scope: the
adapters already expose the engine's own ``*_args`` string parameters for that.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

PARAM_KEY = "extra_params"
KEYS_LABEL_LIMIT = 6

_QUOTES = "'\""
_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _is_flag(token: str) -> bool:
    """``--name`` / ``-n`` but not a negative number such as ``-1``."""
    return len(token) > 1 and token[0] == "-" and not token[1].isdigit() and token[1] != "."


def _tokens(text: str) -> list[tuple[str, bool]]:
    tokens: list[tuple[str, bool]] = []
    current: list[str] = []
    quoted = False
    quote = ""
    for char in text or "":
        if quote:
            if char == quote:
                quote = ""
            else:
                current.append(char)
            continue
        if char in _QUOTES:
            quote = char
            quoted = True
        elif char.isspace():
            if current or quoted:
                tokens.append(("".join(current), quoted))
            current, quoted = [], False
        else:
            current.append(char)
    if quote:
        raise ValueError(f"引号 {quote} 没有成对出现")
    if current or quoted:
        tokens.append(("".join(current), quoted))
    return tokens


def _typed(token: str, quoted: bool) -> object:
    if quoted:
        return token
    lowered = token.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if _NUMBER.match(token):
        if "." not in token and "e" not in lowered:
            return int(token)
        return float(token)
    return token


def parse(text: str) -> dict[str, object]:
    """The config keys the text sets, in order, with duplicates collapsed."""
    values: dict[str, object] = {}
    pending: str | None = None
    for token, quoted in _tokens(text):
        if _is_flag(token):
            if pending:
                values[pending] = True
            name = token.lstrip("-")
            if not name:
                raise ValueError("参数名不完整，请写成 --参数名 的形式")
            pending = name
        elif pending is None:
            raise ValueError(f"“{token}” 前面缺少参数名，请写成 --参数名 值")
        else:
            values[pending] = _typed(token, quoted)
            pending = None
    if pending:
        values[pending] = True
    return values


def duplicates(text: str) -> list[str]:
    """Parameter names given more than once."""
    seen: set[str] = set()
    repeated: list[str] = []
    for token, _ in _tokens(text):
        if not _is_flag(token):
            continue
        name = token.lstrip("-")
        if name in seen and name not in repeated:
            repeated.append(name)
        seen.add(name)
    return repeated


def _listed(names: Iterable[str]) -> str:
    shown = list(names)
    text = "、".join("--" + name for name in shown[:KEYS_LABEL_LIMIT])
    return f"{text} 等 {len(shown)} 项" if len(shown) > KEYS_LABEL_LIMIT else text


def problems(text: str, provided: Iterable[str]) -> list[tuple[str, str]]:
    """``(severity, message)`` pairs for the text.

    ``error`` means the text cannot be used at all (unparseable, or the same
    parameter twice, where nothing can tell which value was meant). ``warning``
    means it repeats a field LP already fills in: the caller is expected to ask
    the user before submitting, and the engine config ends up with the custom
    value because the extra keys are merged last.
    """
    if not (text or "").strip():
        return []
    try:
        values = parse(text)
    except ValueError as error:
        return [("error", str(error))]
    repeated = duplicates(text)
    if repeated:
        return [("error", f"同一个参数给了多次，无法确定用哪个值：{_listed(repeated)}")]
    known = {str(name).lstrip("-") for name in provided}
    clashes = [name for name in values if name in known]
    if clashes:
        return [("warning", f"自定义参数会覆盖上面已设置的字段：{_listed(clashes)}")]
    return []


def merge(target: Mapping[str, object], text: str) -> dict[str, object]:
    """``target`` plus the extra keys. Callers validate first, so this only extends."""
    merged = dict(target)
    if (text or "").strip():
        merged.update(parse(text))
    return merged


def provided_keys(params: Iterable[object], architecture: str, values: Mapping[str, object],
                  *, skip_false: bool, extra: Iterable[str] = ()) -> set[str]:
    """Engine keys the config already carries, mirroring how each adapter renders it.

    ``skip_false`` matches the adapters' own rule: kohya writes ``false`` keys into
    its TOML, musubi leaves them out of its config file.
    """
    keys = {str(name).lstrip("-") for name in extra}
    for p in params:
        if not getattr(p, "native", True) or not p.applies_to(architecture):
            continue
        value = values.get(p.key, p.default)
        if value is None or (isinstance(value, str) and value == "") or (skip_false and value is False):
            continue
        keys.add(str(p.native_key or p.key).lstrip("-"))
    return keys
