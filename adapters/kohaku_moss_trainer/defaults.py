"""Read kogine's declared knobs without importing an installed training environment."""

import ast
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from .params import PARAMS


@lru_cache(maxsize=24)
def _read(path, mtime_ns, size):
    knobs = {}
    for stmt in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if isinstance(stmt, ast.AnnAssign):
            target = stmt.target
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
        else:
            continue
        if not isinstance(target, ast.Name) or not target.id.isupper():
            continue
        node = stmt.value
        # TARGET_AREA is declared as the product of two integer literals.
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if all(isinstance(n, ast.Constant) and type(n.value) is int for n in (node.left, node.right)):
                knobs[target.id] = node.left.value * node.right.value
        else:
            try:
                knobs[target.id] = ast.literal_eval(node)
            except (ValueError, TypeError):
                continue
    return knobs


def script_defaults(source, script):
    path = Path(source) / script
    try:
        stat = path.stat()
        return _read(path, stat.st_mtime_ns, stat.st_size)
    except (OSError, UnicodeError, SyntaxError):
        return {}


def form_value(spec, value):
    if isinstance(value, (dict, list, tuple)) and spec.type == "string":
        return repr(value)
    if isinstance(value, bool) and spec.type == "enum":
        return str(value).lower()
    return value


def installed_params(installation, train_script, cache_stages):
    if installation is None:
        return PARAMS
    source = installation.source_path
    train = script_defaults(source, train_script)
    text_cache, latent_cache = (script_defaults(source, stage) for stage in cache_stages)
    defaults = dict(train)
    for ui, stage, native in (
        ("ENCODE_BATCH", text_cache, "ENCODE_BATCH"),
        ("VAE_ENCODE_BATCH", latent_cache, "ENCODE_BATCH"),
        ("IMAGE_ENCODER", latent_cache, "IMAGE_ENCODER"),
    ):
        if native in stage:
            defaults[ui] = stage[native]
    return tuple(replace(p, default=form_value(p, defaults[p.key])) if p.key in defaults else p
                 for p in PARAMS)
