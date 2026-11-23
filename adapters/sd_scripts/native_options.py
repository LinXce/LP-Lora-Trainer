"""Read the selected sd-scripts entry's argparse declarations without importing torch."""
import ast
from functools import lru_cache
import warnings


@lru_cache(maxsize=64)
def _declarations(path, modified):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    imports, functions = {}, {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = node
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            prefix = (node.module or "") + "." if isinstance(node, ast.ImportFrom) else ""
            for alias in node.names:
                imports[alias.asname or alias.name] = prefix + alias.name
    return functions, imports


def options_for(root, script):
    """Follow setup_parser and its add_*_arguments helpers in the bound checkout."""
    options, visited = {}, set()

    def collect(module, function):
        if (module, function) in visited:
            return
        visited.add((module, function))
        path = root.joinpath(*module.split(".")).with_suffix(".py")
        functions, imports = _declarations(path, path.stat().st_mtime_ns)
        if function not in functions:
            raise ValueError(function)
        for call in ast.walk(functions[function]):
            if not isinstance(call, ast.Call):
                continue
            name = ast.unparse(call.func)
            if name.endswith(".add_argument"):
                flags = [a.value for a in call.args if isinstance(a, ast.Constant)
                         and isinstance(a.value, str) and a.value.startswith("--")]
                if not flags:
                    continue
                kwargs = {kw.arg: kw.value for kw in call.keywords}
                def literal(key, default=None):
                    try:
                        return ast.literal_eval(kwargs[key])
                    except (KeyError, ValueError, TypeError):
                        return default
                dest = literal("dest", flags[0][2:].replace("-", "_"))
                kind = ast.unparse(kwargs["type"]).split(".")[-1] if "type" in kwargs else "str"
                if literal("action") in ("store_true", "store_false"):
                    kind = "bool"
                options[dest] = (kind, literal("choices"), literal("nargs"))
            elif name.endswith("setup_parser") or (name.split(".")[-1].startswith("add_")
                                                  and name.endswith(("_argument", "_arguments"))):
                head, _, tail = name.partition(".")
                target = imports.get(head, module + "." + head) + ("." + tail if tail else "")
                collect(*target.rsplit(".", 1))

    try:
        collect(script.removesuffix(".py"), "setup_parser")
    except (OSError, UnicodeError, SyntaxError, ValueError):
        # Older/custom checkouts can build the parser dynamically.
        return None
    return options or None
