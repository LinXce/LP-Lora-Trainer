# Engine adapters

**English** · [中文](./README.zh-CN.md)

Every training engine LP can drive is a directory here. An adapter owns **all**
engine-specific knowledge — detection, installation, parameter surface, native
config rendering, launch argv, log parsing, artifact collection. The application
and the frontend never name an engine: they render whatever the adapter catalog
reports. A test enforces that (`tests/test_shell_is_engine_agnostic.py`).

## Layout

| Path | Purpose |
|---|---|
| `base.py` | The adapter contract (`EngineAdapter`) and `ADAPTER_CONTRACT_VERSION`. |
| `capabilities.py` | `ArchSpec` (base-model type → entry script) and `Param` (one parameter). |
| `registry.py` | Directory discovery, isolation of broken plugins, duplicate-id handling. |
| `extra_params.py` | Shared "extra arguments" bridge for adapters that expose one. |
| `sampling.py` | Shared sample-prompt file handling. |
| `<engine_id>/adapter.py` | Exactly one concrete `EngineAdapter` subclass. |
| `<engine_id>/params.py` | The engine's architecture and parameter tables (convention). |

Engine ids must match `[a-z][a-z0-9_]*`. Discovery is static: it never imports
or executes copied engine source, and never loads plugins from `engine/`.

## The contract

Required (abstract in `base.py`):

`detect` · `capabilities` · `validate` · `write_native_config` · `build_launch` ·
`parse_log` · `recovery_capabilities` · `collect_artifacts`

Commonly overridden:

- **Installation** — `installation_plan`, `installation_sources`, `source_issues`,
  `environment_strategy` (`precreate_venv` / `uv_project` / `native`),
  `environment_dir`, `python_version`, `torch_index_mode` (`named` / `env` / `none`).
- **Behaviour declarations** — `supports_nested_images`, `creates_output_dir`,
  `submittable`, `native_format`, `diagnostic_modules`.
- **UI metadata** — `display_name`, `adapter_status`, `training_notice`,
  `installation_help`, `clone_url` / `clone_branch` / `clone_commit`.
- **Extension points** — `preview_commands`, `native_config_text`,
  `export_parameters`, `export_parameter_files`, `caption_cache_notices`,
  `finalize_artifacts` (post-run work such as writing a third-party-readable copy).

`finalize_artifacts` runs once after the last stage exits successfully; its text is
logged and can never turn a successful run into a failure.

## Parameter tables

- `ARCHITECTURES: tuple[ArchSpec, ...]` — one entry per base-model type, each with
  its training entry script.
- `PARAMS: tuple[Param, ...]` — the engine's real surface:
  - `architectures=("id",)` scopes a parameter to those base models; leaving it
    empty means **every** architecture of this engine (legitimate for one trainer
    with several base models, wrong for two separate trainers in one clone).
  - `native_key` maps the UI key to the engine's own flag/field name.
  - `section` and `display_order` decide where and in which order it renders.
  - `min` / `max` / `options` are validated before a job is queued.
  - Divisor-like settings (save/sample intervals) must be `min=1`: zero would be
    written into the engine and crash it.

Hard rules enforced by tests: no duplicate keys, **no duplicate labels within one
architecture**, and every `native_key` must exist in the engine's own definitions
(`tests/test_adapter_contract.py`).

## Adding an engine

1. Put the engine source under `engine/<name>/`.
2. Create `adapters/<engine_id>/adapter.py` with one concrete subclass, plus
   `params.py` with `ARCHITECTURES` and `PARAMS`.
3. Implement the eight required methods; override the declarations that apply
   (environment strategy, `clone_url`, status text, …).
4. Keep `detect` specific: check the engine's own package name and entry script so
   a sibling engine with a similar layout is not misdetected.
5. Run `python -m unittest discover -s tests`. The contract tests pick the new
   engine up automatically. No change to `app/` or `frontend/` is expected.

## Pitfalls seen in practice

- **Boolean defaults that are `True`** need the explicit `--no-<flag>` form; sending
  nothing keeps the engine's default on.
- **Caches live inside the output folder** for some engines — exclude them in
  `collect_artifacts`, or cache files get listed as checkpoints.
- **Same flag name does not mean the same meaning**: per-engine defaults differ, so
  do not merge a parameter across two different trainers.
- **Deleting or rewriting the engine's files is out of bounds**; adapt only from the
  outside, through flags and generated config files.
