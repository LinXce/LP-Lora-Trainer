"""Exercise same-name submission, output retention and resume in a temporary workspace."""
import hashlib
from pathlib import Path
import unittest

from supervisor.worker import Worker
from tests import test_functionality as fixtures


# The weight depends only on the requested seed. A resumed run must actually
# read its parent's state and produces a different value in its own directory.
CPU_ENGINE = r"""
import struct, sys, tomllib
from pathlib import Path

cfg = tomllib.loads(Path(sys.argv[sys.argv.index('--config_file') + 1]).read_text(encoding='utf-8'))
out = Path(cfg['output_dir'])
out.mkdir(parents=True, exist_ok=True)
value = cfg['seed']
if cfg.get('resume'):
    stored = (Path(cfg['resume']) / 'model.safetensors').read_bytes()
    value = struct.unpack('<f', stored[-4:])[0] + 1
header = b'{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
header += b' ' * (-len(header) % 8)
weight = struct.pack('<Q', len(header)) + header + struct.pack('<f', value)
(out / (cfg['output_name'] + '.safetensors')).write_bytes(weight)
if cfg.get('save_state') and not cfg.get('resume'):
    state = out / 'weights-step00000001-state'
    state.mkdir()
    (state / 'model.safetensors').write_bytes(weight)
    (state / 'optimizer.bin').write_bytes(b'optimizer')
    (state / 'random_states_0.pkl').write_bytes(b'rng')
    sys.exit(1)  # Failed run with a complete save point available.
"""
SEED_11_HASH = "263efb9cc6351dab3eb07600f6e8390dce256771282ba771446f3c2025bb0f77"
SEED_22_HASH = "f4970b760d41ec757b8625e7f7a03f682c304e6b5e5efa586416149162a05ed9"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SameNameRunTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ServiceTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.training = self.fixture.training
        self.draft = dict(self.fixture.draft(self.fixture.ready(script=CPU_ENGINE)), output_name="same")

    def submit(self, name, seed, **params):
        return self.training.submit(dict(self.draft, name=name, params={**self.draft["params"], "seed": seed, **params}))

    def execute(self, task, expected_state="succeeded"):
        key = task["task_id"]
        Worker(self.fixture.store).execute(self.training.task(key))
        actual = self.training.task(key)
        self.assertEqual(actual["state"], expected_state, actual.get("error_summary"))
        checkpoints = [a for a in self.training.artifacts(key) if a["kind"] == "checkpoint"]
        self.assertEqual(len(checkpoints), 1)
        checkpoint = checkpoints[0]
        self.assertTrue(checkpoint["complete"])
        self.assertEqual(checkpoint["task_id"], key)
        self.assertEqual(Path(checkpoint["path"]), Path(task["output_dir"]) / "same.safetensors")
        return checkpoint

    def test_same_name_runs_keep_both_weights_after_first_record_deleted(self):
        first = self.submit("portrait", 11)
        second = self.submit("portrait", 22)
        for task in (first, second):
            self.assertEqual(self.training.task(task["task_id"])["state"], "queued")
            self.assertEqual(Path(task["output_dir"]).name, "portrait--" + task["task_id"])
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        original = self.execute(first)
        before = sha(original["path"])
        self.assertEqual(before, SEED_11_HASH)
        latest = self.execute(second)
        self.assertEqual(sha(latest["path"]), SEED_22_HASH)
        self.assertEqual(sha(original["path"]), before)
        self.assertNotEqual(original["artifact_id"], latest["artifact_id"])
        self.assertEqual(self.training.artifacts(first["task_id"])[0]["artifact_id"], original["artifact_id"])
        self.training.delete(first["task_id"])
        historical = self.training.artifacts(first["task_id"])[0]
        self.assertEqual(historical["artifact_id"], original["artifact_id"])
        self.assertEqual(historical["path"], original["path"])
        self.assertEqual(sha(historical["path"]), before)
        self.assertEqual(sha(self.training.artifacts(second["task_id"])[0]["path"]), SEED_22_HASH)

    def test_colliding_windows_task_names_create_distinct_folders(self):
        left = self.submit("portrait:a", 11)
        right = self.submit("portrait?a", 22)
        self.assertEqual(Path(left["output_dir"]).name, "portrait_a--" + left["task_id"])
        self.assertEqual(Path(right["output_dir"]).name, "portrait_a--" + right["task_id"])
        first = self.execute(left)
        second = self.execute(right)
        self.assertEqual(sha(first["path"]), SEED_11_HASH)
        self.assertEqual(sha(second["path"]), SEED_22_HASH)

    def test_resumed_child_reads_parent_save_point_and_writes_a_new_weight(self):
        parent = self.submit("portrait", 11, save_state=True)
        original = self.execute(parent, expected_state="failed")
        info = self.training.resume_info(parent["task_id"])
        self.assertTrue(info["supported"], info["reason"])
        source = Path(info["sources"][0]["path"])
        parent_files = {p: sha(p) for p in Path(parent["output_dir"]).rglob("*") if p.is_file()}
        child = self.training.resume(parent["task_id"], str(source))
        self.assertEqual(child["resumed_from"], parent["task_id"])
        self.assertEqual(Path(child["output_dir"]),
                         Path(parent["output_dir"]).parent / ("portrait--" + child["task_id"]))
        resumed = self.execute(child)
        self.assertEqual(sha(original["path"]), SEED_11_HASH)
        self.assertNotEqual(sha(resumed["path"]), sha(original["path"]))
        self.assertEqual({p: sha(p) for p in Path(parent["output_dir"]).rglob("*") if p.is_file()}, parent_files)
