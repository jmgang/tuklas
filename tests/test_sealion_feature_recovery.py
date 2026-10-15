"""Exercise real v3 recovery function using memory-only storage and real NumPy."""
import ast
from hashlib import sha256
import io
import json
import os
from pathlib import Path, PurePosixPath
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class MemoryPath:
    def __init__(self, name, files):
        self.name_path = PurePosixPath(name)
        self.files = files

    def __truediv__(self, part):
        return MemoryPath(str(self.name_path / part), self.files)

    @property
    def name(self):
        return self.name_path.name

    @property
    def stem(self):
        return self.name_path.stem

    def with_suffix(self, suffix):
        return MemoryPath(str(self.name_path.with_suffix(suffix)), self.files)

    def mkdir(self, exist_ok=False):
        pass

    def read_text(self):
        return self.files[str(self.name_path)].decode()

    def open(self, mode):
        key = str(self.name_path)
        if mode == "xb":
            if key in self.files:
                raise FileExistsError(key)
            files = self.files
            class Writer(io.BytesIO):
                def close(self):
                    files[key] = self.getvalue()
                    super().close()
            return Writer()
        return io.BytesIO(self.files[key])

    def glob(self, pattern):
        return [MemoryPath(k, self.files) for k in self.files
                if PurePosixPath(k).parent == self.name_path and PurePosixPath(k).match(pattern)]

    def __lt__(self, other):
        return str(self.name_path) < str(other.name_path)


class FeatureRecoveryTests(unittest.TestCase):
    def run_recovery(self, corrupt_bytes=False, corrupt_alignment=False):
        rows = [{"text": f"literal text {i}"} for i in range(6546)]
        bodies, specs = {}, {}
        for start in range(0, len(rows), 1024):
            texts = [r["text"] for r in rows[start:start+1024]]
            if corrupt_alignment and start == 0:
                texts[0] = "unapproved changed text"
            out = io.BytesIO()
            np.save(out, np.ones((len(texts),1024), dtype=np.float32), allow_pickle=False)
            for extension, payload in (("json", json.dumps(texts).encode()), ("npy", out.getvalue())):
                filename = f"features-{start:08d}.{extension}"
                bodies[filename] = payload
                specs[filename] = {"key":filename, "bytes":len(payload), "sha256":sha256(payload).hexdigest()}
        if corrupt_bytes:
            value = bodies["features-00000000.npy"]
            bodies["features-00000000.npy"] = value[:-1] + bytes([value[-1]^1])
        layout = b"original layout bytes"
        manifest = {"bucket":"memory", "approved_features": {"source_job":"preserved-source",
                    "layout_sha256":sha256(layout).hexdigest(), "files":specs}}
        files = {"manifest":json.dumps(manifest).encode(), "work/approved_articles-training-layout.jsonl":layout}
        class S3:
            def get_object(self, Bucket, Key):
                return {"Body":io.BytesIO(bodies[Key]), "ContentLength":len(bodies[Key])}
        class Preservation:
            s3 = S3()
            events = []
            def tree(self, directory, relative):
                self.preserved = len(list(directory.glob("*")))
            def event(self, stage, **kwargs):
                self.events.append({"stage":stage, **kwargs})
        source = ast.parse((ROOT/"scripts/aws_sealion_train_repair_v3.py").read_text())
        fn = next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=="restore_approved_features")
        namespace = {"Path":lambda name: MemoryPath(name,files), "WORK":MemoryPath("work",files),
                     "os":os, "json":json, "sha256":sha256,
                     "digest":lambda path: sha256(files[str(path.name_path)]).hexdigest()}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),"real-v3-recovery","exec"),namespace)
        real_load = np.load
        preservation = Preservation()
        with patch.dict(os.environ,{"SEALION_RECOVERY_MANIFEST":"manifest"}), \
             patch.object(np,"load",lambda path,**kwargs: real_load(io.BytesIO(files[str(path.name_path)]),**kwargs)):
            namespace["restore_approved_features"](rows,preservation)
        self.assertEqual(preservation.preserved,14)
        self.assertEqual(preservation.events[-1]["rows_done"],6546)

    def test_real_recovery_verifies_complete_features_without_disk(self):
        self.run_recovery()

    def test_byte_corruption_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"checksum mismatch"):
            self.run_recovery(corrupt_bytes=True)

    def test_rehashed_but_changed_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"recovery text"):
            self.run_recovery(corrupt_alignment=True)
