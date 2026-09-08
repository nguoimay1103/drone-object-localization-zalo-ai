import ast
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "06-inference-main.ipynb"


def notebook_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and not any(
            line.lstrip().startswith(("!", "%"))
            for line in "".join(cell.get("source", [])).splitlines()
        )
    )


SOURCE = notebook_source()
TREE = ast.parse(SOURCE)
NODES = {
    node.name: ast.get_source_segment(SOURCE, node)
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
}


def cache_namespace():
    namespace = {"os": os, "gzip": gzip, "json": json}
    for name in [
        "read_json",
        "resolve_precomputed_cache_path",
        "read_cache_json",
        "read_compressed_cache",
        "cache_signature_mismatch_keys",
    ]:
        exec(NODES[name], namespace)
    return namespace


class CacheLoadingTests(unittest.TestCase):
    def test_valid_gzip_tmp_is_supported(self):
        ns = cache_namespace()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "candidate_scores.json.gz.tmp"
            payload = {"complete": True, "signature": {"version": 1}}
            with gzip.open(cache_path, "wt", encoding="utf-8") as stream:
                json.dump(payload, stream)
            resolved = ns["resolve_precomputed_cache_path"](
                str(cache_path), "candidate cache", search_root=directory
            )
            self.assertEqual(Path(resolved), cache_path)
            self.assertEqual(
                ns["read_compressed_cache"](resolved, "candidate cache"), payload
            )

    def test_final_name_can_resolve_existing_tmp_variant(self):
        ns = cache_namespace()
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "tile.json.gz"
            tmp_path = Path(str(final_path) + ".tmp")
            with gzip.open(tmp_path, "wt", encoding="utf-8") as stream:
                json.dump({"complete": True}, stream)
            resolved = ns["resolve_precomputed_cache_path"](
                str(final_path), "SAHI tile cache", search_root=directory
            )
            self.assertEqual(Path(resolved), tmp_path)

    def test_plain_json_with_gzip_tmp_suffix_is_supported(self):
        ns = cache_namespace()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "candidate_scores.json.gz.tmp"
            payload = {"complete": True, "signature": {"version": 1}}
            cache_path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                ns["read_cache_json"](str(cache_path), "candidate cache"), payload
            )

    def test_corrupt_tmp_reports_atomic_write_hint(self):
        ns = cache_namespace()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "candidate.json.gz.tmp"
            cache_path.write_bytes(b"incomplete")
            with self.assertRaisesRegex(RuntimeError, "atomic write was interrupted"):
                ns["read_compressed_cache"](str(cache_path), "candidate cache")

    def test_missing_precomputed_path_fails_early(self):
        ns = cache_namespace()
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json.gz"
            with self.assertRaisesRegex(FileNotFoundError, "no file exists"):
                ns["resolve_precomputed_cache_path"](
                    str(missing), "candidate cache", search_root=directory
                )


if __name__ == "__main__":
    unittest.main()
